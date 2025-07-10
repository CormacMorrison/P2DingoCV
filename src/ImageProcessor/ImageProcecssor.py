import cv2 as cv
import numpy as np
from typing import Final, Tuple
from sklearn.cluster import DBSCAN


class ImageProcessor:
    def __init__(self, height: int, width: int) -> None:
        self.height: int = height
        self.width: int = width
        self.frameCount: int = 0

        #Parmaters dependant on resolution
        params = self.normalize_params(height, width)
        self.kernelSize: int = params["kernelSize"]
        self.sigma: float = params["sigma"]
        self.clipLimit: float = params["clipLimit"]
        self.tileGridSize: tuple[int, int] = params["tileGridSize"]
        self.edgeSlideFactor: float = params["edgeSlideFactor"]
        self.edgeWindowSize: float = params["edgeWindowSize"]
        self.rho: int = params["rho"]
        self.theta: float = params["theta"]
        self.threshold: int = params["threshold"]
        self.minLineLength: int = params["minLineLength"]
        self.maxLineGap: int = params["maxLineGap"]
        self.lineCount: int = params["lineCount"] 
        self.lineBuffer: int = params["lineBuffer"]

    def normalize_params(self, height: int, width: int) -> dict:
        # 2202.90717008  for 1080p
        diag: float = (height**2 + width**2) ** 0.5
        kernelSize: int = max(3, int(min(height, width) * 0.005) // 2 * 2 + 1)  # odd
        sigma: float = diag * 0.010
        edgeSlideFactor: float = 99 # auto adjusts until it finds lineCount number of lines
        edgeWindowSize: float = 0.44
        clipLimit: float = 2.0
        tileGridSize: tuple[int, int] = (max(1, width // 64), max(1, height // 64))
        rho: int = 1
        theta: float = np.pi / 180
        threshold: int = int(diag * 0.05)
        minLineLength: int = int(diag * 0.03)
        maxLineGap: int = int(diag * 0.04)
        lineCount: int = 40
        lineBuffer: int = 10

        return {
            "kernelSize": kernelSize,
            "sigma": sigma,
            "edgeSlideFactor": edgeSlideFactor,
            "edgeWindowSize": edgeWindowSize,
            "clipLimit": clipLimit,
            "tileGridSize": tileGridSize,
            "rho": rho,
            "theta": theta,
            "threshold": threshold,
            "minLineLength": minLineLength,
            "maxLineGap": maxLineGap,
            "lineCount": lineCount,
            "lineBuffer": lineBuffer
        }


    def preProcess(self, frame: np.ndarray) -> np.ndarray:
        # kernelSize: int = 5
        # sigma: int = 10
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #increase contrast -> doesn't help
        # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # enhanced = clahe.apply(grey)
        return cv.GaussianBlur(grey, (self.kernelSize, self.kernelSize), self.sigma)

    def edgeDetection(self, frame: np.ndarray) -> np.ndarray:
        # Normalising the paramaters for median light level 
        sigma: float = 0.33
        v: float = self.edgeSlideFactor * float(np.median(frame))
        lower: int = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        edges = cv.Canny(frame, lower, upper)

        return edges
    
    def lineDetection(self, frame: np.ndarray) -> np.ndarray | None:
        # rho: float = 1
        # theta: float = np.pi / 180
        # threshold: int = 160
        # minLineLength: float = 50
        # maxLineGap: float = 50
        lines: np.ndarray | None =  cv.HoughLinesP(
            frame,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.minLineLength,
            maxLineGap=self.maxLineGap
        )
        if (lines is None):
            self.edgeSlideFactor -= 1

        elif (len(lines) > self.lineCount + self.lineBuffer):
            self.edgeSlideFactor += 0.05
        
        elif (len(lines) <= self.lineCount + self.lineBuffer):
            self.edgeSlideFactor -= 0.05
        
        if (lines is not None):
            lens = len(lines)
        else:
            lens = 0
        # print(f"||||len ={lens}||||")
        # print("||||start||||") 
        # print(lines) 
        # print("||||end||||")
        
        return lines

    def splitByAngle(self, lines, threshold_deg=10):
        """
        Splits Hough lines into horizontal and vertical groups based on angle.

        Args:
            lines (np.ndarray): Array of shape (N, 1, 4) or (N, 4), format [x1, y1, x2, y2]
            threshold_deg (float): Degrees within 0° or 90° to consider "horizontal" or "vertical"

        Returns:
            tuple:
                horizontal_lines: lines close to 0° (horizontal)
                vertical_lines: lines close to 90° (vertical)
        """
        if (lines is None or len(lines) < 30 or len(lines) > 70):
            return None

        # Reshape if needed (HoughLinesP returns with double wrapped arrays)
        lines = lines.reshape(-1, 4)
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

        # Calculate deltas
        dx = x2 - x1
        dy = y2 - y1

        # Compute angles in degrees
        angles_deg = np.degrees(np.arctan2(dy, dx))
        angles_deg = np.abs(angles_deg)
        angles_deg = np.where(angles_deg > 90, 180 - angles_deg, angles_deg)

        # Create masks
        horizontal_mask = angles_deg <= threshold_deg
        vertical_mask = np.abs(angles_deg - 90) <= threshold_deg

        # Filter
        horizontal_lines = lines[horizontal_mask]
        vertical_lines = lines[vertical_mask]

        return horizontal_lines, vertical_lines

    def determineAngles(self, lines: np.ndarray) -> tuple[float, float, float]:
        
        return 0, 0 ,0
    
    def drawLines(self, frame, lines: np.ndarray | None) -> np.ndarray:
        if (lines is None):
            return frame
                # Prepare BGR output from binary input
        output: np.ndarray = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        # Draw detected lines in green
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return output

    def draw_vanishing_points(self, image, vp1, vp2, color1=(0, 0, 255), color2=(255, 0, 0), radius=100):
        img_vis = image.copy()

        vp1_int = tuple(np.round(vp1).astype(int))
        vp2_int = tuple(np.round(vp2).astype(int))

        # Draw circles for vanishing points
        cv.circle(img_vis, vp1_int, radius, color1, -1)
        cv.circle(img_vis, vp2_int, radius, color2, -1)

        cv.imshow("Vanish", img_vis)

        return img_vis



    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process the frame by applying preprocessing then edge detection.
        """
        self.frameCount+=1
        # print(self.frameCount)
        blurred = self.preProcess(frame)
        edges = self.edgeDetection(blurred)
        lines = self.lineDetection(edges)
        output = self.drawLines(edges, lines)
        return output 
    
    

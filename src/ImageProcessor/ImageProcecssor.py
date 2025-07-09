import cv2 as cv
import numpy as np
from typing import Final
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
        cv.imshow("preedge", frame)
        sigma: float = 0.33
        v: float = self.edgeSlideFactor * float(np.median(frame))
        lower: int = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        edges = cv.Canny(frame, lower, upper)

        cv.imshow("edges", edges)
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
    import numpy as np


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
        if lines is None or len(lines) == 0:
            return np.empty((0, 4)), np.empty((0, 4))

        # Reshape if needed
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

   
    
    def vanishingLineGrouping(self, lines: np.ndarray | None):
        
        # Extract endpoints
        angleThrehold = 10
        
        if (lines is None):
            return None
        elif (len(lines) < 30 or len(lines) > 70):
            return None
        
        x1, y1, x2, y2 = lines[:, 0, 0], lines[:, 0, 1], lines[:, 0, 2], lines[:, 0, 3]
        
        # convert to homogenous equation
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        homoLines = np.stack([a, b, c], axis=1)
        
        #compute intersections 
        n = len(homoLines)
        i, j = np.triu_indices(n, k=1)
        L1, L2 = homoLines[i], homoLines[j]
        intersections = np.cross(L1, L2)
        valid = np.abs(intersections[:, 2]) > 1e-6
        points = intersections[valid]
        points = points[:, :2] / points[:, 2:3]
        
        #Find the two vanishing points
        db = DBSCAN(eps=100, min_samples=2).fit(points)
        labels = db.labels_
    
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        top2 = unique[np.argsort(-counts)[:2]]
        if (len(top2) == 0):
            print("You Should Never end up here")
            return None
        vp1 = points[labels == top2[0]].mean(axis=0)
        vp2 = points[labels == top2[1]].mean(axis=0)
        vanishing_points = np.stack([vp1, vp2])
        
        midpoints = np.stack([(x1 + x2) / 2, (y1 + y2) / 2], axis=1)
        directions = np.stack([x2 - x1, y2 - y1], axis=1)
        directions /= (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6)

        vpVecs = vanishing_points[None, :, :] - midpoints[:, None, :]
        vpVecs /= np.linalg.norm(vpVecs, axis=2, keepdims=True) + 1e-6

        cosTheta = np.einsum('nd,ndk->nk', directions, vpVecs)
        angles = np.degrees(np.arccos(np.clip(cosTheta, -1.0, 1.0)))
        
        min_angle = np.min(angles, axis=1)
        valid_lines = min_angle < angleThrehold 
        bestVp = np.argmin(angles, axis=1)

        group1 = lines[(bestVp == 0) & valid_lines]
        group2 = lines[(bestVp == 1) & valid_lines]
        
        return group1, group2

        


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
    
    

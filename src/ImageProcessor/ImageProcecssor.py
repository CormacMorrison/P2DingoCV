import cv2 as cv
import numpy as np
from typing import Final, Tuple
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

_NO_PANEL_DETECTED = -1


class ImageProcessor:
    def __init__(self, height: int, width: int) -> None:
        self.height: int = height
        self.width: int = width
        self.frameCount: int = 0

        # Parmaters dependant on resolution
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
        edgeSlideFactor: float = (
            3  # auto adjusts until it finds lineCount number of lines
        )
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
            "lineBuffer": lineBuffer,
        }

    def preProcess(self, frame: np.ndarray) -> np.ndarray:
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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
        lines: np.ndarray | None = cv.HoughLinesP(
            frame,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.minLineLength,
            maxLineGap=self.maxLineGap,
        )
        # Infinite l
        changeFactor = 1
        if lines is None:
            self.edgeSlideFactor -= changeFactor
            return lines
        
        #prevent falling into negatives
        elif self.edgeSlideFactor < 0:
            self.edgeSlideFactor = 0

        elif len(lines) > self.lineCount + self.lineBuffer:
            self.edgeSlideFactor += 0.05

        elif len(lines) <= self.lineCount + self.lineBuffer:
            self.edgeSlideFactor -= 0.05
        
        print(self.edgeSlideFactor)

        return lines

    def angleClusters(self, lines: np.ndarray | None):
        if lines is None or len(lines) < 30 or len(lines) > 70:
            return None

        # Reshape if needed (HoughLinesP returns with double wrapped arrays)
        lines = lines.reshape(-1, 4)
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

        # calculate deltas
        dx = x2 - x1
        dy = y2 - y1

        anglesDeg = np.degrees(np.arctan2(dy, dx)) % 180

        output = self.KmeansCluster(anglesDeg)

        return output

    def KmeansCluster(self, angles: np.ndarray, maxIter=300, randomState=None):
        angles = np.asarray(angles, dtype=float)
        if np.any((angles < 0) | (angles > 180)):
            raise ValueError("All angles must be in the range [0, 180].")

        theta = np.deg2rad(angles)

        pts = np.column_stack([np.cos(theta), np.sin(theta)])

        km = KMeans(n_clusters=2, max_iter=maxIter, random_state=randomState)
        km.fit(pts)
        labels = km.labels_

        # Convert cluster centres back into angles:
        cx, cy = km.cluster_centers_[:, 0], km.cluster_centers_[:, 1]
        center_theta = np.arctan2(cy, cx)
        # atan2 returns in (–π, π]; map to [0, 2π):
        center_theta = np.mod(center_theta, 2 * np.pi)
        # Map back to [0, 180) degrees:
        centersDeg = center_theta * (180.0 / np.pi)

        return pts, labels, km.cluster_centers_, centersDeg

    def process(self, frame: np.ndarray) -> Tuple[float, float]:
        blurred = self.preProcess(frame)
        edges = self.edgeDetection(blurred)
        lines = self.lineDetection(edges)
        kmeans = self.angleClusters(lines)
        if kmeans is not None:
            # 4th return is the centre of clusters in degrees from the positve x axis
            return kmeans[3][0], kmeans[3][1]
        return _NO_PANEL_DETECTED, _NO_PANEL_DETECTED

    ############### Visualisation Functions ###########################
    def processFrame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process the frame by applying preprocessing then edge detection.
        """
        blurred = self.preProcess(frame)
        edges = self.edgeDetection(blurred)
        lines = self.lineDetection(edges)
        output = self.showAngleBins(edges, lines)
        if output is not None:
            return output
        return edges

    def showAngleBins(self, frame, lines: np.ndarray | None):
        if lines is None:
            return frame
        kmeans = self.angleClusters(lines)
        if kmeans is None:
            return frame
        _, labels, _, centers_deg = kmeans

        bin1 = []
        bin2 = []

        # stops the visualiser from pulsing
        binSwapVal = 1
        if centers_deg[0] > centers_deg[1]:
            binSwapVal = 0

        for i, line in enumerate(lines):
            if labels[i] == binSwapVal:
                bin1.append(line)
            else:
                bin2.append(line)
        frame = self.drawLines(frame, np.array(bin1), "g")
        return self.drawLines(frame, np.array(bin2), "r")

    def drawLines(
        self, frame: np.ndarray, lines: np.ndarray | None, colour: str
    ) -> np.ndarray:
        if lines is None:
            return frame
            # Prepare BGR output from binary input
        b = 0
        g = 0
        r = 0
        if colour == "g":
            g = 255
        elif colour == "r":
            r = 255
        elif colour == "b":
            b = 255
        if len(frame.shape) == 2:
            output = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            output = frame.copy()
        # Draw detected lines in green
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv.line(output, (x1, y1), (x2, y2), (b, g, r), 2)

        return output

    def visualize(self, angles: np.ndarray):
        pts, labels, centers, centers_deg = self.KmeansCluster(angles)

        # Ensure pts and centers are 2D arrays with shape (N, 2)
        pts = np.asarray(pts).reshape(-1, 2)
        centers = np.asarray(centers).reshape(-1, 2)

        plt.figure(figsize=(6, 6))
        plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="viridis", edgecolor="k")
        plt.scatter(
            centers[:, 0], centers[:, 1], marker="x", s=100, linewidths=2, color="red"
        )

        # Add circular guide
        ax = plt.gca()
        ax.add_patch(Circle((0, 0), 1.0, fill=False, linestyle="--", color="gray"))

        plt.axis("equal")
        plt.title("Circular K‑Means Clustering on Angles")
        plt.xlabel("Cos(theta)")
        plt.ylabel("Sin(theta)")
        plt.grid(True)
        plt.show()
        plt.savefig("angle_clusters.png")

        print("Cluster centers (deg):", centers_deg)

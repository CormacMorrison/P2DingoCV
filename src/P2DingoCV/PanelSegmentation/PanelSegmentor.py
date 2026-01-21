from logging import config
import cv2 as cv
from matplotlib import lines
import numpy as np
from typing import Dict, Final, List, Tuple
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import stats
from ..Types.Types import *
from numpy.typing import NDArray
import logging
from pathlib import Path
from ..Util.MiscUtil import MiscUtil
from ..Util.VisualUtil import VisualUtils
import json

_NO_PANEL_DETECTED = -1


class PanelSegmentor:
    def __init__(
        self, height: int, width: int, outputPath: str, config: str | None = None
    ) -> None:

        self.height: int = height
        self.width: int = width
        self.frameCount: int = 0
        self.outputPath: str = outputPath
        self.visualUtil: VisualUtils = VisualUtils(outputPath)
        if config is None:
            self.config = {}
        else:
            self.config: Dict = MiscUtil.loadConfig(config)

        # Parmaters dependant on resolution
        params = self.normalize_params(height, width)
        self.kernelSize: int = params["kernelSize"]
        self.sigma: float = params["sigma"]
        self.clipLimit: float = params["clipLimit"]
        self.tileGridSize: tuple[int, int] = params["tileGridSize"]
        self.edgeSlideFactor: float = params["edgeSlideFactor"]
        self.rho: int = params["rho"]
        self.theta: float = params["theta"]
        self.threshold: int = params["threshold"]
        self.minLineLength: int = params["minLineLength"]
        self.maxLineGap: int = params["maxLineGap"]
        self.lineCount: int = params["lineCount"]
        self.lineBuffer: int = params["lineBuffer"]
        self.denoiseLambdaWeight: float = params["denoiseLambdaWeight"]
        self.denoiseMeanKernelSize: int = params["denoiseMeanKernelSize"]
        self.denoiuseGaussianSigma: float = params["denoiuseGaussianSigma"]
        self.denoiseDownsampleFactor: int = params["denoiseDownsampleFactor"]

    def normalize_params(self, height: int, width) -> dict:
        if self.config is None:
            self.config = {}

        # Constants
        diag: float = (height**2 + width**2) ** 0.5
        kernelSize: int = max(5, int(min(height, width) * 0.005) // 2 * 2 + 1)  # odd
        tileGridSize: tuple[int, int] = (max(1, width // 64), max(1, height // 64))

        # Config params
        lineCount = self.config.get("lineCount", 100)
        lineBuffer = self.config.get("lineBuffer", 5)
        denoiseLambdaWeight = self.config.get("denoiseLambdaWeight", 2.0)
        denoiseMeanKernelSize = self.config.get("denoiseMeanKernelSize", 5)
        denoiuseGaussianSigma = self.config.get("denoiuseGaussianSigma", 1.0)
        denoiseDownsampleFactor = self.config.get("denoiseDownsampleFactor", 2)
        edgeSlideFactor = self.config.get("edgeSlideFactor", 5)
        clipLimit = self.config.get("clipLimit", 2.0)
        rho = self.config.get("rho", 1)
        theta = self.config.get("theta", np.pi / 180)
        aspectRatio = self.config.get("aspectRatio", 1.0 / 1.7)

        # Modifiers
        sigmaMultipler = self.config.get("sigmaMultipler", 0.010)
        edgeThresholdMultiplier = self.config.get("edgeThresholdMultiplier", 0.05)
        minLineLengthMutiplier = self.config.get("minLineLengthMutiplier", 0.03)
        maxLineGapMultipler = self.config.get("maxLineGapMultipler", 0.04)

        # 2202.90717008  for 1080p
        sigma: float = diag * sigmaMultipler
        threshold: int = int(diag * edgeThresholdMultiplier)
        minLineLength: int = int(diag * minLineLengthMutiplier)
        maxLineGap: int = int(diag * maxLineGapMultipler)

        return {
            "aspectRatio": aspectRatio,
            "kernelSize": kernelSize,
            "sigma": sigma,
            "edgeSlideFactor": edgeSlideFactor,
            "clipLimit": clipLimit,
            "tileGridSize": tileGridSize,
            "rho": rho,
            "theta": theta,
            "threshold": threshold,
            "minLineLength": minLineLength,
            "maxLineGap": maxLineGap,
            "lineCount": lineCount,
            "lineBuffer": lineBuffer,
            "denoiseLambdaWeight": denoiseLambdaWeight,
            "denoiseMeanKernelSize": denoiseMeanKernelSize,
            "denoiuseGaussianSigma": denoiuseGaussianSigma,
            "denoiseDownsampleFactor": denoiseDownsampleFactor,
        }

    def resetParameters(self, height: int, width: int, logPath: str) -> None:
        self.height = height
        self.width = width
        params = self.normalize_params(height, width)
        self.aspectRatio: float = params["aspectRatio"]
        self.kernelSize: int = params["kernelSize"]
        self.sigma: float = params["sigma"]
        self.clipLimit: float = params["clipLimit"]
        self.tileGridSize: tuple[int, int] = params["tileGridSize"]
        self.edgeSlideFactor: float = params["edgeSlideFactor"]
        self.rho: int = params["rho"]
        self.theta: float = params["theta"]
        self.threshold: int = params["threshold"]
        self.minLineLength: int = params["minLineLength"]
        self.maxLineGap: int = params["maxLineGap"]
        self.lineCount: int = params["lineCount"]
        self.lineBuffer: int = params["lineBuffer"]
        self.denoiseLambdaWeight: float = params["denoiseLambdaWeight"]
        self.denoiseMeanKernelSize: int = params["denoiseMeanKernelSize"]
        self.denoiuseGaussianSigma: float = params["denoiuseGaussianSigma"]
        self.denoiseDownsampleFactor: int = params["denoiseDownsampleFactor"]

    def preProcess(
        self,
        frame: Frame,
        lambdaWeight: float,
        meanKernelSizeL: int,
        gaussianSigma: float,
        downsampleFactor: int,
    ) -> Frame:
        """
        Preprocess an input image frame using guided filtering and edge enhancement,
        followed by grayscale conversion and Gaussian denoising.

        This pipeline enhances structural edges while suppressing noise, making the
        output suitable for downstream tasks such as line detection, grid extraction,
        or feature analysis.

        The processing steps are:
            1. Normalize the input frame to float32 in the range [0, 1].
            2. Split into color channels.
            3. Downsample each channel.
            4. Generate a guidance image using mean filtering.
            5. Upsample the guidance image and apply guided filtering.
            6. Perform edge enhancement using: Ai = Qi + λ (Ii − Qi).
            7. Merge enhanced channels.
            8. Convert to grayscale.
            9. Apply Gaussian blur for final denoising.

        Args:
            frame (Frame):
                Input image frame in BGR format. Expected as a NumPy array
                of dtype uint8 or float32 with shape (H, W, 3).

            lambdaWeight (float):
                Edge enhancement strength λ. Higher values increase edge contrast.

            meanKernelSizeL (int):
                Kernel size for the mean filter used to generate the guidance image.

            gaussianSigma (float):
                Standard deviation for the final Gaussian denoising filter.

            downsampleFactor (int):
                Downsampling factor applied before guidance image computation.

        Returns:
            Frame:
                A single-channel (grayscale) denoised image frame suitable for
                further processing.

        Raises:
            ValueError:
                If the input frame has an unexpected shape or dtype.

        Notes:
            - Requires OpenCV's ximgproc module for guided filtering.
            - Kernel sizes should be positive odd integers for best results.
        """
        lambda_weight = lambdaWeight  # Edge enhancement strength (λ)
        mean_kernel_size = meanKernelSizeL  # Mean filter size for guidance image
        gaussian_sigma = gaussianSigma  # Gaussian filter strength
        downsample_factor = downsampleFactor  # Downsampling ratio
        # Convert to float32 and normalize to [0, 1]
        I: FloatFrame = frame.astype(np.float32) / 255.0

        # Split into channels
        I_channels: ChannelList = cv.split(I)

        enhanced_channels: List[Frame] = []

        # Guided filtering and edge enhancement ---
        for Ii in I_channels:
            # Downsample
            small: Channel = cv.resize(
                Ii,
                None,
                fx=1 / downsample_factor,
                fy=1 / downsample_factor,
                interpolation=cv.INTER_AREA,
            )

            # Guidance image Gi = mean filter of downsampled image
            Gi: Channel = cv.blur(small, (mean_kernel_size, mean_kernel_size))

            # Upsample Gi back to original size for guided filtering
            Gi_up: Channel = cv.resize(
                Gi, (Ii.shape[1], Ii.shape[0]), interpolation=cv.INTER_LINEAR
            )

            # Guided filter using Gi as guide
            # OpenCV provides ximgproc.guidedFilter (if available)
            guided: Channel = cv.ximgproc.guidedFilter(
                guide=Gi_up, src=Ii, radius=8, eps=1e-3
            )

            Qi: Channel = guided

            # Edge enhancement Ai = Qi + λ (Ii − Qi)
            Ai: Channel = Qi + lambda_weight * (Ii - Qi)
            enhanced_channels.append(np.clip(Ai, 0, 1))

        #  Merge enhanced channels
        A: Frame = cv.merge(enhanced_channels)

        # Convert to grayscale for final denoising
        A_gray: Frame = cv.cvtColor((A * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)

        # Gaussian filter to suppress noise
        A_denoised: Frame = cv.GaussianBlur(
            A_gray, (self.kernelSize, self.kernelSize), self.sigma
        )

        return A_denoised

    def edgeDetection(self, frame: Frame) -> Frame:
        """
        Perform adaptive Canny edge detection based on the median intensity
        of the input frame.

        The lower and upper Canny thresholds are computed through a loop that converges on
        a number of detected lines close to the expected line count.

        Thresholds are computed as:
            v = edgeSlideFactor * median(frame)
            lower = max(0, (1 - σ) * v)
            upper = min(255, (1 + σ) * v)

        Args:
            frame (Frame):
                Input image frame, expected to be a single-channel (grayscale)
                or 3-channel image in uint8 format.

        Returns:
            Frame:
                A single-channel binary edge map produced by the Canny detector.

        Notes:
            - Uses OpenCV's Canny edge detector.
        """
        sigma: float = 0.33
        v: float = self.edgeSlideFactor * float(np.median(frame))
        lower: int = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        edges = cv.Canny(frame, lower, upper)

        return edges

    def lineDetection(
        self, frame: Frame, expectedLines: int = 100
    ) -> tuple[NDArray[np.integer], Frame] | None:
        """
        Detect line segments in an image using adaptive Canny + Hough Transform.

        This method iteratively adjusts the edge detection sensitivity
        (`edgeSlideFactor`) to converge on an expected number of detected
        line segments. It is designed to be robust across varying contrast
        and noise conditions.

        The process is:
            1. Apply adaptive Canny edge detection.
            2. Detect line segments using probabilistic Hough transform.
            3. Adjust edge sensitivity until the number of detected lines
            falls within the acceptable buffer range.

        Args:
            frame (Frame):
                Input image frame (typically grayscale) used for line detection.

            expectedLines (int, optional):
                Target number of line segments to detect. The algorithm
                adapts thresholds to approach this value. Defaults to 100.

        Returns:
            tuple[np.ndarray, Frame] | None:
                A tuple containing:
                    - lines: NumPy array of shape (N, 4), where each row is
                    (x1, y1, x2, y2) for a detected line segment.
                    - edges: The binary edge image used for Hough detection.

                Returns None only if no valid detection occurs (normally a
                ValueError is raised instead).

        Raises:
            ValueError:
                If no lines are detected after exhausting all threshold adjustments,
                or if the detection logic enters an invalid state.

        Notes:
            - Uses OpenCV's cv.HoughLinesP for line detection.
            - The parameter `edgeSlideFactor` is automatically tuned per frame.
            - `lineBuffer` defines the acceptable deviation from `expectedLines`.
        """
        detectionSuccess: bool = False
        self.edgeSlideFactor = 5.0  # reset for each frame
        prevBig: bool = False
        prevSmall: bool = False
        self.logger.info(f"Starting line detection with target of {expectedLines} lines.")
        while not detectionSuccess:
            increment: float = 0.03
            edges: Frame = self.edgeDetection(frame)
            lines: np.ndarray | None = cv.HoughLinesP(
                edges,
                rho=self.rho,
                theta=self.theta,
                threshold=self.threshold,
                minLineLength=self.minLineLength,
                maxLineGap=self.maxLineGap,
            )
            self.logger.info(f"Edge slide factor: {self.edgeSlideFactor}")
            if lines is None:
                self.edgeSlideFactor -= 0.25
                if self.edgeSlideFactor < 0:
                    detectionSuccess = False
                    raise ValueError("No lines detected.")
            # ideal case
            elif (
                expectedLines - self.lineBuffer
                <= len(lines)
                <= expectedLines + self.lineBuffer
            ):
                detectionSuccess = True
                prevBig = False
                prevSmall = False
                increment = 0.03
            # prevent falling into negatives
            elif self.edgeSlideFactor < 0:
                self.edgeSlideFactor = 0

            elif len(lines) > expectedLines + self.lineBuffer:
                if (prevSmall == True):
                    increment /= 2
                    self.edgeSlideFactor += increment
                else:
                    self.edgeSlideFactor += increment
                
                prevBig = True
            elif len(lines) <= max(expectedLines - self.lineBuffer , 10):
                if (prevBig == True):
                    increment /= 2
                    self.edgeSlideFactor -= increment
                else:
                    self.edgeSlideFactor -= increment
                prevSmall = True
            else:
                raise ValueError("Line detection logic error.")

        lines = np.array(lines).reshape(-1, 4)
        self.logger.info(f"Detected {len(lines)} lines.")

        return lines, edges

    def mergeLines(
        self,
        lines: NDArray[np.integer],
        angle_thresh: float = np.deg2rad(5),
        dist_thresh: int = 50,
    ) -> NDArray[np.integer]:
        """
        Merge approximately collinear and nearby line segments into longer "superlines".

        This function groups 2D line segments that have similar orientation (within
        `angle_thresh`) and are spatially close (within `dist_thresh` measured along
        the normal direction). For each group, it fits a single representative line
        using PCA and returns the endpoints of the merged segment.

        Parameters
        ----------
        lines : NDArray[np.integer]
            Array of line segments with shape (N, 4), where each row is
            [x1, y1, x2, y2].
        angle_thresh : float, optional
            Maximum angular difference (in radians) between two lines for them
            to be considered collinear. Default is 5 degrees in radians.
        dist_thresh : int, optional
            Maximum perpendicular distance (in pixels) between two lines for them
            to be considered part of the same group. Default is 50.

        Returns
        -------
        NDArray[np.integer]
            Array of merged line segments with shape (M, 4), where each row is
            [x1, y1, x2, y2] representing the endpoints of a fitted "superline".

        Notes
        -----
        - The algorithm:
        1. Iterates over all input lines.
        2. Groups lines with similar angle and nearby perpendicular offset.
        3. Collects all endpoints from each group.
        4. Fits a line using PCA to find the dominant direction.
        5. Projects points onto that direction and uses min/max projections
            as the merged segment endpoints.
        - Output endpoints are rounded to integer pixel coordinates.
        """

        if lines is None or len(lines) == 0:
            return np.array([]).reshape(0, 4)

        lines = np.array(lines).reshape(-1, 4)
        used: NDArray[np.bool_] = np.zeros(len(lines), dtype=bool)
        merged: List[tuple[int, int, int, int]] = []

        for i in range(len(lines)):
            if used[i]:
                continue

            x1, y1, x2, y2 = lines[i]
            theta1: np.float64 = np.arctan2(y2 - y1, x2 - x1)
            n: NDArray[np.float64] = np.array([-np.sin(theta1), np.cos(theta1)])
            n /= np.linalg.norm(n)
            pts: NDArray[np.float64] = np.array([[x1, y1], [x2, y2]])

            # normal vector for distance checks
            rho1: np.float64 = abs(np.dot([x1, y1], n))

            # Collect lines to merge
            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue
                x3, y3, x4, y4 = lines[j]
                theta2: np.float64 = np.arctan2(y4 - y3, x4 - x3)

                dtheta: float = np.abs(theta1 - theta2)
                dtheta: float = min(dtheta, np.pi - dtheta)
                if dtheta < angle_thresh:
                    rho2: np.float64 = abs(np.dot([x3, y3], n))
                    if np.abs(rho1 - rho2) < dist_thresh:
                        pts: NDArray[np.float64] = np.vstack(
                            (pts, [[x3, y3], [x4, y4]])
                        )
                        used[j] = True

            # Fit a superline using PCA (via NumPy)
            mean: np.float64 = np.mean(pts, axis=0)
            cov: NDArray[np.float64] = np.cov(pts.T).astype(np.float64)
            eigvals: NDArray[np.float64]
            eigvecs: NDArray[np.float64]
            eigvals, eigvecs = np.linalg.eig(cov)
            direction: np.ndarray = eigvecs[:, np.argmax(eigvals)]

            # Project all points onto the direction vector
            projections: np.ndarray = np.dot(pts - mean, direction)
            p1: NDArray[np.float64] = mean + direction * np.min(projections)
            p2: NDArray[np.float64] = mean + direction * np.max(projections)
            p1 = np.round(p1).astype(int)
            p2 = np.round(p2).astype(int)
            merged.append((*p1, *p2))
            used[i] = True

        return np.array(merged)

    def angleClusters(self, lines: NDArray[np.integer] | None, k: int = 2) -> (
        Tuple[
            NDArray[np.float64],
            NDArray[np.integer],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
        | None
    ):
        """
        Cluster line segments by orientation using circular k-means.

        This function takes 2D line segments, computes their orientation angles
        (in degrees), maps them into a circular-safe representation, and clusters
        them into `k` groups using k-means. It is robust to wrap-around effects
        (e.g., near 0°/180°) by using absolute deltas and modulo arithmetic.

        Parameters
        ----------
        lines : NDArray[np.integer] | None
            Array of line segments with shape (N, 4), where each row is
            [x1, y1, x2, y2]. Typically from HoughLinesP. If None or empty,
            the function returns None.
        k : int, optional
            Number of orientation clusters to form. Default is 2.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.integer], NDArray[np.float64], NDArray[np.float64]] | None
            If input is valid, returns a 4-tuple:
            - pts : (N, 2) array of unit-circle coordinates used for clustering
            - labels : (N,) array of cluster indices for each line
            - centersXY : (k, 2) array of cluster centers in unit-circle space
            - centersDeg : (k,) array of cluster center angles in degrees [0, 180)

            Returns None if `lines` is None or empty.

        Notes
        -----
        - Line angles are computed using atan2(|dy|, |dx|) and wrapped to [0, 180)
        to treat lines as undirected.
        - Clustering is performed by `self.KmeansCluster`, which handles the
        circular nature of angular data.
        """
        # Reshape if needed (HoughLinesP returns with double wrapped arrays)
        if lines is None or len(lines) == 0:
            return None
        lines = lines.reshape(-1, 4)
        x1: NDArray[np.integer]
        y1: NDArray[np.integer]
        x2: NDArray[np.integer]
        y2: NDArray[np.integer]
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

        # calculate deltas as abs to solve the wrap around issue
        dx: NDArray[np.float64] = np.abs(x2 - x1)
        dy: NDArray[np.float64] = np.abs(y2 - y1)

        anglesDeg: NDArray[np.float64] = np.degrees(np.arctan2(dy, dx)) % 180

        output: Tuple[
            NDArray[np.float64],
            NDArray[np.integer],
            NDArray[np.float64],
            NDArray[np.float64],
        ] = self.KmeansCluster(anglesDeg, k)

        return output

    def KmeansCluster(
        self, angles: NDArray[np.float64], k: int = 2, maxIter=300, randomState=None
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.integer],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Cluster line orientations (in degrees) using circular k-means on the unit circle.

        This function takes angles in degrees in the range [0, 180], maps them onto
        the unit circle to handle circularity (so 0° and 180° are close), performs
        k-means clustering, and converts the resulting cluster centers back to angles.

        Parameters
        ----------
        angles : np.ndarray
            Array of angles in degrees, expected in the range [0, 180].
        k : int, optional
            Number of clusters. Default is 2.
        maxIter : int, optional
            Maximum number of k-means iterations. Default is 300.
        randomState : Optional[int], optional
            Random seed for reproducibility. Default is None.

        Returns
        -------
        pts : NDArray[np.float64]
            Points on the unit circle corresponding to the input angles, shape (N, 2).
        labels : NDArray[np.integer]
            Cluster labels for each input angle, shape (N,).
        centersXY : NDArray[np.float64]
            Cluster centers in unit-circle (x, y) coordinates, shape (k, 2).
        centersDeg : NDArray[np.float64]
            Cluster centers converted back to angles in degrees in the range [0, 180).

        Raises
        ------
        ValueError
            If any input angle is outside the range [0, 180].
        """
        angles = np.asarray(angles, dtype=float)
        if np.any((angles < 0) | (angles > 180)):
            raise ValueError("All angles must be in the range [0, 180].")

        theta: NDArray[np.float64] = np.deg2rad(angles)

        # Map angles onto unit circle so 0 and 180 are close:
        pts: NDArray[np.float64] = np.column_stack([np.cos(theta), np.sin(theta)])

        km: KMeans = KMeans(n_clusters=k, max_iter=maxIter, random_state=randomState)
        km.fit(pts)
        labels: NDArray[np.integer] = km.labels_

        # Convert cluster centres back into angles:
        cx: NDArray[np.float64]
        cy: NDArray[np.float64]
        cx, cy = km.cluster_centers_[:, 0], km.cluster_centers_[:, 1]
        center_theta: NDArray[np.float64] = np.arctan2(cy, cx)
        # atan2 returns in (–π, π]; map to [0, 2π):
        center_theta: NDArray[np.float64] = np.mod(center_theta, 2 * np.pi)
        # Map back to [0, 180) degrees:
        centersDeg: NDArray[np.float64] = center_theta * (180.0 / np.pi)

        return pts, labels, km.cluster_centers_, centersDeg

    ############### Visualisation Functions ###########################

    def groupAndFilter(
        self,
        frame,
        lines: NDArray[np.integer] | None,
        discardPercentageofMedian: float = 0.8,
    ) -> tuple[
        NDArray[np.integer],
        NDArray[np.integer],
        NDArray[np.integer],
        List[Frame] | None,
    ]:
        """
        Remove noisy/short Hough lines and separate them into horizontal and vertical sets.

        This function performs a full cleanup pipeline on detected line segments:
        1. Discards obvious outliers using k-means clustering.
        2. Iteratively merges nearly parallel and spatially close lines.
        3. Clusters lines by angle into two groups (horizontal vs vertical).
        4. Estimates representative line lengths in each group.
        5. Removes lines shorter than a fraction of the median length.
        6. Returns the filtered horizontal, vertical, and combined line sets.

        Parameters
        ----------
        frame : Any
            Input image/frame associated with the detected lines (used by
            the k-means noise discard step).
        lines : NDArray[np.integer] | None
            Array of line segments of shape (N, 4) in the form
            [x1, y1, x2, y2]. Must contain at least 3 lines.
        discardPercentageofMedian : float, default=0.8
            Fraction of the median line length used as a cutoff. Lines shorter
            than (median_length * discardPercentageofMedian) are discarded
            within each orientation group.

        Returns
        -------
        combinedLines : NDArray[np.integer]
            Array of filtered lines containing both horizontal and vertical
            segments, shape (M, 4).
        npHorizontal : NDArray[np.integer]
            Filtered horizontal lines, shape (H, 4).
        npVertical : NDArray[np.integer]
            Filtered vertical lines, shape (V, 4).
        kmeansVisuals : List[Frame]
            Visualizations of the k-means clustering step, showing lines
            colored by cluster.

        Raises
        ------
        ValueError
            If there are not enough input lines, if k-means discarding fails,
            if angle clustering fails, or if no horizontal/vertical lines remain
            after filtering.

        Notes
        -----
        - Line orientation is determined via angle clustering into exactly two
          bins (horizontal-like and vertical-like).
        - Length thresholds are computed separately for horizontal and vertical
          groups using the median of the top longest lines in each group.
        """
        if lines is None or len(lines) < 3:
            raise ValueError("Not enough lines to perform noise discard.")
        lines, kmeansVisuals = self.kMeansNoiseDiscard(frame, lines, 3)

        if lines is None:
            raise ValueError("No lines detected after k means noise discard.")

        # Loop until no more merges occur
        prevLenLines: int = len(lines) + 1
        while len(lines) < prevLenLines:
            prevLenLines: int = len(lines)
            lines = self.mergeLines(lines, np.deg2rad(5), 30)

        angleClusters: (
            tuple[
                NDArray[np.float64],
                NDArray[np.integer],
                NDArray[np.float64],
                NDArray[np.float64],
            ]
            | None
        ) = self.angleClusters(lines, 2)

        if angleClusters is None:
            raise ValueError("Angle clustering failed.")
        _, labels, _, centers_deg = angleClusters

        # Break into horizontal and vertical bins
        bins: dict[int, list] = {0: [], 1: []}
        for i, line in enumerate(lines):
            bins[labels[i]].append(line)

        horizontal_bin: int = min(
            bins.keys(),
            key=lambda k: min(abs(centers_deg[k] - 0), abs(centers_deg[k] - 180)),
        )
        vertical_bin: int = 1 - horizontal_bin  # since there are exactly two

        horizontalLines: list[NDArray[np.integer]] = bins[horizontal_bin]
        verticalLines: list[NDArray[np.integer]] = bins[vertical_bin]

        # Compute lengths for horizontal lines
        horizontal_lengths = [
            np.hypot(line[2] - line[0], line[3] - line[1]) for line in horizontalLines
        ]
        vertical_lengths = [
            np.hypot(line[2] - line[0], line[3] - line[1]) for line in verticalLines
        ]

        # Sort descending by length
        horizontal_lengths.sort(reverse=True)
        vertical_lengths.sort(reverse=True)

        # Take the top 5 (or fewer if not enough lines)
        top_horizontal = horizontal_lengths[:5]  # should only be 3 theoretically
        top_vertical = vertical_lengths[:5]

        # Compute averages (handle case with fewer than 5 lines)
        med_horizontal_length = np.median(top_horizontal)
        med_vertical_length = np.median(top_vertical)

        horizontalLinesFiltered: NDArray[np.integer] | None = self.discardShortLines(
            np.array(horizontalLines),
            int(med_horizontal_length * discardPercentageofMedian),
        )
        verticalLinesFiltered: NDArray[np.integer] | None = self.discardShortLines(
            np.array(verticalLines),
            int(med_vertical_length * discardPercentageofMedian),
        )

        # Convert to numpy arrays
        if horizontalLinesFiltered is None:
            raise ValueError("No horizontal lines detected after filtering.")
        if verticalLinesFiltered is None:
            raise ValueError("No vertical lines detected after filtering.")

        npHorizontal: NDArray[np.integer] = np.array(horizontalLinesFiltered)
        npVertical: NDArray[np.integer] = np.array(verticalLinesFiltered)

        combinedLines: NDArray[np.integer] = np.concatenate((npHorizontal, npVertical))

        return combinedLines, npHorizontal, npVertical, kmeansVisuals

    def kMeansNoiseDiscard(
        self, frame: Frame, lines: NDArray[np.integer] | None, k: int = 3
    ) -> tuple[NDArray[np.integer], list[Frame] | None]:
        """
        Cluster Hough lines by orientation using k-means and discard the smallest
        cluster as angular noise if it is sufficiently distinct.

        This function groups detected lines into `k` angle-based clusters (via
        `self.angleClusters`). The cluster with the fewest members is treated as
        noise and removed, unless:
        • Its center angle is too close to another cluster center, or
        • Its size exceeds a fixed threshold (i.e., it is not truly noise).

        Optionally, visual debug frames are produced showing each cluster drawn
        in a different color.

        Args:
            frame: Input image frame used only for generating visualization images.
            lines: Array of line segments of shape (N, 4), where each row is
                (x1, y1, x2, y2). Must not be None.
            k: Number of angle clusters to compute (default is 3).

        Returns:
            A tuple of:
                • combined_lines: NDArray[np.integer]
                    Array of lines with the noise cluster removed. If no valid
                    noise cluster is detected, all input lines are returned.
                • visuals: list[Frame] | None
                    List of frames visualizing each cluster in a different color,
                    or None if clustering could not be performed.

        Raises:
            ValueError: If `lines` is None.

        Notes:
            • The smallest cluster is considered "noise" only if its center angle
            differs from all other cluster centers by more than `degreeThreshold`.
            • Clusters larger than a hardcoded size threshold are never discarded.
            • Uses `self.angle_diff`, `self.angleClusters`, and `VisualUtils.drawLines`.
        """
        if lines is None:
            raise ValueError("No lines to discard noise from.")
        if len(lines) < k:
            return lines, None
        kmeans: (
            tuple[
                NDArray[np.float64],
                NDArray[np.integer],
                NDArray[np.float64],
                NDArray[np.float64],
            ]
            | None
        ) = self.angleClusters(lines, 3)
        if kmeans is None:
            return lines, None
        _, labels, _, centers_deg = kmeans
        logging.info("Cluster centers (deg):", centers_deg)

        bins: dict[int, list] = {0: [], 1: [], 2: []}
        for i, line in enumerate(lines):
            bins[labels[i]].append(line)

        visuals: list[Frame] | None = []
        for idx, colour in zip(bins.keys(), ["r", "g", "b"]):
            if len(bins[idx]) > 0:
                visuals.append(
                    VisualUtils.drawLines(frame, np.array(bins[idx]), colour)
                )

        cluster_sizes = {k: len(v) for k, v in bins.items()}
        self.logger.info(f"Cluster sizes: {cluster_sizes}")

        self.logger.info(f"Centres: {centers_deg}")

        # lengths of each cluster
        lengths: dict[int, int] = {k: len(v) for k, v in bins.items()}

        # find the smallest cluster -> noise
        noiseCluster: int = min(lengths, key=lambda cluster_id: lengths[cluster_id])

        # Check if noise cluster is worth keeping
        degreeThreshold: float = 20.0
        for i, angle in enumerate(centers_deg):
            if i == noiseCluster:
                continue
            if MiscUtil.angleDiff(angle, centers_deg[noiseCluster]) < degreeThreshold:
                self.logger.info(
                    "Noise cluster too close to other clusters, keeping all lines."
                )
                return lines, visuals

        # if the smallest cluster is too big, ignore
        if lengths[noiseCluster] > 10:
            self.logger.info(
                f"No noise cluster detected. Keeping all lines. lengths: {lengths}"
            )
            return lines, visuals

        # combine the other two clusters
        combined_lines: list = []
        for k in bins:
            if k != noiseCluster:
                combined_lines.extend(bins[k])
        combinedLinesProcessed: NDArray[np.integer] = np.array(
            combined_lines, dtype=np.int32
        )
        return combinedLinesProcessed, visuals

    def discardShortLines(
        self, lines: NDArray[np.integer] | None, lengthThreshold: int
    ) -> np.ndarray | None:
        if lines is None:
            return None
        filtered_lines: list = []
        for line in lines:
            x1: int
            y1: int
            x2: int
            y2: int
            x1, y1, x2, y2 = line
            length: int = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length >= lengthThreshold:
                filtered_lines.append(line)
        if not filtered_lines:
            return None
        return np.array(filtered_lines, dtype=np.int32)

    def calculateIntersections(
        self, horizontalLines: NDArray[np.integer], verticalLines: NDArray[np.integer]
    ) -> NDArray[np.integer] | None:
        """
        Compute all intersection points between sets of horizontal and vertical lines.

        Each horizontal line is intersected with each vertical line using the
        standard line–line intersection formula. The result is grouped by
        horizontal line: each row in the output corresponds to one horizontal
        input line and contains the (x, y) intersection points with all vertical
        lines.

        Args:
            horizontalLines: Array of shape (H, 4) containing horizontal line
                segments as (x1, y1, x2, y2).
            verticalLines: Array of shape (V, 4) containing vertical line
                segments as (x1, y1, x2, y2).

        Returns:
            An array of shape (H, V, 2) with integer (x, y) intersection points,
            grouped per horizontal line, or None if no valid intersections are found
            or if either input is None.

        Notes:
            • Parallel line pairs are skipped.
            • The output is structured so each sub-array corresponds to one
            horizontal line’s intersections.
            • Coordinates are cast to int before returning.

        """

        if horizontalLines is None or verticalLines is None:
            return None

        intersections: list = []

        for h_line in horizontalLines:
            x1_h, y1_h, x2_h, y2_h = h_line
            row_intersections: list = []  # ← NEW list for this horizontal line
            for v_line in verticalLines:
                x1_v, y1_v, x2_v, y2_v = v_line

                denom = (x1_h - x2_h) * (y1_v - y2_v) - (y1_h - y2_h) * (x1_v - x2_v)
                if denom == 0:
                    continue  # parallel lines

                px = (
                    (x1_h * y2_h - y1_h * x2_h) * (x1_v - x2_v)
                    - (x1_h - x2_h) * (x1_v * y2_v - y1_v * x2_v)
                ) / denom

                py = (
                    (x1_h * y2_h - y1_h * x2_h) * (y1_v - y2_v)
                    - (y1_h - y2_h) * (x1_v * y2_v - y1_v * x2_v)
                ) / denom

                row_intersections.append((int(px), int(py)))

            if row_intersections:
                intersections.append(row_intersections)

        if not intersections:
            return None

        return np.array(intersections, dtype=np.int32)

    def endLines(self, lines: NDArray[np.integer]) -> tuple[int, int]:
        """
        Determine the first and last non-outlier line indices based on line angles.

        This method computes the orientation angle of each line segment using
        `atan2(dy, dx)`, normalizes angles to the range [0, 180), and identifies
        angular outliers via the interquartile range (IQR) method using
        `self.iqrOutliers`. The earliest and latest indices whose angles are not
        classified as outliers are returned.

        Args:
            lines: Array of line segments with shape (N, 4), where each row is
                (x1, y1, x2, y2).

        Returns:
            A tuple (startline, endline) where:
                • startline is the index of the first non-outlier line.
                • endline is the index of the last non-outlier line.

        Notes:
            • Angles are computed in degrees and wrapped into [0, 180).
            • If all angles are considered outliers, the defaults are:
            startline = 0 and endline = len(lines) - 1.
            • Relies on `self.iqrOutliers` to return indices of outliers.

        """
        x1: NDArray[np.integer]
        y1: NDArray[np.integer]
        x2: NDArray[np.integer]
        y2: NDArray[np.integer]
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

        # calculate deltas as abs to solve wrap-around
        dx: NDArray[np.integer] = x2 - x1
        dy: NDArray[np.integer] = y2 - y1

        angles: NDArray[np.float64] = np.degrees(np.arctan2(dy, dx)) % 180

        outliers, (_, _) = MiscUtil.iqrOutliers(angles)
        outliers = set(outliers)  # convert to set for faster membership test

        # find first non-outlier index
        startline = next((i for i, a in enumerate(angles) if a not in outliers), 0)

        # find last non-outlier index
        endline = next(
            (i for i in reversed(range(len(angles))) if angles[i] not in outliers),
            len(angles) - 1,
        )

        return startline, endline

    def lineIntersection(
        self, line1: NDArray[np.integer] | None, line2: NDArray[np.integer] | None
    ) -> NDArray[np.float64] | None:
        """
        Compute the intersection point of two 2D lines.

        Each line is defined by four integer coordinates: (x1, y1, x2, y2).

        Args:
            line1: Line in the form (x1, y1, x2, y2) as integers.
            line2: Line in the form (x3, y3, x4, y4) as integers.

        Returns:
            A NumPy array of shape (2,) with the intersection point [x, y] as floats,
            or None if the lines are parallel.
        """
        if line1 is None or line2 is None:
            return None

        x1: int
        y1: int
        x2: int
        y2: int
        x3: int
        y3: int
        x4: int
        y4: int

        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Line equations: A*x + B*y = C
        A1: int = y2 - y1
        B1: int = x1 - x2
        C1: int = A1 * x1 + B1 * y1

        A2: int = y4 - y3
        B2: int = x3 - x4
        C2: int = A2 * x3 + B2 * y3

        det: int = A1 * B2 - A2 * B1
        if det == 0:
            return None  # Lines are parallel

        x: float = (B2 * C1 - B1 * C2) / det
        y: float = (A1 * C2 - A2 * C1) / det

        return np.array([x, y], dtype=np.float64)

    def calculateCornerPoints(
        self,
        horizontalLines: NDArray[np.integer],
        verticalLines: NDArray[np.integer],
        topline: int,
        bottomline: int,
        leftline: int,
        rightline: int,
    ) -> Dict[str, NDArray[np.float64] | None]:
        """
        Calculate the four corner intersection points of a grid or rectangle.

        Args:
            horizontalLines: List or array of horizontal lines, each as (x1, y1, x2, y2) integers.
            verticalLines: List or array of vertical lines, each as (x1, y1, x2, y2) integers.
            topline: Index of the top horizontal line.
            bottomline: Index of the bottom horizontal line.
            leftline: Index of the left vertical line.
            rightline: Index of the right vertical line.

        Returns:
            A dictionary with keys 'top_left', 'top_right', 'bottom_left', 'bottom_right'.
            Each value is an np.ndarray of shape (2,) with [x, y] as floats, or None if lines are parallel.
        """
        corners: Dict[str, NDArray[np.float64] | None] = {
            "top_left": self.lineIntersection(
                horizontalLines[topline], verticalLines[leftline]
            ),
            "top_right": self.lineIntersection(
                horizontalLines[topline], verticalLines[rightline]
            ),
            "bottom_left": self.lineIntersection(
                horizontalLines[bottomline], verticalLines[leftline]
            ),
            "bottom_right": self.lineIntersection(
                horizontalLines[bottomline], verticalLines[rightline]
            ),
        }
        return corners

    def rectifyGrid(
        self, image: Frame, corners: Dict[str, NDArray[np.float64] | None]
    ) -> Frame:
        """
        Rectify a quadrilateral region of an image to a top-down rectangular view.

        This function takes the four corner points of a grid (or any quadrilateral)
        in an image and computes a perspective transform (homography) to produce
        a rectified, axis-aligned rectangle containing the region.

        Args:
            image (np.ndarray): Input image as a NumPy array (H x W x C or H x W).
            corners (dict[str, np.ndarray]): Dictionary of corner points with keys:
                - "top_left"
                - "top_right"
                - "bottom_right"
                - "bottom_left"
                Each value should be a NumPy array of shape (2,) representing [x, y] coordinates.

        Returns:
            np.ndarray: Rectified image cropped and warped to a rectangle defined
            by the maximum width and height of the quadrilateral.

        Raises:
            ValueError: If any corner is None or if the homography cannot be computed.

        Notes:
            - The output rectangle will have its top-left corner at (0, 0).
            - Width is determined by the maximum of the top and bottom edge lengths.
            - Height is determined by the maximum of the left and right edge lengths.
            - The method logs corner information for debugging purposes.
        """

        self.logger.info("Corner points:")
        for k, v in corners.items():
            self.logger.info(f"  {k}: {v}")

        # Check if any points are the same
        pts: NDArray[np.float64] = np.array(
            [v for v in corners.values()], dtype=np.float64
        )
        self.logger.info(f"Unique corners: {np.unique(pts, axis=0)}")
        pts_src: NDArray[np.float64] = np.array(
            [
                corners["top_left"],
                corners["top_right"],
                corners["bottom_right"],
                corners["bottom_left"],
            ],
            dtype=np.float64,
        )

        topLeft = corners["top_left"]
        topRight = corners["top_right"]
        bottomLeft = corners["bottom_left"]
        bottomRight = corners["bottom_right"]

        if (
            topLeft is None
            or topRight is None
            or bottomLeft is None
            or bottomRight is None
        ):
            raise ValueError("One or more corner points are None; cannot rectify grid.")

        widthTop: float = float(np.linalg.norm(topRight - topLeft))
        widthBottom: float = float(np.linalg.norm(bottomRight - bottomLeft))
        width: int = int(max(widthTop, widthBottom))

        heightLeft: float = float(np.linalg.norm(bottomLeft - topLeft))
        heightRight: float = float(np.linalg.norm(bottomRight - topRight))
        height: int = int(max(heightLeft, heightRight))

        ptsDst: NDArray[np.float64] = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float64,
        )

        # Compute homography
        H: MatLike
        H, _ = cv.findHomography(pts_src, ptsDst)
        if H is None:
            raise ValueError("Homography could not be computed; check corner points.")

        # Warp image
        rectified = cv.warpPerspective(image, H, (width, height))
        return rectified

    def homographyTransform(
        self,
        image: Frame,
        horizontalLines: NDArray[np.integer],
        verticalLines: NDArray[np.integer],
    ):
        """
        Apply a homography-based perspective correction to an image using detected grid lines.

        This method selects the outermost horizontal and vertical lines from the detected grid,
        computes the four corner points of the grid via line intersections, and then rectifies
        the image so that the grid appears fronto-parallel.

        Steps:
            1. Sort horizontal lines by their average y-coordinate and vertical lines by their average x-coordinate.
            2. Identify the topmost, bottommost, leftmost, and rightmost lines using `endLines`.
            3. Compute the corner points where these lines intersect using `calculateCornerPoints`.
            4. Rectify the image using the computed corners via `rectifyGrid`.

        Args:
            image (Frame): The input image to be transformed.
            horizontalLines (NDArray[np.integer]): Array of shape (N, 4) containing detected horizontal lines
                in the format (x1, y1, x2, y2).
            verticalLines (NDArray[np.integer]): Array of shape (M, 4) containing detected vertical lines
                in the format (x1, y1, x2, y2).

        Returns:
            Frame: The rectified image after perspective correction.

        Raises:
            ValueError: If `horizontalLines` or `verticalLines` is None.
        """
        if horizontalLines is None or verticalLines is None:
            raise ValueError("Horizontal or vertical lines are None.")
        avgY: NDArray[np.float64] = (horizontalLines[:, 1] + horizontalLines[:, 3]) / 2
        sortIdx: NDArray[np.integer] = np.argsort(avgY)
        horizontalLines = horizontalLines[sortIdx]

        avgX: NDArray[np.float64] = (verticalLines[:, 0] + verticalLines[:, 2]) / 2
        verticalLines = verticalLines[np.argsort(avgX)]

        topline: int
        bottomline: int
        leftline: int
        rightline: int

        topline, bottomline = self.endLines(horizontalLines)
        leftline, rightline = self.endLines(verticalLines)

        self.logger.info("Selected grid lines:")
        self.logger.info(f"topline: {topline}, bottomline: {bottomline}, leftline: {leftline}, rightline: {rightline}")

        # Calculate Via intersections

        corners: Dict[str, NDArray[np.float64] | None] = self.calculateCornerPoints(
            horizontalLines, verticalLines, topline, bottomline, leftline, rightline
        )

        return self.rectifyGrid(image, corners)

    def calculateGridSpacings(
        self,
        horizontalLines: NDArray[np.integer],
        verticalLines: NDArray[np.integer],
        isHorizontal: bool,
        tolerancePercent: float = 5.0,
    ) -> NDArray | None:
        """
        Estimate the spacing between grid lines in an image based on detected intersections.

        This method analyzes intersections between horizontal and vertical lines to determine
        the most likely number of evenly spaced columns or rows along a specified axis
        (horizontal or vertical). It iteratively adjusts a guess for the spacing, considering
        a tolerance, and returns whether a regular spacing pattern was detected along with
        the estimated count.

        Parameters
        ----------
        horizontalLines : NDArray[np.integer]
            Array of horizontal lines in the format (x1, y1, x2, y2).
        verticalLines : NDArray[np.integer]
            Array of vertical lines in the format (x1, y1, x2, y2).
        isHorizontal : bool
            If True, compute spacings along the horizontal axis (columns),
            otherwise along the vertical axis (rows).
        tolerancePercent : float, optional
            Percentage of the axis length allowed as deviation from the expected spacing
            to still be considered a valid interval. Default is 5.0.

        Returns
        -------
        np.ndarray | None
            An array of shape (N, 2), where N is the number of grid lines analyzed.
            Each row contains:
                [spacingDetected: bool, estimatedCount: int]
            - `spacingDetected` indicates whether a regular spacing pattern was found.
            - `estimatedCount` is the estimated number of evenly spaced segments along the axis.
            Returns None if no intersections are found.

        Raises
        ------
        ValueError
            If no intersections between horizontal and vertical lines are found.

        Notes
        -----
        - The method sorts intersections along the axis to ensure proper spacing calculation.
        - It iteratively guesses the number of columns/rows (`colsGeuss`) and checks distances
        between intersections against the guessed spacing with a specified tolerance.
        - The detection algorithm attempts to handle multiple panels by doubling and halving
        the guessed count when necessary.
        """

        axisLength: int = self.width if isHorizontal else self.height

        axis: int = 0 if isHorizontal else 1

        if isHorizontal:
            horizontalLines = horizontalLines[np.argsort(horizontalLines[:, 0])]
        else:
            horizontalLines = horizontalLines[np.argsort(horizontalLines[:, 1])]

        intersections = self.calculateIntersections(horizontalLines, verticalLines)

        if intersections is None:
            raise ValueError(
                "No intersections found between horizontal and vertical lines."
            )

        # ensure sorted
        intersections = np.array(
            [row[np.argsort(row[:, axis])] for row in intersections], dtype=object
        )

        detections: NDArray = np.empty((0, 2), dtype=object)
        for horizontalLine in intersections:
            spacingDetected: bool = False
            finallyReached = False
            colsGeuss: int = 1
            distancesTemp: list[int] = [0]

            for i in range(1, len(horizontalLine)):
                p0 = horizontalLine[0]
                pi = horizontalLine[i]

                if isHorizontal:
                    distancesTemp.append(abs(pi[0] - p0[0]))  # X distance
                else:
                    distancesTemp.append(abs(pi[1] - p0[1]))  # Y distance

            distances: NDArray[np.integer] = np.array(distancesTemp)
            self.logger.info("DISTANCE CHECK START")
            self.logger.info(f"IsHorizontal: {isHorizontal}")
            self.logger.info(f"ColsGuess: {colsGeuss}")
            self.logger.info(f"distances: {distances}")

            while finallyReached == False:
                SpacingGuess: float = axisLength / colsGeuss
                toleranceFactor: float = (tolerancePercent / 100) * axisLength

                # check if point is within tolerance

                pointer1: int = 0
                pointer2: int = 1
                successCount: int = 0

                for j in range(len(distances) - 1):
                    dist3 = distances[pointer2] - distances[pointer1]
                    self.logger.info(f"Distance between points: {dist3}")
                    self.logger.info(f"SpacingGuess: {SpacingGuess}")
                    if (
                        abs((distances[pointer2] - distances[pointer1]) - SpacingGuess)
                        < toleranceFactor
                    ):
                        successCount += 1
                        pointer1 = pointer2
                        pointer2 += 1
                        self.logger.info("SUCCESS:")
                    else:
                        pointer2 += (
                            1  # since pointer1 HAS to be included pointer2 moves on
                        )
                        self.logger.info("FAIL:")

                # regular fail case increase colsGeuss
                if successCount != colsGeuss and spacingDetected == False:
                    colsGeuss += 1
                    if colsGeuss > len(verticalLines) - 1:  # give up case
                        spacingDetected = False
                        detections = np.vstack((detections, [False, 0]))
                        break

                # success case double incase of more panels and try again
                elif successCount == colsGeuss:
                    spacingDetected = True
                    colsGeuss *= 2

                # if double check fails half colsGeuss and break
                elif successCount != colsGeuss and spacingDetected == True:
                    colsGeuss //= 2
                    detections = np.vstack((detections, [True, colsGeuss]))
                    finallyReached = True
            self.logger.info("DISTANCE CHECK END")

        self.logger.info(f"detections: {detections}")
        return detections

    def determineGridLines(
        self, horizontalLines: NDArray[np.integer], verticalLines: NDArray[np.integer]
    ) -> tuple[int, int]:
        """
          Determine the most likely regular grid spacings for horizontal and vertical lines.

          This method analyzes detected horizontal and vertical lines to estimate the number of
          evenly spaced rows and columns in a grid. It uses `calculateGridSpacings` to detect
          candidate spacings along each axis, filters out invalid detections, and returns
          the median spacing for each axis, ensuring consistency with the mode.

          Parameters
          ----------
          horizontalLines : NDArray[np.integer]
              Array of detected horizontal lines, shape (N, 4), where each row is (x1, y1, x2, y2).
          verticalLines : NDArray[np.integer]
              Array of detected vertical lines, shape (M, 4), where each row is (x1, y1, x2, y2).

          Returns
          -------
          tuple[int, int]
              A tuple `(horizontalSpacing, verticalSpacing)` representing the estimated
              number of evenly spaced segments along the horizontal and vertical axes.

          Raises
          ------
          ValueError
              - If no grid spacings are detected on either axis.
              - If no valid spacings remain after filtering.
              - If the median and mode of the detected spacings do not match, indicating
              inconsistent or unreliable detections.

          Notes
          -----
          - Filtering ensures only grid lines with a detected regular spacing pattern are considered.
          - The median of the filtered spacings is used as the primary estimate, with the mode
          used as a consistency check.
          - This method assumes that `calculateGridSpacings` returns an array where each row
        contains `[spacingDetected: bool, estimatedCount: int]`.
        """
        horizontalDetections: NDArray | None = self.calculateGridSpacings(
            horizontalLines, verticalLines, True
        )
        verticalDetections: NDArray | None = self.calculateGridSpacings(
            verticalLines, horizontalLines, False
        )

        if horizontalDetections is None or verticalDetections is None:
            raise ValueError("No grid spacings detected.")

        horizontalFiltered: NDArray[np.integer] = horizontalDetections[
            horizontalDetections[:, 0] == True
        ]
        verticalFiltered: NDArray[np.integer] = verticalDetections[
            verticalDetections[:, 0] == True
        ]
        # true tuple entries

        if len(horizontalFiltered) == 0 or len(verticalFiltered) == 0:
            # Should never get here
            raise ValueError("No valid grid spacings detected.")

        # get the spacings only
        horizontalSpacings: NDArray[np.integer] = horizontalFiltered[:, 1]
        verticalSpacings: NDArray[np.integer] = verticalFiltered[:, 1]

        horizontalSpacing: int = MiscUtil.safeInt(np.median(horizontalSpacings))
        verticalSpacing: int = MiscUtil.safeInt(np.median(verticalSpacings))

        h_mode: NDArray[np.integer] = MiscUtil.getMode(horizontalSpacings)
        v_mode: NDArray[np.integer] = MiscUtil.getMode(verticalSpacings)
        if horizontalSpacing != h_mode or verticalSpacing != v_mode:
            # If the median and mode do not match, it indicates a potential error in spacing detection.
            raise ValueError(
                "Median and mode of grid spacings do not match. Check grid lines."
            )

        return horizontalSpacing, verticalSpacing

    def execute(
        self,
        image,
        frameCount: int,
        logPath: str = "",
        visuals: bool = True,
        visualPath: str = "",
        diagonstics: bool = True,
    ) -> List[Frame]:
        self.resetParameters(image.shape[0], image.shape[1], logPath)
        if (diagonstics):
            self.logger = MiscUtil.setupLogger(f"segmentLogger{frameCount}", logPath)
        #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        lines, horizontalLines, verticalLines, kMeansVisualsMain = self.gridPipeline(
            image, self.lineCount
        )
        originalArea = image.shape[0] * image.shape[1]
        
        

        rectified = self.homographyTransform(image, horizontalLines, verticalLines)
        
        rectifedArea = rectified.shape[0] * rectified.shape[1]
        self.logger.info(f"Original Area: {originalArea}, Rectified Area: {rectifedArea}")
        
        rectifiedLines, rectifiedHoriz, rectifiedVert, kMeansVisualsRectified = (
            self.rectifiedPipeline(rectified, int(self.lineCount * (rectifedArea / originalArea)), logPath)
        )
        hspacing, vspacing, gridRectified = self.spacingPipeline(rectified, rectifiedHoriz, rectifiedVert)

        cells: List[Frame] = VisualUtils.splitImageToGrid(
            rectified, vspacing, hspacing
        )

        self.logger.info("H Spacing: %d", hspacing)
        self.logger.info("V Spacing: %d", vspacing)
        
        reaspectedFrames: List[Frame] = []
        for cell in cells:
            reaspectedFrames.append(
                VisualUtils.stretchToAspectRatio(
                    cell, int(self.width * self.aspectRatio), self.aspectRatio
                )
            )

        if (reaspectedFrames is None) or (len(reaspectedFrames) == 0):
            raise ValueError("No cells extracted from image.")

        original_with_lines = VisualUtils.drawLines(image, lines, "r")
        rectified_with_lines = VisualUtils.drawLines(rectified, rectifiedLines, "r")

        if visuals:
            self.visualUtil.savePanelArray(
                reaspectedFrames,
                frameCount,
                self.logger,
                visualPath + "/segmentation/cells",
            )
            self.visualUtil.saveFrame(
                original_with_lines,
                "original_with_lines",
                frameCount,
                self.logger,
                visualPath + "/segmentation",
            )
            self.visualUtil.saveFrame(
                rectified_with_lines,
                "rectified_with_lines",
                frameCount,
                self.logger,
                visualPath + "/segmentation",
            )
            self.visualUtil.saveFrame(
                gridRectified,
                "grid_rectified",
                frameCount,
                self.logger,
                visualPath + "/segmentation",
            )
            if kMeansVisualsMain is not None:
                for idx, visual in enumerate(kMeansVisualsMain):
                    self.visualUtil.saveFrame(
                        visual,
                        f"kmeans_main_{idx}",
                        frameCount,
                        self.logger,
                        visualPath + "/diagonstics",
                    )
            if kMeansVisualsRectified is not None:
                for idx, visual in enumerate(kMeansVisualsRectified):
                    self.visualUtil.saveFrame(
                        visual,
                        f"kmeans_rectified_{idx}",
                        frameCount,
                        self.logger,
                        visualPath + "/diagonstics",
                    )

        return reaspectedFrames

    def rectifiedPipeline(
        self, image: Frame, expectedLines: int, logPath: str
    ) -> tuple[
        NDArray[np.integer],
        NDArray[np.integer],
        NDArray[np.integer],
        list[Frame] | None,
    ]:
        """
        Full pipeline to process an image and return the original grid.
        """
        self.logger.info(f"Processing image of shape: {image.shape}")
        self.resetParameters(image.shape[0], image.shape[1], logPath)
        self.logger.info("NORMALIZER")
        self.logger.info(f"Width: {self.width}, Height: {self.height}")
        blurred = self.preProcess(
            image,
            self.denoiseLambdaWeight,
            self.denoiseMeanKernelSize,
            self.denoiuseGaussianSigma,
            self.denoiseDownsampleFactor,
        )
        lineOutput = self.lineDetection(blurred, expectedLines)
        # Add Corner Lines
        if lineOutput is None:
            raise ValueError("No lines detected in rectified pipeline.")
        lines, frame = lineOutput
        topLine = np.array([0, 0, self.width - 1, 0])
        bottomLine = np.array([0, self.height - 1, self.width - 1, self.height - 1])
        leftLine = np.array([0, 0, 0, self.height - 1])
        rightLine = np.array([self.width - 1, 0, self.width - 1, self.height - 1])
        lines = np.vstack((leftLine, topLine, lines, bottomLine, rightLine))
        groupAndFilterOutput = self.groupAndFilter(image, lines)
        if groupAndFilterOutput is None:
            raise ValueError("No lines detected in rectified pipeline.")
        lines, horizontalLines, verticalLines, kMeansVisuals = groupAndFilterOutput
        return lines, horizontalLines, verticalLines, kMeansVisuals

    def gridPipeline(
        self, image: np.ndarray, expectedLines: int, logPath: str = ""
    ) -> tuple[
        NDArray[np.integer],
        NDArray[np.integer],
        NDArray[np.integer],
        list[Frame] | None,
    ]:
        """
        Full pipeline to process an image and return the rectified grid.
        """
        self.logger.info(f"Processing image of shape: {image.shape}")
        self.resetParameters(image.shape[0], image.shape[1], logPath)
        self.logger.info("NORMALIZER")
        self.logger.info(f"Width: {self.width}, Height: {self.height}")
        blurred = self.preProcess(
            image,
            self.denoiseLambdaWeight,
            self.denoiseMeanKernelSize,
            self.denoiuseGaussianSigma,
            self.denoiseDownsampleFactor,
        )
        lineOutput = self.lineDetection(blurred, expectedLines)
        if lineOutput is None:
            raise ValueError("No lines detected in rectified pipeline.")
        lines, edges = lineOutput
        # showImages([image, blurred, edges], ["Original", "Preprocessed", "Edges"])
        groupAndFilterOutput = self.groupAndFilter(image, lines)
        if groupAndFilterOutput is None:
            raise ValueError("No lines detected in rectified pipeline.")
        lines, horizontalLines, verticalLines, kMeansVisuals = groupAndFilterOutput
        return lines, horizontalLines, verticalLines, kMeansVisuals

    def spacingPipeline(
        self,
        image: Frame,
        horizontalLines: NDArray[np.integer],
        verticalLines: NDArray[np.integer],
    ) -> tuple[int, int, Frame]:
        """
        Full pipeline to process an image and return the grid spacings.
        """
        horizontalSpacing, verticalSpacing = self.determineGridLines(
            horizontalLines, verticalLines
        )
        self.logger.info(
            "Determined Spacings - Horizontal: %s | Vertical: %s",
            horizontalSpacing,
            verticalSpacing,
        )
        gridOnRectified = VisualUtils.drawSpacingLines(
            image, verticalSpacing, horizontalSpacing, self.height, self.width
        )
   
        return horizontalSpacing, verticalSpacing, gridOnRectified

import re
import cv2 as cv
import numpy as np
from typing import Any, Final, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib

# import pytesseract
from datetime import datetime
import os
from abc import ABC, abstractmethod
import json

from P2DingoCV.Camera.Camera import Camera
from .Exceptions.TempDetectionFail import TempDetectionFailed
from ..Types.Types import *
from ..Util.VisualUtil import VisualUtils
from ..Util.MiscUtil import MiscUtil
from logging import Logger

from numpy.typing import NDArray
from typing import List


# DEV
matplotlib.use("Agg")


class HotspotDetector():
    """Abstract base class for hotspot detection in thermal or visual frames.

    This class provides core utilities for detecting hotspots using clustering,
    contrast analysis, and shape metrics. Specific hotspot detection strategies
    should be implemented in subclasses by overriding :meth:`execute`.

    Attributes:
        cam (Camera): The camera object used to read in frames.
        outputPath (str): Directory where results and diagnostics will be saved.
        utility (VisualUtils): Utility object for saving and plotting diagnostics.
        labFrame (Frame): Current frame converted to CIELAB color space.
        frame (Frame): Current raw frame in BGR format.
        frameArea (int): Total number of pixels in the frame.
        frameCount (int): Number of frames processed.
        k (int): Number of clusters for k-means segmentation.
        clusterJoinKernel (int): Kernel size for morphological joining of clusters.
        hotSpotThreshold (float): Score threshold for hotspot classification.
        sigmoidSteepnessDeltaP (float): Steepness parameter for ΔP sigmoid scaling.
        sigmoidSteepnessZ (float): Steepness parameter for Z-score sigmoid scaling.
        compactnessCutoff (float): Normalization cutoff for compactness.
        dilationSize (int): Kernel dilation factor for contrast calculation.
        wDeltaP (float): Weight for ΔP score in final hotspot scoring.
        wZscore (float): Weight for Z-score in final hotspot scoring.
        wCompactness (float): Weight for compactness in final hotspot scoring.
        wAspectRatio (float): Weight for aspect ratio in final hotspot scoring.
        wEccentricity (float): Weight for eccentricity in final hotspot scoring.
        pixelCounts (np.ndarray): Pixel counts per cluster from k-means.
        colours (list[tuple[int, int, int]]): List of BGR colours for visualization.
    """

    def __init__(
        self, outputPath: str, config: str | None = None
    ) -> None:
        self.outputPath: str = outputPath
        self.utility = VisualUtils(self.outputPath)
        self.labFrame: Frame
        self.frame: Frame
        self.frameArea: int
        self.frameCount: int = 0
        # Paramaters To Tune set to defaults
        self.k: int = 10
        self.clusterJoinKernel: int = 3
        self.hotSpotThreshold: float = 0.6
        self.sigmoidSteepnessDeltaP: float = 0.25
        self.sigmoidSteepnessZ: float = 0.23
        self.compactnessCutoff: float = 0.6
        # Temparature Contrast
        self.dilationSize = 5

        # HotspotScore Weightings
        self.wDeltaP: float = 0.35
        self.wZscore: float = 0.35
        self.wCompactness: float = 0.3
        self.wAspectRatio: float = 0
        self.wEccentricity: float = 0

        # Diagnostics
        self.pixelCounts: NDArray[np.integer] = np.zeros(self.k).astype(np.int32)

        self.colours = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Load Paramaters from JSON config file
        if config is not None: 
            self.loadConfig(config)
         
        # ------------Temparature Hardware Dependant------------------
        # Uncomment if You Wish to Implement OCR Temparature Detection
        # ------------------------------------------------------------
        # self.clusterTemps: np.ndarray = np.zeros(self.k)
        # self.maxTemp: float | None = 50 # starter values
        # self.minTemp: float | None = 10 # starter values
        # self.tempDetected: bool = False
        # self.coldestPixel: float | None = 10 # hardware dependant
        # self.hottestPixel: float | None = 255 # hardware dependant
        # self.gradient: float | None
        # self.intercept: float | None
        # ------------------------------------------------------------
    
    def loadConfig(self, configPath: str ) -> None:
        """Load hotspot detector parameters from a JSON configuration file.

        This will overwrite the default tunable parameters set in `__init__`.

        Args:
            config_path (str): Path to the JSON configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the config file is not valid JSON.
        """
        with open(configPath, "r") as f:
            config = json.load(f)

        # Update only known attributes
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def updateOutputPath(self, newPath: str) -> None:
        """
        Update the output path used by this instance.

        Parameters
        ----------
        newPath : str
            The new file system path where output should be stored.

        Returns
        -------
        None
        """
        self.outputPath = newPath
                
    def perFrameProcessing(
        self, frame: Frame, saveFrames: bool, diagonstics: bool, frameCount: int
    ) -> tuple[list, bool]:
        """Process a single frame through the hotspot detection pipeline.

        Args:
            frame (Frame): Input image frame in BGR format.
            saveFrames (bool): Whether to save visual outputs.
            diagonstics (bool): Whether to save intermediate diagnostic results.

        Returns:
            tuple[list, bool]:
                - List of hotspot detection results for this frame.
                - Boolean indicating whether a hotspot was detected.
        """
        self.frame: Frame = frame
        self.logger: Logger = MiscUtil.setupLogger(f"hotspotLogger{frameCount}", self.outputPath + "/logs")
        self.labFrame: Frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        frame = self.perFrameSetup(frame)
        maskArray: List[Frame] = self.kMeansThermalGrouping(frame, diagonstics)
        return self.determineHotspots(frame, maskArray, saveFrames)

    def perFrameSetup(self, frame: Frame) -> Frame:
        """Initialize state for a new frame.

        Resets internal counters, computes frame area, and sets up
        per-frame parameters.

        Args:
            frame (Frame): Input frame in BGR format.

        Returns:
            Frame: The original frame after setup.
        """
        self.resetFrameData()
        self.frameArea = frame.shape[0] * frame.shape[1]
         
        # ------------Temparature Hardware Dependant------------------
        # Uncomment if You Wish to Implement OCR Temparature Detection
        # ------------------------------------------------------------
        
        # self.resetTempData()
        # self.tempDetected = True
        # try:
        #     self.updateTemps(frame)
        # except TempDetectionFailed:
        #     self.resetTempData()
        #     self.tempDetected = False
        
        # ------------------------------------------------------------
        return frame

    def kMeansThermalGrouping(self, frame: Frame, saveDiagonstics: bool) -> List[Frame]:
        """Cluster pixels into thermal groups using k-means.

        Args:
            frame (Frame): Input frame in BGR format.
            saveDiagonstics (bool): Whether to save segmented outputs.

        Returns:
            List[Frame]: List of binary masks for each k-means cluster.
        """
        original = frame.copy()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        pixels = frame.reshape((-1, 3)).astype(np.float32)

        # Define criteria and number of clusters (k)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)

        bestLabels = np.empty((pixels.shape[0], 1), dtype=np.int32)
        _, labels, centers = cv.kmeans(
            pixels, self.k, bestLabels, criteria, 10, cv.KMEANS_RANDOM_CENTERS
        )

        # Determine Pixel Counts
        self.pixelCounts = np.bincount(labels.flatten(), minlength=self.k)

        # ------------Temparature Hardware Dependant------------------
        # Uncomment if You Wish to Implement OCR Temparature Detection
        # ------------------------------------------------------------
        # try:
        #     self.determineClusterTemp(centers)
        # except TempDetectionFailed:
        #     self.resetTempData()
        # ------------------------------------------------------------

        # convert to ints to get a single pixel value
        centers = centers.astype(np.uint8)
        # make int to satisfy python labels are ints corresponding to their respective centre
        labels = labels.flatten().astype(int)
        segmented: Frame = centers[labels]

        # Reshape to original image shape
        segmentedImg: Frame = segmented.reshape(frame.shape)

        # Optional: convert back to BGR for visualization
        segmentedBGR: Frame = cv.cvtColor(segmentedImg, cv.COLOR_LAB2BGR)

        outlined: Frame = original.copy()
        # sort by intensity
        sortedIndices: np.ndarray = np.argsort(centers[:, 0])[::-1]

        # Mask for Highest Intensity Region
        mask: Frame = (labels.reshape(frame.shape[:2]) == sortedIndices[0]).astype(
            np.uint8
        ) * 255
        # self.saveFrame(mask)
        
        # Mask Arraying
        
        maskArray: List[Frame] = []
        for i in range(self.k):
            mask_i: Frame = (labels.reshape(frame.shape[:2]) == sortedIndices[i]).astype(
                np.uint8
            ) * 255
            maskArray.append(mask_i)
        
        # Outlines of top 2 k regions and BGR Segments
        if saveDiagonstics == True:   
            for i in range(3):
                mask2: Frame = (labels.reshape(frame.shape[:2]) == sortedIndices[i]).astype(
                    np.uint8
                ) * 255
                contours, _ = cv.findContours(
                    mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
                )
                cv.drawContours(
                    outlined, contours, -1, self.colours[i % len(self.colours)], 2
                )
            self.utility.saveFrame(
                outlined, f"kMeansOutlines", self.frameCount, self.logger, self.outputPath + "/diagonstics"
            )
            self.utility.saveFrame(
                segmentedBGR, "kMeansSegments", self.frameCount, self.logger, self.outputPath + "/diagonstics"
            )

        return maskArray

    def connectedComponentsFromMask(self, masks: List[Frame]) -> List[List[Dict[str, Any]]]:
        """
        Given a list of binary masks, compute connected components for each mask.

        Parameters
        ----------
        masks : list of np.ndarray
            Each mask should be a 2D binary image (0/255 or False/True).

        Returns
        -------
        list[list[dict]]
        
            Outer list = per mask
            Inner list = components in that mask
            
            Each component dict contains:
                - 'mask': binary mask of the component
                - 'bbox': (x1, y1, x2, y2)
                - 'area': pixel area
                - 'centroid': (cx, cy)
                
        """
        all_components = []

        for mask in masks:
            # Ensure binary uint8
            bin_mask = (mask > 0).astype(np.uint8)

            num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
                bin_mask, connectivity=8
            )

            kernel = cv.getStructuringElement(cv.MORPH_RECT, (self.clusterJoinKernel, self.clusterJoinKernel))
            components = []
            for i in range(1, num_labels):  # Skip background
                x, y, w, h, area = stats[i]
                
                bbox = (x, y, x + w, y + h)
                if area < self.frameArea * 0.005:
                    continue

                comp_mask = (labels == i).astype(np.uint8) * 255
                
                h, w = comp_mask.shape[:2]
                iterations = 3

                x1, y1, x2, y2 = bbox
                pad = self.clusterJoinKernel * iterations

                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(w, x2 + pad)
                y2p = min(h, y2 + pad)

                roi = comp_mask[y1p:y2p, x1p:x2p]
                roi = cv.dilate(roi, kernel, iterations=iterations)
                comp_mask[y1p:y2p, x1p:x2p] = roi

                components.append({
                    "mask": comp_mask,
                    "bbox": bbox,
                    "area": int(area),
                    "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                })

            all_components.append(components)

        return all_components
    
    def dilateHotspots(self, hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dilate the binary masks of hotspots to expand their regions.

        This function applies morphological dilation to each hotspot mask within 
        a region of interest (ROI) defined by the hotspot's bounding box, expanded 
        to avoid clipping. The hotspot dictionaries are updated in-place: 
        - 'mask' is modified
        - 'bbox', 'area', and 'centroid' are recalculated

        Parameters
        ----------
        hotspots : list of dict
            Each dict represents a hotspot and must contain at least:
                - 'mask': np.ndarray, binary mask of the hotspot
                - 'bbox': tuple (x_min, y_min, x_max, y_max)

        Returns
        -------
        list of dict
            The same list of hotspot dicts with dilated masks and updated properties.
        """

        kernel: MatLike = cv.getStructuringElement(
            cv.MORPH_RECT, (self.clusterJoinKernel, self.clusterJoinKernel)
        )
        iterations: int = 3
        kernel_radius: int = self.clusterJoinKernel // 2

        for hotspot in hotspots:
            comp_mask = hotspot["mask"]

            h, w = comp_mask.shape[:2]
            x1, y1, x2, y2 = hotspot["bbox"]

            # Expand ROI to avoid clipping during dilation
            pad = kernel_radius * iterations

            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            roi = comp_mask[y1p:y2p, x1p:x2p]
            roi = cv.dilate(roi, kernel, iterations=iterations)
            comp_mask[y1p:y2p, x1p:x2p] = roi

            # Recompute bbox / area / centroid
            ys, xs = np.where(comp_mask > 0)
            if len(xs) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                hotspot["bbox"] = (x_min, y_min, x_max + 1, y_max + 1)
                hotspot["area"] = int(len(xs))
                hotspot["centroid"] = (float(xs.mean()), float(ys.mean()))
            else:
                hotspot["bbox"] = (0, 0, 0, 0)
                hotspot["area"] = 0
                hotspot["centroid"] = (0.0, 0.0)

        return hotspots
    
    
    def erodeHotspots(self, hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Erode the binary masks of hotspots to shrink their regions.

        This function applies morphological erosion to each hotspot mask within 
        a region of interest (ROI) defined by the hotspot's bounding box, expanded 
        to avoid clipping. The hotspot dictionaries are updated in-place:
        - 'mask' is modified
        - 'bbox', 'area', and 'centroid' are recalculated

        Parameters
        ----------
        hotspots : List[Dict[str, Any]]
            Each dict represents a hotspot and must contain at least:
                - 'mask': np.ndarray, binary mask of the hotspot
                - 'bbox': tuple (x_min, y_min, x_max, y_max)

        Returns
        -------
        List[Dict[str, Any]]
            The same list of hotspot dicts with eroded masks and updated properties.
        """

        kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (self.clusterJoinKernel, self.clusterJoinKernel)
        )
        iterations = 3
        kernel_radius = self.clusterJoinKernel // 2

        for hotspot in hotspots:
            comp_mask = hotspot["mask"]

            h, w = comp_mask.shape[:2]
            x1, y1, x2, y2 = hotspot["bbox"]

            # Expand ROI to avoid clipping
            pad = kernel_radius * iterations

            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            roi = comp_mask[y1p:y2p, x1p:x2p]
            roi = cv.erode(roi, kernel, iterations=iterations)
            comp_mask[y1p:y2p, x1p:x2p] = roi

            # Recompute bbox / area / centroid
            ys, xs = np.where(comp_mask > 0)
            if len(xs) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                hotspot["bbox"] = (x_min, y_min, x_max + 1, y_max + 1)
                hotspot["area"] = int(len(xs))
                hotspot["centroid"] = (float(xs.mean()), float(ys.mean()))
            else:
                hotspot["bbox"] = (0, 0, 0, 0)
                hotspot["area"] = 0
                hotspot["centroid"] = (0.0, 0.0)

        return hotspots


    def higherOrderOverlap(self, componentsFirstLayer: List[Dict[str, Any]], componentsSecondLayer: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Placeholder for overlap logic
        """
        Compute overlaps between two layers of component masks and merge them.

        For each component in `componentsFirstLayer`, this method checks for 
        overlapping bounding boxes with components in `componentsSecondLayer`. 
        If an overlap exists and their masks intersect, a new merged component 
        is created with:
        
            - A mask that is the union of the overlapping masks.
            - An updated bounding box covering both components.
            - Recalculated area based on the combined mask.
            - Centroid computed as the center of the new bounding box.

        Parameters
        ----------
        componentsFirstLayer : List[Dict[str, Any]]
            List of component dictionaries in the first layer. Each dictionary
            must contain at least:
            
                - 'mask': numpy array representing the component mask
                - 'bbox': tuple (x_min, y_min, x_max, y_max) defining the bounding box

        componentsSecondLayer : List[Dict[str, Any]]
            List of component dictionaries in the second layer. Each dictionary
            must contain at least 'mask' and 'bbox', same as above.

        Returns
        -------
        List[Dict[str, Any]]
            A list of merged component dictionaries with keys:
                - 'mask': combined mask of overlapping components
                - 'bbox': updated bounding box
                - 'area': area of the combined mask
                - 'centroid': center coordinates of the new bounding box
                
        """
        returnComponents: List[Dict[str, Any]] = []
        for comp1 in componentsFirstLayer:
            for comp2 in componentsSecondLayer:
                if MiscUtil.boxesOverlap(comp1['bbox'], comp2['bbox']):
                    # Handle overlap logic here
                    if np.any(comp1['mask'] & comp2['mask']):
                        newMask = cv.bitwise_or(comp1['mask'], comp2['mask'])
                        newBox = (
                            min(comp1['bbox'][0], comp2['bbox'][0]),
                            min(comp1['bbox'][1], comp2['bbox'][1]),    
                            max(comp1['bbox'][2], comp2['bbox'][2]),
                            max(comp1['bbox'][3], comp2['bbox'][3])
                        )
                        newArea = cv.contourArea(cv.findContours(newMask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0])
                        newCentroid = (float(newBox[0] + (newBox[2] - newBox[0]) / 2), float(newBox[1] + (newBox[3] - newBox[1]) / 2))
                        returnComponents.append({
                            "mask": newMask,
                            "bbox": newBox,
                            "area": int(newArea),
                            "centroid": newCentroid
                        })
        return returnComponents

    def layeredCandidates(self, componentsArray: List[List[Dict[str, Any]]], depthLimit: int = 3) -> List[Dict[str, Any]]:
        """
        Generate a flattened list of candidate components by progressively combining layers of components.

        This method starts with the first layer of components and iteratively merges it with
        subsequent layers using the `higherOrderOverlap` method. The result is a single list
        of candidate components that accounts for overlaps or relationships between layers.

        Parameters
        ----------
        componentsArray : List[List[Dict[str, Any]]]
            A list of layers, where each layer is a list of component dictionaries.
            Each component dictionary contains attributes describing a detected component
            (e.g., bounding box, confidence score, etc.).

        Returns
        -------
        List[Dict[str, Any]]
            A flattened list of candidate components after combining all layers. Components
            from higher layers that overlap or relate to existing candidates are included
            according to the logic in `higherOrderOverlap`.

        Notes
        -----
        - The `higherOrderOverlap` method is responsible for determining which components
            from the next layer should be merged with the current candidates.
        - The first layer of components is always included in the returned list.
        """
        canididates: List[Dict[str, Any]] = []
        canididates.extend(componentsArray[0])
        canididates.extend(componentsArray[1])
        canididates.extend(componentsArray[2])
        canididates.extend(componentsArray[3])
        
        if (len(componentsArray) - 1) < depthLimit:
            depthLimit = len(componentsArray) - 1
        
        for i in range(len(componentsArray) - 1 - (self.k - depthLimit)):
            layerComponents = self.higherOrderOverlap(canididates, componentsArray[i + 1])
            canididates.extend(layerComponents)
            
        return canididates

    def hotspotMetricPass(self, frame: Frame, canididates: List[Dict[str, Any]], otherHotspotsMask: Frame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Placeholder for overlap logic
        """
        Evaluate candidate regions in a frame and select those that qualify as hotspots 
        based on intensity contrast, shape, and compactness metrics.

        This method converts the input frame to the Lab color space and analyzes the 
        L-channel (lightness) to assess pixel contrast for each candidate region. 
        Additional metrics, such as shape compactness, aspect ratio, and eccentricity, 
        are also computed. A combined hotspot score is calculated, and candidates with 
        a score above the threshold are returned as detected hotspots.

        Parameters
        ----------
        frame : Frame
            The input image/frame in BGR color format.
            
        canididates : List[Dict[str, Any]]
        
            A list of candidate regions to evaluate. Each candidate dictionary should 
            contain at least:
            - 'mask': binary mask of the candidate region
            - 'area': area of the candidate region
                
        otherHotspotsMask : Frame
            Binary mask representing other already-detected hotspots. Used to exclude 
            overlapping pixels when computing contrast.

        Returns
        -------
        List[Dict[str, Any]]
            List of candidate dictionaries that passed the hotspot score threshold.
        List[Dict[str, Any]]
            List of candidate dictionaries that failed the hotspot score threshold.
            
            

        Notes
        -----
        - The method relies on `pixelContrast`, `shapeAndCompactness`, and `hotSpotScore`
            helper methods to compute the necessary metrics for each candidate.
        - `hotSpotThreshold` is a class attribute that defines the minimum score for a 
            region to be considered a hotspot.
            
        """
        LABFrame: Frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        LChannel: Frame = LABFrame[:, :, 0]
        
        
        hotspotsDetection: List[Dict[str, Any]] = []
        hotspotsFails: List[Dict[str, Any]] = []
        i = 0
        for candidate in canididates:
            i+=1
            self.logger.info(f'{i}')
            otherHotspotsMaskTemp = cv.bitwise_xor(otherHotspotsMask, candidate['mask'])
            deltaPRobust, zScore, deltaPScore, zScoreNorm, localMask = self.pixelContrast(
                LChannel, candidate['mask'], otherHotspotsMaskTemp, candidate['area']
            )
            compactness, aspectRatioNorm, eccentricity = self.shapeAndCompactness(
                candidate['mask'], candidate['area']
            )
            hotspotScore: float = self.hotSpotScore(
                deltaPScore=deltaPScore,
                zScoreNorm=zScoreNorm,
                compactness=compactness,
                aspectRatio=aspectRatioNorm,
                eccentricity=eccentricity,
            )
            
            if hotspotScore >= self.hotSpotThreshold:
                hotspotsDetection.append(candidate)    
            else:
               hotspotsFails.append(candidate) 
            
        return hotspotsDetection, hotspotsFails
    
    
    def mergeOverlappingHotspots(self, hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping hotspot regions into unified hotspots.

        This method iteratively examines each pair of hotspot regions and merges them
        if their bounding boxes overlap and their masks have any intersecting pixels. 
        The merged hotspot's mask is the union of the overlapping masks, its bounding 
        box is expanded to cover both regions, and the area and centroid are recalculated 
        based on the combined mask. The process continues until no further merges are possible.

        Parameters
        ----------
        hotspots : List[Dict[str, Any]]
            A list of hotspot dictionaries, each containing at least:
                - 'mask': binary mask of the hotspot (np.uint8)
                - 'bbox': bounding box as (x_min, y_min, x_max, y_max)
                
            Optional fields will be recalculated (area, centroid) during merging.

        Returns
        -------
        List[Dict[str, Any]]
            List of merged hotspots, each with updated:
                - 'mask': merged binary mask
                - 'bbox': updated bounding box covering all merged regions
                - 'area': total area of the merged mask
                - 'centroid': true centroid of the merged mask

        Notes
        -----
        - Masks are ensured to be binary (0 or 255) before merging.
        - Centroid is computed from the mask using image moments; if the mask is empty,
            the centroid defaults to the center of the bounding box.
        - The method uses iterative merging until no overlapping hotspots remain.
        
        """
    
        n = len(hotspots)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        def bbox_overlap(b1, b2):
            return not (
                b1[2] < b2[0] or b2[2] < b1[0] or
                b1[3] < b2[1] or b2[3] < b1[1]
            )

        # --- Build unions ---
        for i in range(n):
            for j in range(i + 1, n):
                mask_i = hotspots[i]['mask'].astype(np.uint8)
                mask_j = hotspots[j]['mask'].astype(np.uint8)
                
                if np.any(cv.bitwise_and(mask_i, mask_j)):
                    self.logger.info("Detection")
                    union(i, j)

        # --- Group by root ---
        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(hotspots[i])

        # --- Merge each group ---
        def merge_many(group):
            merged_mask = np.zeros_like(group[0]['mask'])

            for h in group:
                merged_mask = cv.bitwise_or(merged_mask, h['mask'])

            ys, xs = np.where(merged_mask > 0)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            area = int(len(xs))
            cx, cy = float(xs.mean()), float(ys.mean())

            return {
                "mask": merged_mask,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "centroid": (cx, cy),
            }

        return [merge_many(group) for group in groups.values()]


    def finalMetricPass(self, frame: Frame, hotspots: List[Dict[str, Any]], otherHotspotMaskFinal: Frame) -> List[Tuple]:
        # Placeholder for overlap logic
        """
        Compute detailed metrics for each hotspot and return a structured results list.

        This method evaluates each hotspot region in the input frame using pixel contrast 
        and shape metrics. The frame is converted to Lab color space, and the L-channel 
        (lightness) is used to compute contrast-based metrics. Shape-based metrics such 
        as compactness, aspect ratio, and eccentricity are also calculated. A combined 
        hotspot score is computed for each hotspot.

        Parameters
        ----------
        frame : Frame
            Input image/frame in BGR color format.
        hotspots : List[Dict[str, Any]]
            List of hotspot dictionaries to evaluate. Each dictionary should contain at least:
                - 'mask': binary mask of the hotspot region
                - 'area': area of the hotspot
                - 'centroid': (x, y) coordinates of the hotspot centroid
                
        otherHotspotMaskFinal : Frame
            Binary mask representing other hotspots (not used in current placeholder logic).

        Returns
        -------
        List[Tuple]
            List of tuples containing metrics for each hotspot in the following format:
                (
                    index,          # Index of the hotspot in the input list
                    hotspotScore,   # Combined hotspot score
                    None,           # Placeholder (currently unused)
                    centroid,       # Centroid coordinates (x, y)
                    deltaPScore,    # Pixel contrast score
                    deltaPRobust,   # Robust pixel contrast
                    zScore,         # Raw z-score for contrast
                    zScoreNorm,     # Normalized z-score
                    compactness,    # Shape compactness
                    aspectRatioNorm,# Normalized aspect ratio
                    eccentricity,   # Shape eccentricity
                    area            # Hotspot area
                    
                )

        Notes
        -----
        - Uses helper methods `pixelContrast`, `shapeAndCompactness`, and `hotSpotScore`.
        - The `otherHotspotMaskFinal` parameter is currently not used in the pixel contrast calculation.
        - Designed as the final evaluation pass to generate a comprehensive metric set for downstream processing or reporting.
        
    """
        LABFrame: Frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        LChannel: Frame = LABFrame[:, :, 0]
        
        
        results: List = []
        LChannel = self.robustConstrastStretch(LChannel)
        
        
        for i, hotspot in enumerate(hotspots):
            otherHotspotMaskInput = cv.bitwise_xor(otherHotspotMaskFinal, hotspot['mask'])
            deltaPRobust, zScore, deltaPScore, zScoreNorm, localMask = self.pixelContrast(
                LChannel, hotspot['mask'], otherHotspotMaskInput, hotspot['area']
            )
            compactness, aspectRatioNorm, eccentricity = self.shapeAndCompactness(
                hotspot['mask'], hotspot['area']
            )
            hotspotScore: float = self.hotSpotScore(
                deltaPScore=deltaPScore,
                zScoreNorm=zScoreNorm,
                compactness=compactness,
                aspectRatio=aspectRatioNorm,
                eccentricity=eccentricity,
            )
            
            results.append(
                (
                    i,
                    hotspotScore,
                    None,
                    hotspot['centroid'],
                    deltaPScore,
                    deltaPRobust,
                    zScore,
                    zScoreNorm,
                    compactness,
                    aspectRatioNorm,
                    eccentricity,
                    hotspot['area'],
                )
            )
                  
            
        return results
    
    def determineHotspots(self, frame: Frame, maskArray: List[Frame], saveVisuals: bool) -> tuple[list, bool]:
        """Determine hotspots across multiple frames.

        Args:
            frame (Frame): Input image frame in BGR format.
            maskArray (List[Frame]): List of input binary masks.
            saveVisuals (bool): Whether to save annotated hotspot frames.

        Returns:
            tuple[list, bool]:
                - List of hotspot detection results for all frames.
                - Boolean indicating whether any hotspot was detected.
        """
        if Frame is None or len(maskArray) == 0:
            return [], False
        diagonsticFrame: Frame = frame.copy()
        hotSpotFrame = frame.copy()
        failFrame = frame.copy()
        filterMaskArray = [self.filterMask(mask) for mask in maskArray]
        hotspotDetection: bool = False
        
        kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (self.clusterJoinKernel, self.clusterJoinKernel)
        )
        closedMaskArray: List[Frame] = [cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel) for mask in filterMaskArray]
        # Erode again to seperate joined regions
        closedMaskArray = [cv.erode(mask, kernel, iterations=1) for mask in closedMaskArray]

        componentsArray = self.connectedComponentsFromMask(closedMaskArray)
        self.logger.info("COMPONENTS ARRAY")
        self.logger.info(f'{sum(len(row) for row in componentsArray)}')

        candiadateHotspots = self.layeredCandidates(componentsArray) 
        
        self.logger.info("CANDIDATE HOTSPOTS")
        self.logger.info(f'{len(candiadateHotspots)}')
        
        self.logger.info("AREA")
        self.logger.info(f'{self.frameArea}')

        for i, candidate in enumerate(candiadateHotspots):
            self.utility.drawFrameCountours(
                frame=diagonsticFrame,
                componentMask=candidate['mask'],
                cx=int(candidate['centroid'][0]),
                cy=int(candidate['centroid'][1]),
                lbl=i,
            )  
            
        
        
        otherHotspotsMaskPrelim = filterMaskArray[0].copy()
        candiadateHotspots = self.erodeHotspots(candiadateHotspots)
        
        
        hotspots, fails = self.hotspotMetricPass(frame, candiadateHotspots, otherHotspotsMaskPrelim)
        
        self.logger.info("Original Hotspots")
        self.logger.info(f'{len(hotspots)}')
        
        #dilate so merges work
        hotspots = self.dilateHotspots(hotspots)
        
        hotspots = self.mergeOverlappingHotspots(hotspots)
       
        self.logger.info("Merged Hotspots") 
        self.logger.info(f'{len(hotspots)}')
        
        otherHotspotMaskFinal = np.zeros_like(maskArray[0])
        for hotspot in hotspots:
            otherHotspotMaskFinal = cv.bitwise_or(otherHotspotMaskFinal, hotspot['mask'])
            
        finalResults = self.finalMetricPass(frame, hotspots, otherHotspotMaskFinal)
        
        sortedResults = sorted(finalResults, key=lambda row: row[1], reverse=True)
        
        if (len(sortedResults) > 0):
            hotspotDetection = True
        
        if saveVisuals:
            
            for i, hotspot in enumerate(hotspots):
                cx: int; cy: int
                cx, cy = map(int, hotspot['centroid'])
                self.utility.drawFrameCountours(frame=hotSpotFrame, componentMask=hotspot['mask'], cx=cx, cy=cy, lbl=i)
                
            for i, hotspotFail in enumerate(fails):
                cx: int; cy: int
                cx, cy = map(int, hotspot['centroid'])
                self.utility.drawFrameCountours(frame=failFrame, componentMask=hotspot['mask'], cx=cx, cy=cy, lbl=i)
                
            self.utility.saveFrame(hotSpotFrame, "hotspots", self.frameCount, self.logger, self.outputPath + "/hotspots")
            self.utility.saveFrame(failFrame, "fails", self.frameCount, self.logger, self.outputPath + "/fails")
            self.utility.saveFrame(diagonsticFrame, "localDilations", self.frameCount, self.logger, self.outputPath + "/localDilations")
           
        return sortedResults, hotspotDetection


    # Filters out Noise
    def filterMask(self, mask: np.ndarray) -> Frame:
        """Filter out noise regions from a binary mask.

        Args:
            mask (np.ndarray): Input binary mask.

        Returns:
            np.ndarray: Filtered mask containing only valid regions.
        """
        n_labels, labels, stats, _ = cv.connectedComponentsWithStats(
            mask, connectivity=8
        )
        filteredMask = np.zeros_like(mask)

        for lbl in range(1, n_labels):
            compArea = stats[lbl, cv.CC_STAT_AREA]
            if compArea < self.frameArea / 2000:
                continue

            filteredMask[labels == lbl] = 255

        return filteredMask

    def shapeAndCompactness(
        self, componentMask: Frame, area: float
    ) -> tuple[float, float, float]:
        """Calculate shape metrics for a region.

        Args:
            componentMask (Frame): Binary mask of a connected component.
            area (float): Pixel area of the component.

        Returns:
            tuple[float, float, float]:
                - Normalized compactness.
                - Aspect ratio.
                - Eccentricity (or aspect ratio if ellipse fitting fails).
        """
        contours, _ = cv.findContours(
            componentMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        # Should only be one contour
        perimeter: float = cv.arcLength(contours[0], True)

        # Elipse Required 5 pixels already been filtered for size

        eccentricity: None | float = None
        if len(contours[0]) > 5:
            (_, _), (major_axis, minor_axis), _ = cv.fitEllipse(contours[0])
            eccentricity = np.sqrt(
                1 - (min(major_axis, minor_axis) / max(major_axis, minor_axis)) ** 2
            )
        _, _, w, h = cv.boundingRect(contours[0])

        # square has 0.785 compactness normalised by definition
        compactness: float = (4 * np.pi * area) / (perimeter**2)
        aspectRatioNorm: float = min(w, h) / max(w, h)
        compactness = np.clip(compactness / self.compactnessCutoff, 0, 1)

        if eccentricity == None:
            eccentricity = aspectRatioNorm
        # normalised by definition

        return compactness, aspectRatioNorm, eccentricity
    

    def robustConstrastStretch(self, region: Frame, lower_pct=1, upper_pct=99) -> Frame:
        """
        Perform a robust contrast stretch on an image region using percentiles.

        This method rescales the pixel intensities of `region` so that the 
        intensity at `lower_pct` maps to 0 and the intensity at `upper_pct` 
        maps to 255, effectively stretching the contrast while ignoring extreme
        outliers.

        Parameters
        ----------
        region : Frame
            Input image region as a numpy array. Can be grayscale or single-channel.

        lower_pct : float, optional (default=1)
            Lower percentile used for contrast stretching. Pixels below this 
            percentile are clipped to 0.

        upper_pct : float, optional (default=99)
            Upper percentile used for contrast stretching. Pixels above this 
            percentile are clipped to 255.

        Returns
        -------
        Frame
            Contrast-stretched image region as a numpy array of type np.uint8.
            Returns a copy of the input if the region is empty or if `p_high - p_low == 0`.
        """
        if region.size == 0:
            return region

        # Compute percentiles
        p_low, p_high = np.percentile(region, [lower_pct, upper_pct])
        
        # Avoid division by zero
        if p_high - p_low == 0:
            return region.copy()
        
        stretched = np.clip((region - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
        return stretched

    def pixelContrast(
        self,
        LChannel: Frame,
        componentMask: Frame,
        otherHotspotsMask: Frame,
        area: float, optimised: bool = True
    ) -> tuple[float, float, float, float, Frame]:
        """Compute contrast metrics for a hotspot region.

        Args:
            LChannel (Frame): Luminance channel of LAB frame.
            componentMask (Frame): Mask of the region of interest.
            otherHotspotsMask (Frame): Mask of other hotspots to exclude.
            area (float): Pixel area of the region.
            optimised (bool): Whether to use optimized percentile calculation.

        Returns:
            tuple[float, float, float, float, Frame]:
                - Robust ΔP contrast.
                - Global Z-score.
                - ΔP probability score.
                - Normalized Z-score.
                - Mask of Inspected Areas
        """
        ksize: int = int(max(3, self.dilationSize * np.sqrt(area)))
        kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
        # Could potentially add this as a diagonstic showing the dilation region
        dilated = cv.dilate(componentMask, kernel, iterations=1)

        hotMask: Frame = componentMask.astype(bool)
        localMask: Frame = (dilated.astype(bool)) & (~hotMask) & (~otherHotspotsMask)
        localMaskFrame = (localMask.astype(np.uint8)) * 255
        

        hotVals: Frame = LChannel[hotMask].ravel()
        localVals: Frame = LChannel[localMask].ravel()

        if len(hotVals) == 0:
            return 0, 0, 0, 0, localMask
        elif len(localVals) == 0:
            return 10, 10, 1, 1, localMask

        muHot: float = hotVals.mean()
        mu: float = LChannel.mean()
        sigma: float = LChannel.std()
        
        if optimised == True:
            self.logger.info("Optimised Percentile Calculation")
            max_sample = 5000
            if len(localVals) > max_sample:
                sample = np.random.choice(localVals, max_sample, replace=True)
            else:
                sample = localVals
            q25, q50, q75 = np.percentile(sample, [25, 50, 75])
            iqrLocal = q75 - q25
            medianLocal = q50

        else:
            self.logger.info("Optimised Percentile Calculation - np.partition")
            n = len(localVals)

            # Compute indices
            i25 = int(n * 0.25)
            i50 = n // 2
            i75 = int(n * 0.75)

            # np.partition gives k-th element in O(n) average
            
            indices = [i25, i50, i75]
            part = np.partition(localVals, indices)  # O(n) average
            q25, q50, q75 = part[i25], part[i50], part[i75]
            
            medianLocal: float = q50
            iqrLocal: float = q75 - q25

            
        # Robust Contrast
        deltaPRobust: float = (muHot - medianLocal) / (iqrLocal + 1e-6)
        
        deltaPRobustClipped: np.floating = np.clip(deltaPRobust, -50, 50).item()

        # Robust Constrast Probability

        deltaPProbabilityScore: float = np.clip(
            np.exp(self.sigmoidSteepnessDeltaP * deltaPRobustClipped) - 1, 0, 1
        )

        # Global Z score
        z = (muHot - mu) / sigma + 1e-6
        zScoreNorm: float = np.clip(np.exp(self.sigmoidSteepnessZ * z) - 1, 0, 1)

        return deltaPRobust, z, deltaPProbabilityScore, zScoreNorm, localMaskFrame

    def hotSpotScore(
        self,
        deltaPScore: float,
        zScoreNorm: float,
        compactness: float,
        aspectRatio: float,
        eccentricity: float,
    ) -> float:
        """Compute a weighted hotspot score.

        Args:
            deltaPScore (float): Probability score from robust ΔP.
            zScoreNorm (float): Normalized Z-score.
            compactness (float): Compactness of the hotspot region.
            aspectRatio (float): Aspect ratio of the hotspot region.
            eccentricity (float): Eccentricity of the hotspot region.

        Returns:
            float: Final weighted hotspot score.
        """
        # Normalise Z score
        return (
            deltaPScore * self.wDeltaP
            + zScoreNorm * self.wZscore
            + compactness * self.wCompactness
            + aspectRatio * self.wAspectRatio
            + (eccentricity) * self.wEccentricity
        )



    def processingDiagonstics(self) -> None:
        """Plot diagnostic data such as pixel count distributions.

        Saves plots to the configured output path.
        """
        # self.utility.plotFreqArray(self.clusterTemps, "Temp of Clusters")
        self.utility.plotFreqArray(self.pixelCounts, "Pixel Count of Clusters")

    # ----------------------------------------------------------------
    # --------------------Hotspot Helpers-----------------------------
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # ----------------To Be Modified Dependant on Hardware ------------
    # ----------------------------------------------------------------
    def pixelToTemp(self, pixelVal) -> float:
        """Convert a pixel value to temperature using calibration.

        Args:
            pixelVal (float): Pixel luminance value.

        Returns:
            float: Estimated temperature.

        Raises:
            TempDetectionFailed: If calibration parameters are missing.
        """
        if self.gradient is None or self.intercept is None:
            raise TempDetectionFailed()
        return self.gradient * pixelVal + self.intercept



    def resetFrameData(self) -> None:
        """Reset frame-dependent state variables.

        Resets temperature calibration, frame buffers, and cluster statistics.
        """
        self.maxTemp = None
        self.minTemp = None
        self.coldestPixel = 10  # TEMP
        self.hottestPixel = 255  # TEMP
        self.gradient = None
        self.intercept = None
        self.labFrame = np.zeros(0)
        self.frame = np.zeros(0)
        self.frameArea = 0
        # Data Points based on k
        self.clusterTemps = np.zeros(self.k)
        self.pixelCounts = np.zeros(self.k).astype(np.int32)

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    
    # def componentTemp(self, LChannel: np.ndarray, componentMask: np.ndarray) -> float | None:
    #     """Estimate the temperature of a component region.
    #
    #     This method is **hardware dependent** and must be implemented
    #     based on the thermal camera's pixel-to-temperature calibration.
    #
    #     Args:
    #         LChannel (np.ndarray): Luminance channel of the LAB frame.
    #         componentMask (np.ndarray): Binary mask of the component region.
    #
    #     Returns:
    #         float | None: Estimated temperature of the component, or None if unavailable.
    #     """
    #     if (self.tempDetected is False):
    #         return None
    #     componentTemp: float = cv.mean(LChannel, componentMask.astype(np.uint8))[0]
    #     return self.pixelToTemp(componentTemp)

    # ----------------------------------------------------------------
    # ----------------To Be Modified Dependant on Hardware ------------
    # ----------------------------------------------------------------
    # def updateTemps(self, frame: np.ndarray) -> None:
    #     """Update minimum and maximum temperature readings from the frame.
    #
    #     This method is **hardware dependent**. It may require OCR or
    #     camera-specific metadata extraction to obtain temperature values.
    #
    #     Args:
    #         frame (np.ndarray): Input frame from the camera.
    #
    #     Raises:
    #         TempDetectionFailed: If temperature readings cannot be extracted.
    #     """
    #     ...
    #     height, width = frame.shape[:2]

    #     ## Will Need to update based on camera if using frame temps -> These two lines are hardware dependant
    #     crop1: Frame = frame[int(height - height * 0.12):int(height - height * 0.061), int(width * 0.888):int(width * 0.96125)]
    #     crop2: Frame = frame[int(height * 0.061):int(height * 0.12), int(width * 0.82625):int(width * 0.96125)]

    #     gray1: Frame = cv.cvtColor(crop1, cv.COLOR_BGR2GRAY)
    #     gray2: Frame = cv.cvtColor(crop2, cv.COLOR_BGR2GRAY)

    #     # basic threshold
    #     _, thresh1 = cv.threshold(gray1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #     _, thresh2 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #     custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.°C'
    #     min: str = pytesseract.image_to_string(thresh1, config=custom_config)
    #     max: str = pytesseract.image_to_string(thresh2, config=custom_config)

    #     if (len(max) == 0 or len(min) == 0):
    #         raise TempDetectionFailed()

    #     match_max = re.search(r"\d{2}\.\d", max)
    #     match_min = re.search(r"\d{2}\.\d", min)

    #     self.maxTemp: float | None = float(match_max.group()) if match_max else None
    #     self.minTemp: float | None = float(match_min.group()) if match_min else None

    # def determineClusterTemp(self, centres: np.ndarray) -> None:
    #     """Determine the temperature of each k-means cluster.
    #
    #     This method is **hardware dependent** and requires a valid
    #     calibration function from pixel values to temperatures.
    #
    #     Args:
    #         centres (np.ndarray): Array of cluster centers from k-means.
    #
    #     Raises:
    #         TempDetectionFailed: If calibration parameters are missing.
    #     """
    #     if (self.hottestPixel is None or self.coldestPixel is None or self.maxTemp is None or self.minTemp is None):
    #         raise TempDetectionFailed()
    #     self.gradient =  (self.maxTemp - self.minTemp) / (self.hottestPixel - self.coldestPixel)

    #     self.intercept = self.maxTemp - self.gradient * self.hottestPixel

    #     for i in range(len(centres)):
    #         # centres looks like [[L ,A ,B], [L, A, B]]
    #         pixel_value: np.ndarray = centres[i]
    #         self.clusterTemps[i] = self.pixelToTemp(pixel_value[0])

    #     # sort in reverse order
    #     sortId: np.ndarray = np.argsort(self.clusterTemps)[::-1]
    #     self.clusterTemps = self.clusterTemps[sortId]
    #     self.pixelCounts = self.pixelCounts[sortId]
    
    # def resetTempData(self) -> None:
    #     """Reset all hardware-dependent temperature data.
    #
    #     This method is **hardware dependent**. It should reset
    #     min/max temperature values and any calibration parameters.
    #     """
    #     self.maxTemp = None
    #     self.minTemp = None
    #     self.coldestPixel = None
    #     self.hottestPixel = None
    #     self.gradient = None
    #     self.intercept = None
    #     self.clusterTemps = np.zeros(self.k)
    
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
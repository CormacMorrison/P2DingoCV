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
        self.hotSpotThreshold: float = 0.7
        self.sigmoidSteepnessDeltaP: float = 0.25
        self.sigmoidSteepnessZ: float = 0.23
        self.compactnessCutoff: float = 0.6
        # Temparature Contrast
        self.dilationSize = 5

        # HotspotScore Weightings
        self.wDeltaP: float = 0.3
        self.wZscore: float = 0.3
        self.wCompactness: float = 0.4
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
        self.frameArea = frame.shape[:2][1] * frame.shape[:2][0]
         
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
            for i in range(2):
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
                outlined, "kMeansOutlines", self.frameCount, self.logger, "diagonstics"
            )
            self.utility.saveFrame(
                segmentedBGR, "kMeansSegments", self.frameCount, self.logger, "diagonstics"
            )

        return maskArray

    def classifyHotspot(
        self, frame: Frame, mask: Frame, saveVisuals: bool
    ) -> tuple[list, bool]:
        """Classify and score hotspots from a clustered mask.

        Args:
            frame (Frame): Original BGR frame.
            mask (Frame): Binary mask of hotspot candidate regions.
            saveVisuals (bool): Whether to save annotated hotspot frames.

        Returns:
            tuple[list, bool]:
                - Sorted list of hotspot data tuples, containing:
                  (label, score, temp, centroid, ΔPscore, ΔP, z, zNorm,
                  compactness, aspect ratio, eccentricity, area).
                - Boolean indicating whether any hotspot exceeds threshold.
        """
        diagonsticFrame = frame.copy()
        filterMask = self.filterMask(mask)
        hotspotDetection: bool = False

        # Join Clusters
        kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (self.clusterJoinKernel, self.clusterJoinKernel)
        )
        closedMask: Frame = cv.morphologyEx(filterMask, cv.MORPH_CLOSE, kernel)
        n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            closedMask, connectivity=8
        )
        LABFrame: Frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        LChannel: Frame = LABFrame[:, :, 0]
        results: list = []

        for lbl in range(1, n_labels):
            componentMask: np.ndarray = ((labels == lbl) * 255).astype(np.uint8)
            otherHotspotsMask = (labels != lbl) & (labels != 0)
            componentArea = stats[lbl, cv.CC_STAT_AREA]
            
            # ------------Temparature Hardware Dependant------------------
            # Uncomment if You Wish to Implement OCR Temparature Detection
            # ------------------------------------------------------------
            # try:
            #     componentTemp = self.componentTemp(LChannel, componentMask)
            # except TempDetectionFailed:
            #     self.resetTempData()
            #     componentTemp = None
            # ------------------------------------------------------------
            
            deltaPRobust, zScore, deltaPScore, zScoreNorm, localMask = self.pixelContrast(
                LChannel, componentMask, otherHotspotsMask, componentArea
            )
            compactness, aspectRatioNorm, eccentricity = self.shapeAndCompactness(
                componentMask, componentArea
            )
            hotspotScore: float = self.hotSpotScore(
                deltaPScore=deltaPScore,
                zScoreNorm=zScoreNorm,
                compactness=compactness,
                aspectRatio=aspectRatioNorm,
                eccentricity=eccentricity,
            )
            if hotspotScore >= self.hotSpotThreshold:
                hotspotDetection = True
            cx: int; cy: int
            cx, cy = map(int, centroids[lbl])
            self.utility.drawFrameCountours(frame=diagonsticFrame, componentMask=localMask, cx=cx, cy=cy, lbl=lbl)
            if hotspotScore > self.hotSpotThreshold and saveVisuals:
                self.utility.drawFrameCountours(frame=frame, componentMask=componentMask, cx=cx, cy=cy, lbl=lbl)
                
            # component temp set to None for now
            centroidTuple = tuple(float(x) for x in centroids[lbl])
            results.append(
                (
                    lbl,
                    hotspotScore,
                    None,
                    centroidTuple,
                    deltaPScore,
                    deltaPRobust,
                    zScore,
                    zScoreNorm,
                    compactness,
                    aspectRatioNorm,
                    eccentricity,
                    float(componentArea),
                )
            )
        if saveVisuals:
            self.utility.saveFrame(frame, "hotspots", self.frameCount, self.logger, self.outputPath + "/hotspots")
            self.utility.saveFrame(diagonsticFrame, "localDilations", self.frameCount, self.logger, self.outputPath + "/localDilations")

        sortedResults = sorted(results, key=lambda row: row[1], reverse=True)
        return sortedResults, hotspotDetection
    

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

            components = []
            for i in range(1, num_labels):  # Skip background
                x, y, w, h, area = stats[i]
                if area == 0:
                    continue

                comp_mask = (labels == i).astype(np.uint8) * 255

                components.append({
                    "mask": comp_mask,
                    "bbox": (x, y, x + w, y + h),
                    "area": int(area),
                    "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                })

            all_components.append(components)

        return all_components

    def higherOrderOverlap(self, componentsFirstLayer: List[Dict[str, Any]], componentsSecondLayer: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Placeholder for overlap logic
        returnComponents: List[Dict[str, Any]] = []
        for comp1 in componentsFirstLayer:
            for comp2 in componentsSecondLayer:
                if self.boxesOverlap(comp1['bbox'], comp2['bbox']):
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

    def layeredCandidates(self, componentsArray: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        # Placeholder for overlap logic
        canididates: List[Dict[str, Any]] = []
        canididates.extend(componentsArray[0])
        
        for i in range(len(componentsArray) - 1):
            layerComponents = self.higherOrderOverlap(canididates, componentsArray[i + 1])
            canididates.extend(layerComponents)
            
        return canididates

    def hotspotMetricPass(self, frame: Frame, canididates: List[Dict[str, Any]], otherHotspotsMask: Frame) -> List[Dict[str, Any]]:
        # Placeholder for overlap logic
        LABFrame: Frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        LChannel: Frame = LABFrame[:, :, 0]
        
        
        hotspotsDetection: List[Dict[str, Any]] = []
        for candidate in canididates:
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
            
        return hotspotsDetection
    
    
    def mergeOverlappingHotspots(self, hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged = True

        # Ensure all masks are binary once
        for h in hotspots:
            h['mask'] = (h['mask'] > 0).astype(np.uint8) * 255

        while merged:
            merged = False
            result = []
            used = [False] * len(hotspots)

            for i in range(len(hotspots)):
                if used[i]:
                    continue

                hi = hotspots[i]
                cur_mask = hi['mask']
                cur_box = hi['bbox']

                for j in range(i + 1, len(hotspots)):
                    if used[j]:
                        continue

                    hj = hotspots[j]

                    if self.boxesOverlap(cur_box, hj['bbox']):
                        overlap = cv.bitwise_and(cur_mask, hj['mask'])
                        if np.any(overlap):
                            # Merge
                            cur_mask = cv.bitwise_or(cur_mask, hj['mask'])
                            cur_box = (
                                min(cur_box[0], hj['bbox'][0]),
                                min(cur_box[1], hj['bbox'][1]),
                                max(cur_box[2], hj['bbox'][2]),
                                max(cur_box[3], hj['bbox'][3]),
                            )
                            used[j] = True
                            merged = True

                # Compute area from all contours
                contours, _ = cv.findContours(cur_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                area = sum(cv.contourArea(c) for c in contours)

                # True centroid from mask
                M = cv.moments(cur_mask)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx = (cur_box[0] + cur_box[2]) / 2
                    cy = (cur_box[1] + cur_box[3]) / 2

                result.append({
                    "mask": cur_mask,
                    "bbox": cur_box,
                    "area": int(area),
                    "centroid": (float(cx), float(cy)),
                })

            hotspots = result

        return hotspots

    def finalMetricPass(self, frame: Frame, hotspots: List[Dict[str, Any]], otherHotspotMaskFinal: Frame) -> List[Tuple]:
        # Placeholder for overlap logic
        LABFrame: Frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        LChannel: Frame = LABFrame[:, :, 0]
        
        
        results: List = []
        for i, hotspot in enumerate(hotspots):
            deltaPRobust, zScore, deltaPScore, zScoreNorm, localMask = self.pixelContrast(
                LChannel, hotspot['mask'], np.zeros_like(hotspot['mask']), hotspot['area']
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

        diagonsticFrame = frame.copy()
        hotSpotFrame = frame.copy()
        filterMaskArray = [self.filterMask(mask) for mask in maskArray]
        hotspotDetection: bool = False
        
        kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (self.clusterJoinKernel, self.clusterJoinKernel)
        )
        closedMaskArray: List[Frame] = [cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel) for mask in filterMaskArray]
        # Erode again to seperate joined regions
        closedMaskArray = [cv.erode(mask, kernel, iterations=1) for mask in closedMaskArray]

        componentsArray = self.connectedComponentsFromMask(closedMaskArray)

        candiadateHotspots = self.layeredCandidates(componentsArray) 
        
        otherHotspotsMaskPrelim = filterMaskArray[0].copy()
        
        hotspots = self.hotspotMetricPass(frame, candiadateHotspots, otherHotspotsMaskPrelim)
        
        hotspots = self.mergeOverlappingHotspots(hotspots)
        
        otherHotspotMaskFinal = np.zeros_like(maskArray[0])
        for hotspot in hotspots:
            otherHotspotMaskFinal = cv.bitwise_or(otherHotspotMaskFinal, hotspot['mask'])
            
        
            
        finalResults = self.finalMetricPass(frame, hotspots, otherHotspotMaskFinal)
        
        sortedResults = sorted(finalResults, key=lambda row: row[1], reverse=True)
        
        if (len(sortedResults) > 0):
            hotspotDetection = True
            
        if saveVisuals:
            for hotspot in hotspots:
                cx: int; cy: int
                cx, cy = map(int, hotspot['centroid'])
                self.utility.drawFrameCountours(frame=frame, componentMask=hotspot['mask'], cx=cx, cy=cy, lbl=0)
                
            self.utility.saveFrame(frame, "hotspots", self.frameCount, self.logger, self.outputPath + "/hotspots")
            self.utility.saveFrame(diagonsticFrame, "localDilations", self.frameCount, self.logger, self.outputPath + "/localDilations")
           
        return sortedResults, hotspotDetection




    def boxesOverlap(self, b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)

        return (xb > xa) and (yb > ya)


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
            if compArea < self.frameArea / 5000:
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

    def pixelContrast(
        self,
        LChannel: Frame,
        componentMask: Frame,
        otherHotspotsMask: Frame,
        area: float,
    ) -> tuple[float, float, float, float, Frame]:
        """Compute contrast metrics for a hotspot region.

        Args:
            LChannel (Frame): Luminance channel of LAB frame.
            componentMask (Frame): Mask of the region of interest.
            otherHotspotsMask (Frame): Mask of other hotspots to exclude.
            area (float): Pixel area of the region.

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

        hotVals: Frame = LChannel[hotMask]
        localVals: Frame = LChannel[localMask]

        if len(hotVals) == 0:
            return 0, 0, 0, 0, localMask
        elif len(localVals) == 0:
            return 10, 10, 1, 1, localMask

        muHot: float = hotVals.mean()
        mu: float = LChannel.mean()
        sigma: float = LChannel.std()
        medianLocal: float = np.median(localVals).astype(float)
        iqrLocal: float = (
            np.percentile(localVals, 75) - np.percentile(localVals, 25)
        ).astype(float)

        # Robust Contrast
        deltaPRobust: float = (muHot - medianLocal) / (iqrLocal + 1e-6)

        deltaPRobustClipped: float = np.clip(deltaPRobust, -50, 50)

        # Robust Constrast Probability

        deltaPProbabilityScore: float = np.clip(
            np.exp(self.sigmoidSteepnessDeltaP * deltaPRobustClipped) - 1, 0, 1
        )

        # Global Z score
        z = (muHot - mu) / sigma
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
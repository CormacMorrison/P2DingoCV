import re
import cv2 as cv
import numpy as np
from typing import Final, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib
# import pytesseract
from datetime import datetime
import os
from abc import ABC, abstractmethod

from Camera.Camera import Camera
from HotspotLogic.Exceptions.TempDetectionFail import TempDetectionFailed 
from HotspotLogic.Types.Types import *
from DiagonsticsUtil import VisualUtils


# DEV
matplotlib.use('Agg') 


class HotspotDetector:
    def __init__(self, cam: Camera, exitPath: str = f"{os.getcwd()}/hotspotDetection") -> None:
        self.cam: Camera = cam
        self.utility = VisualUtils(exitPath)
        self.exitPath = exitPath    

        self.labFrame: Frame
        self.frame: np.ndarray 
        self.frameArea: int
        self.outputPath: str
        self.frameCount: int = 0
        # Paramaters To Tune
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
        # self.clusterTemps: np.ndarray = np.zeros(self.k)
        self.pixelCounts: np.ndarray = np.zeros(self.k)

        self.colours = [
            (255, 0, 0),   # Blue
            (0, 255, 0),   # Green
            (0, 0, 255),   # Red
            (255, 255, 0), # Cyan
            (255, 0, 255), # Magenta
            (0, 255, 255)  # Yellow
        ]
        
               
        # Temp Data
        # self.maxTemp: float | None = 50 # starter values
        # self.minTemp: float | None = 10 # starter values
        # self.tempDetected: bool = False
        # self.coldestPixel: float | None = 10 # hardware dependant
        # self.hottestPixel: float | None = 255 # hardware dependant
        # self.gradient: float | None
        # self.intercept: float | None
            
    @abstractmethod 
    def execute(self):
        pass
    
    def perFrameProcessing(self, frame: Frame) -> tuple[list, bool]:
        self.frame: Frame = frame
        self.labFrame: Frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        frame = self.perFrameSetup(frame)
        mask: Frame = self.kMeansThermalGrouping(frame, False)
        return self.classifyHotspot(frame, mask, True)
            
    def perFrameSetup(self, frame: Frame) -> Frame:
        # Calcualte frameArea -> hardware dependant
        self.resetFrameData()
        # self.resetTempData()
        self.tempDetected = True
        self.frameArea = frame.shape[:2][1] * frame.shape[:2][0]
        # try:
        #     self.updateTemps(frame)
        # except TempDetectionFailed:
        #     self.resetTempData()
        #     self.tempDetected = False
        return frame 
    
    def kMeansThermalGrouping(self, frame: Frame, saveDiagonstics: bool) -> Frame:
        original = frame.copy()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        pixels = frame.reshape((-1, 3)).astype(np.float32)
        
        # Define criteria and number of clusters (k)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        
        bestLabels = np.empty((pixels.shape[0], 1), dtype=np.int32)
        _, labels, centers = cv.kmeans(pixels, self.k, bestLabels, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        
        # Determine Pixel Counts 
        self.pixelCounts = np.bincount(labels.flatten(), minlength=self.k)
        
        # Temparature Hardware Dependant
        # try:
        #     self.determineClusterTemp(centers)
        # except TempDetectionFailed:
        #     self.resetTempData()
        
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
        #sort by intensity
        sortedIndices: np.ndarray = np.argsort(centers[:, 0])[::-1]

        # Mask for Highest Intensity Region 
        mask: Frame = (labels.reshape(frame.shape[:2]) == sortedIndices[0]).astype(np.uint8) * 255
        # self.saveFrame(mask)

        for i in range(2):
            mask2: Frame = (labels.reshape(frame.shape[:2]) == sortedIndices[i]).astype(np.uint8) * 255
            contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(outlined, contours, -1, self.colours[i % len(self.colours)], 2)
        # Outlines of top 2 k regions and BGR Segments
        if (saveDiagonstics == True):
            self.utility.saveFrame(outlined, "kMeansOutlines", self.frameCount, "diagonstics")
            self.utility.saveFrame(segmentedBGR, "kMeansSegments", self.frameCount, "diagonstics")
            
        return mask

    def classifyHotspot(self, frame: np.ndarray, mask: np.ndarray, saveVisuals: bool) -> tuple[list, bool]:
        filterMask = self.filterMask(mask)
        hotspotDetection: bool = False
        
        #Join Clusters
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (self.clusterJoinKernel, self.clusterJoinKernel))
        closedMask: Frame = cv.morphologyEx(filterMask, cv.MORPH_CLOSE, kernel)
        n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(closedMask, connectivity=8)
        LABFrame: Frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        LChannel: Frame = LABFrame[:, :, 0]
        results: list = []

        for lbl in range(1, n_labels):
            componentMask: np.ndarray = ((labels == lbl) * 255).astype(np.uint8)
            otherHotspotsMask = (labels != lbl) & (labels != 0) 
            componentArea = stats[lbl, cv.CC_STAT_AREA]
            # try:
            #     componentTemp = self.componentTemp(LChannel, componentMask)
            # except TempDetectionFailed:
            #     self.resetTempData()
            #     componentTemp = None
            deltaPRobust, zScore, deltaPScore, zScoreNorm = self.pixelContrast(LChannel, componentMask, otherHotspotsMask, componentArea)
            compactness, aspectRatioNorm, eccentricity = self.shapeAndCompactness(componentMask, componentArea)
            hotspotScore = self.hotSpotScore(deltaPScore=deltaPScore,
                                             zScoreNorm=zScoreNorm, compactness=compactness, 
                                             aspectRatio=aspectRatioNorm, 
                                             eccentricity=eccentricity)
            if (hotspotScore >= self.hotSpotThreshold):
                hotspotDetection = True
            
            if (hotspotScore > self.hotSpotThreshold):
                cv.drawContours(frame, [cv.findContours(componentMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]],               
                                -1, self.colours[0], 1)
                cx, cy = map(int, centroids[lbl])
                cv.putText(frame, f"{lbl:.2f}", (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
            #component temp set to None for now
            results.append((lbl, hotspotScore, None, centroids[lbl], deltaPScore, deltaPRobust, zScore, zScoreNorm, compactness, aspectRatioNorm, eccentricity, componentArea))
        if saveVisuals:
            self.utility.saveFrame(frame, "hotspots", self.frameCount, self.outputPath)
        
        sortedResults = sorted(results, key=lambda row: row[1], reverse=True)
        return sortedResults, hotspotDetection

    # Filters out Noise
    def filterMask(self, mask: np.ndarray):
        n_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
        filteredMask = np.zeros_like(mask)

        for lbl in range(1, n_labels):
            compArea = stats[lbl, cv.CC_STAT_AREA]
            if (compArea < self.frameArea / 5000):
                continue

            filteredMask[labels == lbl] = 255
        
        return filteredMask
    
    def shapeAndCompactness(self, componentMask: Frame, area: float) -> tuple[float, float, float]:
        contours, _ = cv.findContours(componentMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Should only be one contour 
        perimeter: float = cv.arcLength(contours[0], True)

        # Elipse Required 5 pixels already been filtered for size
        
        eccentricity: None | float = None
        if (len(contours[0]) > 5):
            (_, _), (major_axis, minor_axis), _ = cv.fitEllipse(contours[0])
            eccentricity = np.sqrt(1 - (min(major_axis, minor_axis) / max(major_axis, minor_axis))**2)
        _, _, w, h = cv.boundingRect(contours[0])

        # square has 0.785 compactness normalised by definition
        compactness: float = (4 * np.pi * area) / (perimeter ** 2)
        aspectRatioNorm: float = min(w, h)/max(w, h)
        compactness = np.clip(compactness / self.compactnessCutoff, 0, 1)
        
        if (eccentricity == None):
            eccentricity = aspectRatioNorm
        # normalised by definition

        return compactness, aspectRatioNorm, eccentricity 
    
    def pixelContrast(self, LChannel: Frame, componentMask: Frame, otherHotspotsMask: Frame, area: float) -> tuple[float, float, float, float]:
        ksize: int = int(max(3, self.dilationSize * np.sqrt(area)))
        kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv.dilate(componentMask, kernel, iterations=1)

        hotMask: Frame = componentMask.astype(bool)
        localMask: Frame = (dilated.astype(bool)) & (~hotMask) & (~otherHotspotsMask)

        hotVals: Frame = LChannel[hotMask]
        localVals: Frame = LChannel[localMask]

        if len(hotVals) == 0:
            return 0, 0, 0, 0
        # should not be possible but if so it max temp diff
        elif len(localVals) == 0:
            return 10, 10, 1, 1

        muHot: float = hotVals.mean()
        mu: float = LChannel.mean()
        sigma: float = LChannel.std()
        medianLocal: float = np.median(localVals).astype(float)
        iqrLocal: float = (np.percentile(localVals, 75) - np.percentile(localVals, 25)).astype(float)

        #Robust Contrast
        deltaPRobust: float = (muHot - medianLocal) / (iqrLocal + 1e-6)
        
        #Robust Constrast Probability  
        
        deltaPProbabilityScore: float = np.clip(np.exp(self.sigmoidSteepnessDeltaP * deltaPRobust) - 1, 0, 1)

        #Global Z score
        z = (muHot - mu) / sigma
        zScoreNorm: float = np.clip(np.exp(self.sigmoidSteepnessZ * z) - 1, 0, 1)

        return deltaPRobust, z, deltaPProbabilityScore, zScoreNorm

    def hotSpotScore(self, deltaPScore: float, zScoreNorm: float, compactness: float, aspectRatio: float, eccentricity: float):
        # Normalise Z score
        return deltaPScore * self.wDeltaP + zScoreNorm * self.wZscore + compactness * self.wCompactness + aspectRatio * self.wAspectRatio + (eccentricity) * self.wEccentricity
    
    
    # def componentTemp(self, LChannel: np.ndarray, componentMask: np.ndarray) -> float | None:
    #     if (self.tempDetected is False):
    #         return None
    #     componentTemp: float = cv.mean(LChannel, componentMask.astype(np.uint8))[0]
    #     return self.pixelToTemp(componentTemp)
    
    # ---------------------------------------------------------------- 
    #----------------To Be Modified Dependant on Hardware ------------
    # ---------------------------------------------------------------- 
    # def updateTemps(self, frame: np.ndarray) -> None:
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
    
    def processingDiagonstics(self) -> None:
        # self.utility.plotFreqArray(self.clusterTemps, "Temp of Clusters")
        self.utility.plotFreqArray(self.pixelCounts, "Pixel Count of Clusters")
        

    # ----------------------------------------------------------------
    # --------------------Hotspot Helpers-----------------------------
    # ---------------------------------------------------------------- 

    # ---------------------------------------------------------------- 
    #----------------To Be Modified Dependant on Hardware ------------
    # ---------------------------------------------------------------- 
    def pixelToTemp(self, pixelVal) -> float:
        if (self.gradient is None or self.intercept is None):
            raise TempDetectionFailed()
        return self.gradient * pixelVal + self.intercept 
    
    # def resetTempData(self) -> None:
    #     self.maxTemp = None
    #     self.minTemp = None 
    #     self.coldestPixel = None
    #     self.hottestPixel = None
    #     self.gradient = None
    #     self.intercept = None
    #     self.clusterTemps = np.zeros(self.k)
    
    def resetFrameData(self) -> None:
        self.maxTemp = None
        self.minTemp = None
        self.coldestPixel = 10 # TEMP
        self.hottestPixel = 255 #TEMP
        self.gradient = None
        self.intercept = None
        self.labFrame = np.zeros(0)
        self.frame = np.zeros(0)
        self.frameArea = 0
            # Data Points based on k
        self.clusterTemps: np.ndarray = np.zeros(self.k)
        self.pixelCounts: np.ndarray = np.zeros(self.k)

    # ---------------------------------------------------------------- 
    # ---------------------------------------------------------------- 
    # ---------------------------------------------------------------- 

        
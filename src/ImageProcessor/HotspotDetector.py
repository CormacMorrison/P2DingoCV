import re
import cv2 as cv
import numpy as np
from typing import Final, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib
import pytesseract
from datetime import datetime
import os

from Camera.Camera import Camera
from ImageProcessor.ImageProcessor import ImageProcessor
from ImageProcessor.Exceptions.TempDetectionFail import TempDetectionFailed 
from Types.Types import *


# DEV
matplotlib.use('Agg') 


class HotspotDetector:
    def __init__(self, cam: Camera, processor: ImageProcessor) -> None:
        self.cam: Camera = cam
        self.proccessor = processor 
       
        #Per Image Paramaters
        self.maxTemp: float | None = 50 # starter values
        self.minTemp: float | None = 10 # starter values
        self.tempDetected: bool = False
        self.coldestPixel: float | None = 10 # hardware dependant
        self.hottestPixel: float | None = 255 # hardware dependant
        self.gradient: float | None
        self.intercept: float | None
        self.labFrame: Frame
        self.frame: np.ndarray 
        self.frameArea: int
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

        #Extra Data 
        self.clusterTemps: np.ndarray = np.zeros(self.k)
        self.pixelCounts: np.ndarray = np.zeros(self.k)

        self.colours = [
            (255, 0, 0),   # Blue
            (0, 255, 0),   # Green
            (0, 0, 255),   # Red
            (255, 255, 0), # Cyan
            (255, 0, 255), # Magenta
            (0, 255, 255)  # Yellow
        ]
            
    
    def execute(self):
        results: list = []
        frameCount: int = 0
        headers: list = [
                    "lbl", 
                    "hotspotScore", 
                    "componentTemp", 
                    "centroid", 
                    "deltaPScore", 
                    "deltaPRobust", 
                    "zScore", 
                    "zScoreNorm",
                    "compactness", 
                    "aspectRatioNorm", 
                    "eccentricity",
                    "area"
                ]
        while True:
            self.resetFrameData()
            frame: Frame | None = self.cam.read()
            if frame is None:
                 break  # Stop if no frame was captured
                
            self.frame: Frame = frame
            self.labFrame: Frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
            detection: bool = True
            frame = self.perFrameSetup(frame)
            mask: Frame = self.kMeansThermalGrouping(frame)
            components, detection = self.classifyHotspot(frame, mask)

            frameEntry = {
                "frame": frameCount,
                "detection": detection,
                "components": {}
            }

            for i, comp in enumerate(components):
                compDict = dict(zip(headers, comp))
                frameEntry["components"][i] = compDict

                # self.plotFreqArray(self.clusterTemps, "CLusterTemp")
                # self.plotFreqArray(self.pixelCounts, "Pixel Counts")
            results.append(frameEntry)
            frameCount+=1

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
    def perFrameSetup(self, frame: Frame) -> Frame:
        # Calcualte frameArea -> hardware dependant
        self.tempDetected = True
        self.frameArea = frame.shape[:2][1] * frame.shape[:2][0]
        try:
            self.updateTemps(frame)
        except TempDetectionFailed:
            self.resetTempData()
            self.tempDetected = False
        return frame 
    
    def kMeansThermalGrouping(self, frame: Frame) -> Frame:
        original = frame.copy()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        pixels = frame.reshape((-1, 3)).astype(np.float32)
        
        # Define criteria and number of clusters (k)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        
        bestLabels = np.empty((pixels.shape[0], 1), dtype=np.int32)
        _, labels, centers = cv.kmeans(pixels, self.k, bestLabels, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        
        # Determine Pixel Counts 
        self.pixelCounts = np.bincount(labels.flatten(), minlength=self.k)
        
        # Convert centers to uint8 and labels to clustered image
        try:
            self.determineClusterTemp(centers)
        except TempDetectionFailed:
            self.resetTempData()
        
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

        # self.saveFrame(outlined) 
        # self.saveFrame(outlined)
        # self.saveFrame(segmentedBGR)
            
        return mask

    def classifyHotspot(self, frame: np.ndarray, mask: np.ndarray) -> tuple[list, bool]:
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
            try:
                componentTemp = self.componentTemp(LChannel, componentMask)
            except TempDetectionFailed:
                self.resetTempData()
                componentTemp = None
            deltaPRobust, zScore, deltaPScore, zScoreNorm = self.pixelContrast(LChannel, componentMask, otherHotspotsMask, componentArea)
            compactness, aspectRatioNorm, eccentricity = self.shapeAndCompactness(componentMask, componentArea)
            hotspotScore = self.hotSpotScore(deltaPScore=deltaPScore,
                                             zScoreNorm=zScoreNorm, compactness=compactness, 
                                             aspectRatio=aspectRatioNorm, 
                                             eccentricity=eccentricity)
            if (hotspotScore >= self.hotSpotThreshold):
                hotspotDetection = True
            
            # if (hotspotScore > self.hotSpotThreshold):
            cv.drawContours(frame, [cv.findContours(componentMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]],               
                                -1, self.colours[0], 1)
            cx, cy = map(int, centroids[lbl])
            cv.putText(frame, f"{lbl:.2f}", (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
            results.append((lbl, hotspotScore, componentTemp, centroids[lbl], deltaPScore, deltaPRobust, zScore, zScoreNorm, compactness, aspectRatioNorm, eccentricity, componentArea))
        self.saveFrame(frame)
        
        sortedResults = sorted(results, key=lambda row: row[1], reverse=True)
        return sortedResults, hotspotDetection

    # Filters out Noise
    def filterMask(self, mask: np.ndarray):
        n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
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
    

    def componentTemp(self, LChannel: np.ndarray, componentMask: np.ndarray) -> float | None:
        if (self.tempDetected is False):
            return None
        componentTemp: float = cv.mean(LChannel, componentMask.astype(np.uint8))[0]
        return self.pixelToTemp(componentTemp)

    def hotSpotScore(self, deltaPScore: float, zScoreNorm: float, compactness: float, aspectRatio: float, eccentricity: float):
        # Normalise Z score
        return deltaPScore * self.wDeltaP + zScoreNorm * self.wZscore + compactness * self.wCompactness + aspectRatio * self.wAspectRatio + (eccentricity) * self.wEccentricity
    

    
    # ---------------------------------------------------------------- 
    #----------------To Be Modified Dependant on Hardware ------------
    # ---------------------------------------------------------------- 
    def updateTemps(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        
        ## Will Need to update based on camera if using frame temps -> These two lines are hardware dependant
        crop1: Frame = frame[int(height - height * 0.12):int(height - height * 0.061), int(width * 0.888):int(width * 0.96125)]
        crop2: Frame = frame[int(height * 0.061):int(height * 0.12), int(width * 0.82625):int(width * 0.96125)]
        
        gray1: Frame = cv.cvtColor(crop1, cv.COLOR_BGR2GRAY)
        gray2: Frame = cv.cvtColor(crop2, cv.COLOR_BGR2GRAY)
        
        # basic threshold
        _, thresh1 = cv.threshold(gray1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, thresh2 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.°C'
        min: str = pytesseract.image_to_string(thresh1, config=custom_config)
        max: str = pytesseract.image_to_string(thresh2, config=custom_config)

        if (len(max) == 0 or len(min) == 0):
            raise TempDetectionFailed()
        
        match_max = re.search(r"\d{2}\.\d", max)
        match_min = re.search(r"\d{2}\.\d", min)

        self.maxTemp: float | None = float(match_max.group()) if match_max else None
        self.minTemp: float | None = float(match_min.group()) if match_min else None 
        
    def determineClusterTemp(self, centres: np.ndarray) -> None:
        if (self.hottestPixel is None or self.coldestPixel is None or self.maxTemp is None or self.minTemp is None):
            raise TempDetectionFailed()
        self.gradient =  (self.maxTemp - self.minTemp) / (self.hottestPixel - self.coldestPixel)  
        
        self.intercept = self.maxTemp - self.gradient * self.hottestPixel 
        
        for i in range(len(centres)):
            # centres looks like [[L ,A ,B], [L, A, B]]
            pixel_value: np.ndarray = centres[i]
            self.clusterTemps[i] = self.pixelToTemp(pixel_value[0])            
        
        # sort in reverse order 
        sortId: np.ndarray = np.argsort(self.clusterTemps)[::-1]
        self.clusterTemps = self.clusterTemps[sortId]
        self.pixelCounts = self.pixelCounts[sortId]
    
    def determineIfHotspot(self) -> bool:
        self.plotFreqArray(self.clusterTemps, "Temp")
        self.plotFreqArray(self.pixelCounts, "Count")
        
        return True 

    # def outputData()

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
    
    def resetTempData(self) -> None:
        self.maxTemp = None
        self.minTemp = None 
        self.coldestPixel = None
        self.hottestPixel = None
        self.gradient = None
        self.intercept = None
        self.clusterTemps = np.zeros(self.k)
    
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


    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        assert frame is not None, "file could not be read, check with os.path.exists()"

        # Determine number of channels
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            # Grayscale image
            print("Detected grayscale image")
            gray = frame if len(frame.shape) == 2 else frame[:, :, 0]
            gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])

            plt.figure(figsize=(8, 4))
            plt.title("Grayscale Histogram")
            plt.plot(gray_hist, color='k')
            plt.xlim([0, 256])
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.savefig("histogram_gray.png")
            plt.close()

        elif frame.shape[2] == 3:
            # BGR image
            print("Detected BGR image")
            color = ('b', 'g', 'r')
            plt.figure(figsize=(8, 4))
            plt.title("BGR Histogram")
            for i, col in enumerate(color):
                histr = cv.calcHist([frame], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.savefig("histogram_bgr.png")
            plt.close()

        else:
            raise ValueError("Unsupported image format")

        cv.imshow("frame", frame)
        return frame


    def plotFreqArray(self, arr: np.ndarray, title: str, show_plot: bool = True) -> None:
        """
        Plot a 1D array as a bar chart and save it to 'plots/' directory.

        Args:
            arr (np.ndarray): Array to plot
            title (str): Title of the plot (also used as filename)
            show_plot (bool): Whether to display the plot interactively
        """
        # Ensure the directory exists
        os.makedirs('plots', exist_ok=True)

        # Clear any existing figure
        plt.figure()

        # Plot the array
        plt.bar(range(len(arr)), arr)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(title)

        # Make a safe filename
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        filepath = f'plots/{safe_title}.png'

        # Save the figure
        plt.savefig(filepath)

        # Show the plot if requested
        if show_plot:
            plt.show()

        # Close the figure to free memory
        plt.close()

    def saveFrame(self, frame: np.ndarray, folder: str = "frames"):
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(folder, f"frame_{timestamp}.png")
        
        # Save the frame
        cv.imwrite(filename, frame)
        
        print(f"Frame saved to {filename}")

def print_print_list(data, headers=None, precision=2):
    """
    Pretty-print a list of lists (or tuples) as a table with borders.
    
    Parameters:
    - data: list of lists or tuples
    - headers: optional list of column headers
    - precision: decimal places for numbers
    """
    # Convert all values to strings and measure column widths
    str_data = []
    col_widths = []

    for row in data:
        str_row = []
        for val in row:
            if isinstance(val, (tuple, list)):
                val_str = "(" + ", ".join(f"{v:.{precision}f}" if isinstance(v, (int, float)) else str(v) for v in val) + ")"
            elif isinstance(val, (int, float)):
                val_str = f"{val:.{precision}f}"
            else:
                val_str = str(val)
            str_row.append(val_str)
        str_data.append(str_row)

    # Determine max width for each column
    num_cols = len(str_data[0])
    if headers:
        for i in range(num_cols):
            max_data_len = max(len(str_data[r][i]) for r in range(len(str_data)))
            col_widths.append(max(max_data_len, len(headers[i])))
    else:
        for i in range(num_cols):
            max_data_len = max(len(str_data[r][i]) for r in range(len(str_data)))
            col_widths.append(max_data_len)

    # Function to print a row with borders
    def print_row(row):
        line = "| " + " | ".join(f"{val:{col_widths[i]}}" for i, val in enumerate(row)) + " |"
        print(line)

    # Print header
    if headers:
        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        print(sep)
        print_row(headers)
        print(sep)

    # Print rows
    for row in str_data:
        print_row(row)
        if headers:
            print(sep)
      
                
        
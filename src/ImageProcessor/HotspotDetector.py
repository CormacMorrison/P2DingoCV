import re
import cv2 as cv
import numpy as np
from typing import Final, Tuple
import matplotlib.pyplot as plt
import matplotlib
import pytesseract
from datetime import datetime
import os

from Camera.Camera import Camera
from ImageProcessor.ImageProcessor import ImageProcessor
from ImageProcessor.Exceptions.TempDetectionFail import TempDetectionFailed 

# DEV
matplotlib.use('Agg') 


class HotspotDetector:
    def __init__(self, cam: Camera, processor: ImageProcessor) -> None:
        self.cam: Camera = cam
        self.proccessor = processor 
        self.maxTemp: float | None = 50 # to update
        self.minTemp: float | None = 10 # to update
        self.coldestPixel: float | None = 10 # hardware dependant
        self.hottestPixel: float | None = 255 # hardware dependant
        self.gradient: float | None
        self.intercept: float | None
        self.labFrame: np.ndarray
        self.frame: np.ndarray
        self.frameArea: int

        # Paramaters To Tune
        self.k: int = 10
        # Temparature Contrast 
        self.dilationSize = 10

        # Data Points based on k
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
        while True:
            frame = self.cam.read()
            # if frame is None:
            #     # break  # Stop if no frame was captured
                
            if frame is not None: 
                self.frame = frame
                self.labFrame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
                frame = self.perFrameSetup(frame)
                mask = self.kMeansThermalGrouping(frame)
                annotated, results = self.tempContrast(frame, mask)
                print(results)
                self.saveFrame(annotated)
                self.saveFrame(mask)
                # self.plotFreqArray(self.clusterTemps, "CLusterTemp")
                # self.plotFreqArray(self.pixelCounts, "Pixel Counts")
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
    def perFrameSetup(self, frame: np.ndarray) -> np.ndarray:
        # Calcualte frameArea -> hardware dependant
        self.frameArea = frame.shape[:2][1] * frame.shape[:2][0]
        try:
            self.updateTemps(frame)
        except TempDetectionFailed:
            self.resetTempData()
        return frame 
    
    def kMeansThermalGrouping(self, frame: np.ndarray) -> np.ndarray:
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
        centers: np.ndarray = centers.astype(np.uint8)
        #make int to satisfy python labels are ints corresponding to their respective centre
        labels = labels.flatten().astype(int)
        segmented = centers[labels]  

        # Reshape to original image shape
        segmentedImg = segmented.reshape(frame.shape)

        # Optional: convert back to BGR for visualization
        segmentedBGR = cv.cvtColor(segmentedImg, cv.COLOR_LAB2BGR)
        
        outlined = original.copy()
        #sort by intensity
        sortedIndices = np.argsort(centers[:, 0])[::-1]

        # Mask for Highest Intensity Region 
        mask = (labels.reshape(frame.shape[:2]) == sortedIndices[0]).astype(np.uint8) * 255
        # self.saveFrame(mask)

        for i in range(2):
            mask2 = (labels.reshape(frame.shape[:2]) == sortedIndices[i]).astype(np.uint8) * 255
            contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(outlined, contours, -1, self.colours[i % len(self.colours)], 2)

        self.saveFrame(outlined) 
        #self.saveFrame(outlined)
        # self.saveFrame(segmentedBGR)
            
        return mask

    def findConnectedComponents(self, mask: np.ndarray):
        n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
        print("Number of labels:", n_labels)
        print("Stats:\n", stats)
        print("Centroids:\n", centroids)
        label_viz = cv.applyColorMap((labels * 50).astype(np.uint8), cv.COLORMAP_JET)
        # self.saveFrame(label_viz)
        cv.waitKey(0)
    
    def tempContrast(self, frame: np.ndarray, mask: np.ndarray):
        n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
        LABFrame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        # I want to go over every component, dilate them 100% and calculate the median in the dilated region
        # (from LAB pixel intensity) the mean of the hot area and the IQR of the dilated region
        # \Delta T_{\text{robust}} = \frac{\mu_{\text{hot}} - \text{median}(\text{local})}{\text{IQR}(\text{local}) + \epsilon}
        # I then want a list of the T robusts of each unit and to highlight them on the image
        L = LABFrame[:, :, 0]
        # lbl, robustT, sigma, centroid
        results = []

        for lbl in range(1, n_labels): # skip background label 0
            # Get component mask
            componentMask = (labels == lbl).astype(np.uint8)

            # Dilate mask
            compArea = stats[lbl, cv.CC_STAT_AREA]
            if (compArea < self.frameArea / 10000):
                continue
            # choose dilation size ~ sqrt(area) // 2
            ksize = int(max(3, self.dilationSize * np.sqrt(compArea)))
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
            dilated = cv.dilate(componentMask, kernel, iterations=1)
            otherHotspotsMask = (labels != lbl) & (labels != 0) 

            hotMask = componentMask.astype(bool)
            
            # Calculat the area of this and do checks
            localMask = (dilated.astype(bool)) & (~hotMask) & (~otherHotspotsMask)

            hotVals = L[hotMask]
            localVals = L[localMask]

            if len(hotVals) == 0 or len(localVals) == 0:
                continue

            muHot: float = hotVals.mean()
            mu: float = L.mean()
            sigma: float = L.std()
            medianLocal: float = np.median(localVals).astype(float)
            iqrLocal: float = (np.percentile(localVals, 75) - np.percentile(localVals, 25)).astype(float)

            #Robust Contrast
            deltaTRobust = (muHot - medianLocal) / (iqrLocal + 1e-6)

            #Global Z score
            z = (muHot - mu) / sigma

            cx, cy = map(int, centroids[lbl])
            results.append((lbl, deltaTRobust, z, (cx, cy)))

            cv.putText(frame, f"{deltaTRobust:.2f}", (cx, cy),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
            cv.drawContours(frame, [cv.findContours(componentMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]],
                        -1, (0, 255, 0), 1)
            
        return frame, results




    
    # ---------------------------------------------------------------- 
    #----------------To Be Modified Dependant on Hardware ------------
    # ---------------------------------------------------------------- 
    def updateTemps(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        
        ## Will Need to update based on camera -> These two lines are hardware dependant
        crop1 = frame[int(height - height * 0.12):int(height - height * 0.061), int(width * 0.888):int(width * 0.96125)]
        crop2 = frame[int(height * 0.061):int(height * 0.12), int(width * 0.82625):int(width * 0.96125)]
        
        gray1 = cv.cvtColor(crop1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(crop2, cv.COLOR_BGR2GRAY)
        
        # basic threshold
        _, thresh1 = cv.threshold(gray1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, thresh2 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.°C'
        min = pytesseract.image_to_string(thresh1, config=custom_config)
        max = pytesseract.image_to_string(thresh2, config=custom_config)

        if (len(max) == 0 or len(min) == 0):
            raise TempDetectionFailed()
        
        match_max = re.search(r"\d{2}\.\d", max)
        match_min = re.search(r"\d{2}\.\d", min)

        self.maxTemp = float(match_max.group()) if match_max else None
        self.minTemp = float(match_min.group()) if match_min else None 
        
    def determineClusterTemp(self, centres: np.ndarray) -> None:
        if (self.hottestPixel is None or self.coldestPixel is None or self.maxTemp is None or self.minTemp is None):
            raise TempDetectionFailed()
        self.gradient =  (self.maxTemp - self.minTemp) / (self.hottestPixel - self.coldestPixel)  
        
        self.intercept = self.maxTemp - self.gradient * self.hottestPixel 
        
        for i in range(len(centres)):
            # centres looks like [[L ,A ,B], [L, A, B]]
            pixel_value = centres[i]
            self.clusterTemps[i] = self.pixelToTemp(pixel_value[0])            
       
        # sort in reverse order 
        sortId = np.argsort(self.clusterTemps)[::-1]
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

                
                
        
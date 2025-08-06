import re
import cv2 as cv
import numpy as np
from typing import Final, Tuple
import matplotlib.pyplot as plt
import matplotlib
import pytesseract
import time

from Camera.Camera import Camera
from ImageProcessor.ImageProcessor import ImageProcessor

matplotlib.use('TkAgg') 
class HotspotDetector:
    def __init__(self, cam: Camera, processor: ImageProcessor) -> None:
        self.cam: Camera = cam
        self.proccessor = processor 
        self.upperThresh = 255 # figure out
        self.lowerThresh = 230 # figure out
        self.maxTemp = 50 # to update
        self.minTemp = 10 # to update
        
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
                frame = self.perFrameSetup(frame)
                self.kMeansThermalGrouping(frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
    def perFrameSetup(self, frame: np.ndarray) -> np.ndarray:
        self.updateTemps(frame)
        return frame
    
    def kMeansThermalGrouping(self, frame: np.ndarray):
        original = frame.copy()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        pixels = frame.reshape((-1, 3)).astype(np.float32)
        
        # Define criteria and number of clusters (k)
        k = 10  # for example, 4 clusters
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        
        bestLabels = np.empty((pixels.shape[0], 1), dtype=np.int32)
        _, labels, centers = cv.kmeans(pixels, k, bestLabels, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8 and labels to clustered image
        centers: np.ndarray = centers.astype(np.uint8)
        labels = labels.flatten().astype(int)
        segmented = centers[labels]  

        # Reshape to original image shape
        segmentedImg = segmented.reshape(frame.shape)

        # Optional: convert back to BGR for visualization
        segmentedBGR = cv.cvtColor(segmentedImg, cv.COLOR_LAB2BGR)
        
        outlined = original.copy()
        #sort by intensity
        sorted_indices = np.argsort(centers[:, 0])[::-1]
        
        for i in range(2):
            mask = (labels.reshape(frame.shape[:2]) == sorted_indices[i]).astype(np.uint8) * 255
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(outlined, contours, -1, self.colours[i % len(self.colours)], 2)
            
        cv.imshow('Outlined', outlined)
        cv.imshow('Original', frame)
        cv.imshow('Segmented', segmentedBGR)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        return segmentedBGR
    
    # This is OCR right now but I'm hoping I can get direct output from the camera on max temps 
    def updateTemps(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        ## Will Need to update based on camera
        block1 = frame[int(height * 0.80):, int(width * 0.8):]
        block2 = frame[:int(height * 0.20), int(width * 0.8):]
        gray1 = cv.cvtColor(block1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(block2, cv.COLOR_BGR2GRAY)
        
        # basic threshold
        _, thresh1 = cv.threshold(gray1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, thresh2 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        cv.imshow("test", block2)
        cv.imshow("test", thresh1)
        
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.°C'
        min = pytesseract.image_to_string(thresh1, config=custom_config)
        max = pytesseract.image_to_string(thresh2, config=custom_config)

        if (len(max) == 0 or len(min) == 0):
            self.maxTemp = None
            self.minTemp = None
            return
        
        
        self.maxTemp = re.search(r"\d{2}\.\d", max) # do magic
        self.minTemp = re.search(r"\d{2}\.\d", min) # do magic
    
    # To Do
#    def determineClusterTemp
    
    def binarise(self, img : np.ndarray) -> np.ndarray:
        img = cv.threshold(img, self.lowerThresh, self.upperThresh, cv.THRESH_BINARY)[1]
        cv.imshow("binary", img)
        return img
    
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
    
    
    
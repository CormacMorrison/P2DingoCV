import cv2 as cv
import numpy as np
from typing import Final

class ImageProcessor:
    def __init__(self) -> None:
        pass

    def preProcess(self, frame: np.ndarray) -> np.ndarray:
        kernelSize: int = 5
        sigma: int = 5
        return cv.GaussianBlur(frame, (kernelSize, kernelSize), sigma)

    def edgeDetection(self, frame: np.ndarray) -> np.ndarray:
        cannyThreshold: int = 100
        cannyThreshold2: int = 200
        return cv.Canny(frame, cannyThreshold, cannyThreshold2)
    
    # def lineDetection(self, frame: np.ndarray) -> np.ndarray:
        
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process the frame by applying preprocessing then edge detection.
        """
        blurred = self.preProcess(frame)
        edges = self.edgeDetection(blurred)
        return edges
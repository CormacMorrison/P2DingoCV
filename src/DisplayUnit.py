import cv2 as cv
import numpy as np
from typing import Final
from ImageProcessor.ImageProcecssor import ImageProcessor
from Camera.Camera import Camera

class DisplayUnit:
    def __init__(self, camera: Camera, processor: ImageProcessor) -> None:
        self.camera = camera
        self.processor = processor

    def run(self) -> None:
        while True:
            success, frame = self.camera.read_frame()
            if not success or frame is None:
                print("Failed to grab frame.")
                break

            # Use the combined processing function
            edges = self.processor.process(frame)

            # Show original and processed side by side
            cv.imshow("Processed Stream", edges)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
     
    def mock_run(self) -> None:
        while True:
            success, frame = self.camera.mock_read_frame()
            if not success or frame is None:
                print("Failed to grab frame.")
                break

            # Use the combined processing function
            edges = self.processor.process(frame)

            # Show original and processed side by side
            cv.imshow("Processed Stream", edges)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        
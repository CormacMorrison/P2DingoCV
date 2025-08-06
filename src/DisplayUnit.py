import cv2 as cv
import numpy as np
from typing import Final, Tuple
from ImageProcessor.ImageProcessor import ImageProcessor
from Camera.Camera import Camera

class DisplayUnit:
    def __init__(self, camera: Camera, processor: ImageProcessor) -> None:
        self.camera = camera
        self.processor = processor

    def run(self) -> None:
        while True:
            frame = self.camera.read()
            if frame is None:
                print("Failed to grab frame.")
                break

            # Use the combined processing function
            output: Tuple[float, float] = self.processor.process(frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
    
    def runVisuals(self) -> None:
        while True:
            frame = self.camera.read()
            if frame is None:
                print("Failed to grab frame.")
                break

            # Use the combined processing function
            edges = self.processor.processFrame(frame)

            # Show original and processed side by side
            cv.imshow("Processed Stream", edges)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()

     
    def mock_visuals(self) -> None:
        while True:
            frame = self.camera.testRead()
            if frame is None:
                print("Failed to grab frame.")
                break

            # Use the combined processing function
            edges = self.processor.processFrame(frame)

            # Show original and processed side by side
            cv.imshow("Processed Stream", edges)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
    
    def mockRun(self):
        frameCount = 0
        while True:
            frame = self.camera.testRead()
            if frame is None:
                print("Failed to grab frame.")
                break
                
            output: Tuple[float, float] = self.processor.process(frame)

            print(output)
            frameCount+=1
            print(frameCount)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
    

        
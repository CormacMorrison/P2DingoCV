from ..HotspotDetector import HotspotDetector
import cv2 as cv
from datetime import datetime
import json
import os

from ...Types.Types import *

class MinimalDetector(HotspotDetector):
    """Hotspot detector that outputs minimal information per frame.

    Processes each frame from the camera, detects whether any hotspot exists,
    and saves only the minimal JSON results (frame number and detection flag).

    Inherits from HotspotDetector and overrides the `execute` method.
    
    Example usage:
        detector = MinimalDetector(camera, outputPath="output_dir", config="config.json")
        detector.execute()
    """
    def execute(self):
        """Run minimal hotspot detection on all frames from the camera.

        For each frame:
        1. Reads the frame from the camera.
        2. Resets frame-specific data.
        3. Performs per-frame hotspot detection (no component metrics calculated).
        4. Stores a minimal entry containing only the frame number and detection flag.
        5. Continues until no frames remain or user presses 'q'.

        Output:
            - JSON file 'hotspotOutput.json' in the specified outputPath, containing:
                - frame: Frame index
                - detection: Boolean indicating whether a hotspot was detected

        Example:
            detector = MinimalDetector(camera, outputPath="output_dir", config="config.json")
            detector.execute()
        """
        results: list = []
        self.frameCount = 0
        while True:
            self.resetFrameData()
            frame: Frame | None = self.cam.read()
            if frame is None:
                 break  # Stop if no frame was captured
                
            detection: bool = True
            _, detection = self.perFrameProcessing(frame, False, False)
            

            frameEntry = {
                "frame": self.frameCount,
                "detection": detection,
            }

            results.append(frameEntry)
            self.frameCount+=1

            if cv.waitKey(1) & 0xFF == ord('q'):
               break
           
        os.makedirs(self.outputPath, exist_ok=True) 
        with open(f"{self.outputPath}/hotspotOutput.json", "w") as f:
            json.dump(results, f, indent=4) 
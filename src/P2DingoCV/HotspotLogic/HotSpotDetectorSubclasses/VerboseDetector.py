from ..HotspotDetector import HotspotDetector
import cv2 as cv
from datetime import datetime
import json
import os

from ...Types.Types import *

class VerboseDetector(HotspotDetector):
    """Hotspot detector that outputs verbose component-level information per frame.

    Processes each frame from the camera, detects hotspots, computes detailed
    metrics for each detected component, and saves JSON results. Visual outputs
    are not generated (unlike MaximumDetector).

    Inherits from HotspotDetector and overrides the `execute` method.
    
    Example usage:
        detector = VerboseDetector(camera, outputPath="output_dir", config="config.json")
        detector.execute()
    """
    def execute(self):
        """Run verbose hotspot detection on all frames from the camera.

        For each frame:

        1. Reads the frame from the camera.
        2. Resets frame-specific data.
        3. Performs per-frame hotspot detection and calculates metrics per component.
        4. Stores a dictionary for each component with the following metrics:

        - lbl: Component label/index
        - hotspotScore: Overall hotspot score
        - componentTemp: Component temperature
        - centroid: Centroid coordinates of the component
        - deltaPScore: Delta P score
        - deltaPRobust: Robust delta P score
        - zScore: Z-score
        - zScoreNorm: Normalized Z-score
        - compactness: Shape compactness
        - aspectRatioNorm: Normalized aspect ratio
        - eccentricity: Component eccentricity
        - area: Component area

        5. Appends frame results to the overall results list.
        6. Continues until no frames remain or the user presses 'q'.

        Output:

        - JSON file 'hotspotOutput.json' in the specified outputPath.
        - Each frame contains detection flag and detailed component metrics.

        Example:

        ```python
        detector = VerboseDetector(camera, outputPath="output_dir", config="config.json")
        detector.execute()
        """
        results: list = []
        self.frameCount = 0
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
                    "area",
                ]
        while True:
            self.resetFrameData()
            frame: Frame | None = self.cam.read()
            if frame is None:
                 break  # Stop if no frame was captured
                
            detection: bool = True
            components, detection = self.perFrameProcessing(frame, False, False)

            frameEntry = {
                "frame": self.frameCount,
                "detection": detection,
                "components": {}
            }

            for i, comp in enumerate(components):
                compDict = dict(zip(headers, comp))
                frameEntry["components"][i] = compDict
                
            results.append(frameEntry)
            self.frameCount+=1

            if cv.waitKey(1) & 0xFF == ord('q'):
               break
           
        os.makedirs(self.outputPath, exist_ok=True) 
        with open(f"{self.outputPath}/hotspotOutput.json", "w") as f:
            json.dump(results, f, indent=4) 
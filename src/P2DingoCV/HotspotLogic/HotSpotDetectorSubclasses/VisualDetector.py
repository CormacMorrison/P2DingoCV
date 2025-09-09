from ..HotspotDetector import HotspotDetector
import cv2 as cv
from datetime import datetime
import json
import os

from ...Types.Types import *

class VisualDetector(HotspotDetector):
    """Hotspot detector that outputs visual frames with component-level metrics.

    Processes each frame from the camera, detects hotspots, computes metrics
    for each component, and saves both visual outputs and JSON results. 
    Visual output is enabled, but verbose component logging may be limited
    compared to MaximumDetector.

    Inherits from HotspotDetector and overrides the `execute` method.
    
    Example usage:
        detector = VisualDetector(camera, outputPath="output_dir", config="config.json")
        detector.execute()
    """
    def execute(self):
        """Run hotspot detection and generate visual outputs for all frames.

        For each frame:
        1. Reads the frame from the camera.
        2. Resets frame-specific data.
        3. Performs per-frame hotspot detection and calculates metrics per component.
        4. Stores a dictionary per component with the following metrics:
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
        5. Saves visual outputs to a timestamped folder for each frame.
        6. Appends frame results to the overall results list.
        7. Continues until no frames remain or the user presses 'q'.

        Output:
            - JSON file 'hotspotOutput.json' in the specified outputPath
            - Visual outputs per frame in timestamped subdirectories

        Example:
            detector = VisualDetector(camera, outputPath="output_dir", config="config.json")
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
            self.pathToVisuals = f'{self.outputPath}/frame_{self.frameCount} + {datetime.now().strftime("%y:%m:%d:%H:%S")}'
            components, detection = self.perFrameProcessing(frame, True, True)

            frameEntry = {
                "frame": self.frameCount,
                "detection": detection,
                "pathToVisuals": self.pathToVisuals,
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
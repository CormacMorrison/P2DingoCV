from ..HotspotDetector import HotspotDetector
import cv2 as cv
from datetime import datetime
import json
import os

from ...Types.Types import *

class MaximumDetector(HotspotDetector):
    """Hotspot detector that outputs maximum information per frame.

    Processes each frame from the camera, detects hotspots, computes
    detailed metrics for each component, and saves both JSON results and
    visual outputs for every frame.

    Inherits from HotspotDetector and overrides the execute method.
    """
    def execute(self):
        """Run maximum-detail hotspot detection on all frames from the camera.

        For each frame:
        1. Reads the frame from the camera.
        2. Resets frame-specific data.
        3. Processes the frame to detect components and calculate metrics.
        4. Stores detailed metrics per component in a JSON-compatible dictionary.
        5. Saves visual outputs to a timestamped folder per frame.
        6. Appends frame results to the overall results list.
        7. Continues until no frames remain or user presses 'q'.

        Metrics computed for each component include:
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

        Output:
            - JSON file 'hotspotOutput.json' in the specified outputPath
            - Visual outputs saved in timestamped subdirectories for each frame

        Example:
            detector = MaximumDetector(camera, "output_dir", "config.json")
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
                "components": {}
            }

            for i, comp in enumerate(components):
                compDict = dict(zip(headers, comp))
                frameEntry["components"][i] = compDict
                
            self.processingDiagonstics()

            results.append(frameEntry)
            self.frameCount+=1

            if cv.waitKey(1) & 0xFF == ord('q'):
               break
           
        os.makedirs(self.outputPath, exist_ok=True) 
           
        with open(f"{self.outputPath}/hotspotOutput.json", "w") as f:
            json.dump(results, f, indent=4) 
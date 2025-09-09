from HotspotDetector import HotspotDetector
import cv2 as cv
from datetime import datetime
import json

from HotspotLogic.Types.Types import *

class VisualDetector(HotspotDetector):
    def execute(self):
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
            self.pathToVisuals = f'{self.exitPath}/frame_{self.frameCount} + {datetime.now().strftime("%y:%m:%d:%H:%S")}'
            components, detection = self.perFrameProcessing(frame)

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
           
        with open(f"{self.outputPath}/hotspotOutput.json", "w") as f:
            json.dump(results, f, indent=4) 
from HotspotDetector import HotspotDetector
import cv2 as cv
from datetime import datetime
import json

from HotspotLogic.Types.Types import *

class MinimalDetector(HotspotDetector):
    def execute(self):
        results: list = []
        self.frameCount = 0
        while True:
            self.resetFrameData()
            frame: Frame | None = self.cam.read()
            if frame is None:
                 break  # Stop if no frame was captured
                
            detection: bool = True
            _, detection = self.perFrameProcessing(frame)
            

            frameEntry = {
                "frame": self.frameCount,
                "detection": detection,
            }

            results.append(frameEntry)
            self.frameCount+=1

            if cv.waitKey(1) & 0xFF == ord('q'):
               break
           
        with open(f"{self.outputPath}/hotspotOutput.json", "w") as f:
            json.dump(results, f, indent=4) 
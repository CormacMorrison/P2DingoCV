from abc import ABC, abstractmethod
from P2DingoCV.App.App import App
from ..Types.Types import *
import cv2 as cv
from datetime import datetime
import json
import os
from typing import List, Dict


class Minimal(App):

    def execute(self) -> None:
        """
        Main execution loop for the Minimal application.

        This method processes video frames from the camera (`self.cam`) and performs 
        hotspot detection on the whole frame and, if panel segmentation is enabled, on 
        individual panels. Unlike the Maximal version, this Minimal version simplifies 
        panel processing and does not store detailed component data.

        Workflow per frame:
        1. Reset hotspot detector frame data.
        2. Capture a frame from the camera.
        3. Create output directories for the frame, including logs.
        4. Attempt panel segmentation (optional):
        - If successful, each detected panel is processed individually.
        - If segmentation fails, the frame is still processed as a whole.
        5. Run hotspot detection on the entire frame.
        6. If panels were detected, run hotspot detection on each panel.
        7. Store results in a structured dictionary including:
        - Frame number
        - Detection success flag
        - Detected panels (with detection success)
        8. Append per-frame results to the main results list.
        9. Continue until no more frames are available or 'q' is pressed.

        Outputs:
        - `self.outputPath/hotspotOutput.json`: JSON file containing detection results for all frames.
        - Visual outputs are saved to directories created per frame and per panel, 
        though no per-component data is stored.

        Notes:
        - Uses `self.PanelSegmentor` for optional panel segmentation.
        - Uses `self.HotspotDetector` for per-frame and per-panel hotspot detection.
        - Logs errors during panel segmentation and continues processing subsequent frames.
        - Stops processing when no frames are returned by the camera or the user presses 'q'.
        """
        results: List[Dict] = []
        self.frameCount: int = 0

        os.makedirs(self.outputPath, exist_ok=True)

        while True:
            self.HotspotDetector.resetFrameData()
            frame: Frame | None = self.cam.read()

            if frame is None:
                break  # Stop if no frame was captured

            frameOutputPath: str = f'{self.outputPath}/frame_{self.frameCount}'
            self.HotspotDetector.updateOutputPath(frameOutputPath)
            logPath: str = f'{frameOutputPath}/logs'
            os.makedirs(logPath, exist_ok=True)

            panelDetected: bool = False
            try:
                panels: List[Frame] = self.PanelSegmentor.execute(frame, self.frameCount, logPath, False, "", False)
                panelDetected = True
                frameEntry: dict = {
                    "frame": self.frameCount,
                    "detection": True,
                    "panels": {},
                }
            except Exception as e:
                self.logger.error(f"Error in panel segmentation: {e}")
                panels = []
                frameEntry: Dict = {
                    "frame": self.frameCount,
                    "detection": True,
                    "panels": None,
                }

            detectionMain: bool = False
            _, detectionMain = self.HotspotDetector.perFrameProcessing(frame, False, False, self.frameCount)

            if panelDetected:
                for pIdx, panel in enumerate(panels):
                    panelEntry: Dict = {
                        "panel": pIdx,
                        "detection": True,
                    }

                    detectionPanel: bool = False

                    self.HotspotDetector.resetFrameData()
                    self.HotspotDetector.updateOutputPath(f'{frameOutputPath}/panel_{pIdx}')
                    _, detectionPanel = self.HotspotDetector.perFrameProcessing(panel, False, False, pIdx)

                    panelEntry["detection"] = detectionPanel

                    if detectionPanel:
                        detectionMain = True
                    frameEntry["panels"][f"panel_{pIdx}"] = panelEntry

            frameEntry["detection"] = detectionMain
            results.append(frameEntry)
            self.frameCount += 1

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        with open(f"{self.outputPath}/hotspotOutput.json", "w") as f:
            json.dump(results, f, indent=4)

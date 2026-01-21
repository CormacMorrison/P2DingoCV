from abc import ABC, abstractmethod
from turtle import st
from P2DingoCV.App.App import App
from ..Types.Types import *
import cv2 as cv
from datetime import datetime
import json
import os
from typing import List, Dict


class Verbose(App):

    def execute(self) -> None:
        
        """
        Main execution loop for the Maximal application.

        This method processes video frames from the camera (`self.cam`) and performs 
        hotspot detection on both the whole frame and individual panels. It stores 
        detailed per-component metrics, panel detection results, and creates structured 
        output folders with visualizations.

        Workflow per frame:
        1. Reset hotspot detector frame data.
        2. Capture a frame from the camera.
        3. Create output directories for the frame, including logs.
        4. Attempt panel segmentation:
        - If successful, each detected panel is processed individually.
        - If segmentation fails, only whole-frame hotspot detection is performed.
        5. Run hotspot detection on the whole frame and store component metrics.
        6. If panels were detected, run hotspot detection on each panel.
        7. Store results in a structured dictionary including:
        - Frame number
        - Detection success flag
        - Panel detection results and per-component metrics
        - Whole-frame component metrics
        8. Append per-frame results to the main results list.
        9. Continue until no more frames are available or the user presses 'q'.

        Outputs:
        - `self.outputPath/hotspotOutput.json`: JSON file containing detection results for all frames.
        - Visual outputs are saved in a structured folder hierarchy including frames, panels, diagnostics, and plots.

        Notes:
        - Uses `self.PanelSegmentor` for panel segmentation.
        - Uses `self.HotspotDetector` for per-frame and per-panel hotspot detection.
        - Logs errors during panel segmentation and continues processing subsequent frames.
        - Stops processing when no frames are returned by the camera or the user presses 'q'.
        """
        results: List = []
        self.frameCount = 0
        headers: List = [
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

        os.makedirs(self.outputPath, exist_ok=True) 
        while True:
            self.HotspotDetector.resetFrameData()
            frame: Frame | None = self.cam.read()
            
            if frame is None:
                 break  # Stop if no frame was captured
             
            frameOutputPath: str = f'{self.outputPath}/frame_{self.frameCount}'
            self.HotspotDetector.updateOutputPath(frameOutputPath)
            logPath = f'{frameOutputPath}/logs'
            os.makedirs(logPath, exist_ok=True)

            panelDetected: bool = False
            try:
                panels: List[Frame] = self.PanelSegmentor.execute(frame, self.frameCount, logPath, False, "", False)
                panelDetected = True
                frameEntry = {
                    "frame": self.frameCount,
                    "detection": True,
                    "panels": {},
                    "wholeFrame": {}
                }
            except Exception as e:
                self.logger.error(f"Error in panel segmentation: {e}")
                panels = []
                frameEntry = {
                    "frame": self.frameCount,
                    "detection": True,
                    "panels": None,
                    "wholeFrame": {}
                }
            
            detectionMain: bool = False
            components: List = []
            components, detectionMain = self.HotspotDetector.perFrameProcessing(frame, False, False, self.frameCount)
            
            for i, comp in enumerate(components):
                compDict: Dict = dict(zip(headers, comp))
                frameEntry["wholeFrame"][i] = compDict
                
            if panelDetected:
                for pIdx, panel in enumerate(panels):
                    panelEntry = {
                        "panel": pIdx,
                        "detection": True,
                        "components": {}
                    }
                    
                    detectionPanel: bool = False
                   
                    self.HotspotDetector.resetFrameData()
                    self.HotspotDetector.updateOutputPath(f'{frameOutputPath}/panel_{pIdx}')
                    panelComponents, detectionPanel = self.HotspotDetector.perFrameProcessing(panel, True, True, pIdx)
                    
                    panelEntry["detection"] = detectionPanel
                    
                    if (detectionPanel):
                        detectionMain = True
                    frameEntry["panels"][f"panel_{pIdx}"] = panelEntry
                    for j, pComp in enumerate(panelComponents):
                        pCompDict: Dict = dict(zip(headers, pComp))
                        frameEntry["panels"][f"panel_{pIdx}"]["components"][j] = pCompDict

            frameEntry["detection"] = detectionMain
            results.append(frameEntry)
            self.frameCount+=1
            
            if cv.waitKey(1) & 0xFF == ord('q'):
               break

        with open(f"{self.outputPath}/hotspotOutput.json", "w") as f:
            json.dump(results, f, indent=4)
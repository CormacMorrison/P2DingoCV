from abc import ABC, abstractmethod
from P2DingoCV.App.App import App
from ..Types.Types import *
import cv2 as cv
from datetime import datetime
import json
import os
from typing import List, Dict

class Maximal(App):

    def execute(self) -> None:
        """
        Main execution loop for the Maximal application.

        This method processes video frames from the camera (`self.cam`) and performs 
        hotspot detection on both whole frames and segmented panels. The workflow is as follows:

        Workflow per frame:
        1. Reset hotspot detector frame data.
        2. Capture a frame from the camera.
        3. Create output directories for the frame, including logs.
        4. Attempt panel segmentation:
        - If successful, each detected panel is processed individually.
        - If segmentation fails, the frame is still processed as a whole.
        5. Run hotspot detection on the entire frame and store component metrics.
        6. If panels were detected, run hotspot detection on each panel individually.
        7. Store results in a structured dictionary including:
        - Frame number
        - Detection success flag
        - Paths to visual outputs
        - Detected panels and their components
        - Detected components in the whole frame
        8. Append per-frame results to the main results list.
        9. Continue until no more frames are available or 'q' is pressed.

        Outputs:
        - `self.outputPath/hotspotOutput.json`: JSON file containing detection results for all frames.
        - Visual outputs for each frame and panel in a structured folder hierarchy:
            output/resultsFolder/
            ├── hotspotOutput.json
            ├── frames/
            │   ├── frame_0/
            │   │   ├── panels/
            │   │   │   ├── panel_0/
            │   │   │   │   └── hotspots/
            │   │   │   │       └── panel_0_hotspot.png
            │   │   │   └── panel_1/
            │   │   │       └── hotspots/
            │   │   │           └── panel_1_hotspot.png
            │   │   ├── diagnostics/
            │   │   │   └── frame_0_diagnostic_hotspot.png
            │   │   ├── hotspots/
            │   │   │   └── frame_0_hotspot.png
            │   │   ├── logs/
            │   │   │   └── frame_0.log
            │   │   └── segmentation/
            │   │       ├── frame_0_segmentation.png
            │   │       └── cells/
            │   │           ├── cell_0.png
            │   │           ├── cell_1.png
            │   │           └── ...
            │   │
            │   └── frame_1/
            │       ├── panels/
            │       │   └── panel_0/
            │       │       └── hotspots/
            │       │           └── panel_0_hotspot.png
            │       ├── diagnostics/
            │       ├── hotspots/
            │       ├── logs/
            │       └── segmentation/
            │           ├── frame_1_segmentation.png
            │           └── cells/
            │               ├── cell_0.png
            │               └── ...
            │
            ├── plots/
            │   └── plot1.png
            └── global_diagnostics/
                ├── frame_diagnostic1_hotspot.png
                └── frame_diagnostic2_hotspot.png

        Notes:
        - Uses `self.PanelSegmentor` for panel segmentation.
        - Uses `self.HotspotDetector` for per-frame and per-panel hotspot detection.
        - Logs errors during panel segmentation and continues processing subsequent frames.
        - Stops processing when no frames are returned by the camera or the user presses 'q'.
        
        """

        results: List = []
        self.frameCount: int = 0
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
            logPath: str = f'{frameOutputPath}/logs'
            os.makedirs(logPath, exist_ok=True)

            panelDetected: bool = False
            try:
                panels: List[Frame] = self.PanelSegmentor.execute(frame, self.frameCount, logPath, True, frameOutputPath, True)
                panelDetected = True
                frameEntry: Dict = {
                    "frame": self.frameCount,
                    "detection": True,
                    "pathToVisuals": frameOutputPath,
                    "panels": {},
                    "wholeFrame": {}
                }
            except Exception as e:
                self.logger.error(f"Error in panel segmentation: {e}")
                panels = []
                frameEntry: Dict = {
                    "frame": self.frameCount,
                    "detection": True,
                    "pathToVisuals": frameOutputPath,
                    "panels": None,
                    "wholeFrame": {}
                }
            
            components: List = []
            detectionMain: bool = False
            components, detectionMain = self.HotspotDetector.perFrameProcessing(frame, True, True, self.frameCount)
            
            for i, comp in enumerate(components):
                compDict = dict(zip(headers, comp))
                frameEntry["wholeFrame"][i] = compDict
                
            if panelDetected:
                for pIdx, panel in enumerate(panels):
                    panelEntry: Dict = {
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
                        pCompDict = dict(zip(headers, pComp))
                        frameEntry["panels"][f"panel_{pIdx}"]["components"][j] = pCompDict

            frameEntry["detection"] = detectionMain
            results.append(frameEntry)
            self.frameCount+=1
            
            if cv.waitKey(1) & 0xFF == ord('q'):
               break

        with open(f"{self.outputPath}/hotspotOutput.json", "w") as f:
            json.dump(results, f, indent=4)
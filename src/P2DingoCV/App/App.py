from abc import ABC, abstractmethod
import logging
from P2DingoCV.Util.MiscUtil import MiscUtil
from ..HotspotLogic.HotspotDetector import HotspotDetector
from ..PanelSegmentation.PanelSegmentor import PanelSegmentor
from P2DingoCV.Camera.Camera import Camera
from P2DingoCV.Util.VisualUtil import VisualUtils
from datetime import datetime
from ..Types.Types import *


class App(ABC):
    def __init__(self, cam: Camera, exitPath: str, config: str | None = None) -> None:
        self.cam: Camera = cam
        self.outputPath = exitPath + f'/{datetime.now().strftime("%y|%m|%d|%H:%M:%S")}'
        self.utility = VisualUtils(self.outputPath)
        self.logger: logging.Logger = MiscUtil.setupLogger("AppLogger", self.outputPath + "/logs")
        
        self.HotspotDetector: HotspotDetector = HotspotDetector(self.outputPath, config)
        self.PanelSegmentor: PanelSegmentor = PanelSegmentor(self.cam._HEIGHT, self.cam._WIDTH, self.outputPath, config) 
        
    
    @abstractmethod
    def execute(self) -> None:
        pass


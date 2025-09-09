from abc import ABC, abstractmethod
import numpy as np
from typing import Final
from ..Types.Types import *

class Camera(ABC):
    """
    Abstract base class for camera input sources.

    This class defines the interface for any camera or video input
    source used in image or thermal processing pipelines. Subclasses
    must implement the frame capture, test capture, and resource release methods.

    Attributes:
        _HEIGHT (Final[int]): Default frame height in pixels.
        _WIDTH (Final[int]): Default frame width in pixels.
    """    """
    Abstract base class for camera input sources.
    """
    _HEIGHT: Final[int] = 600
    _WIDTH: Final[int] = 800
    @abstractmethod
    def read(self) -> Frame | None:
        """
        Capture and return the next frame from the camera.

        Returns:
            np.ndarray | None: A NumPy array representing the image frame
            (shape: [_HEIGHT, _WIDTH, channels]) or None if no frame is available.
        """        """
        Read and return the next frame as a NumPy array.
        """
        pass
    
    @abstractmethod 
    def testRead(self) -> Frame | None:
        """
        Capture the next frame in a loop for testing purposes.

        Unlike `read`, this method loops or replays frames when the input
        source reaches the end. Useful for continuous testing without
        stopping at the end of a video or camera stream.

        Returns:
            np.ndarray | None: A NumPy array representing the image frame
            or None if the source is unavailable.
        """        """
        Read but loops when done for testing.
        """
        
        pass
    
    @abstractmethod
    def release(self) -> None:
        """
        Release any resources held by the camera.

        This should close video streams, free memory, and release
        hardware resources as needed.
        """        """
        Release any resources held by the camera.
        """
        pass
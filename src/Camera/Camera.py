from abc import ABC, abstractmethod
import numpy as np
from typing import Final

class Camera(ABC):
    """
    Abstract base class for camera input sources.
    """
    _HEIGHT: Final[int] = 600
    _WIDTH: Final[int] = 800
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """
        Read and return the next frame as a NumPy array.
        """
        pass
    
    @abstractmethod 
    def testRead(self) -> np.ndarray | None:
        """
        Read but loops when done for testing.
        """
        
        pass
    
    @abstractmethod
    def release(self) -> None:
        """
        Release any resources held by the camera.
        """
        pass
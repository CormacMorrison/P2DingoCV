import cv2 as cv
import numpy as np
from .Camera import Camera
from typing import Final

from ..Types.Types import *

class VideoCamera(Camera):
    """
    Camera subclass that reads frames from a webcam, camera device, or video file.

    Attributes:
        cap (cv.VideoCapture): OpenCV video capture object.
        frame (np.ndarray | None): Last frame read from the camera.
        source (int | str): Device index or video file path.
    """    """
    Camera subclass that reads frames from a webcam, camera device, or video file.
    """
    _DESIREDFPS: Final[int] = 24

    def __init__(self, source: int | str = 0) -> None:
        """
        Initialize the VideoCamera with a device index or video file path.

        Args:
            source (int | str): Camera device index or path to a video file.

        Raises:
            ValueError: If the video source cannot be opened.
        """
        self.cap = cv.VideoCapture(source)
        self.frame: np.ndarray | None = None
        self.source = source

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self._WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._HEIGHT)
        self.cap.set(cv.CAP_PROP_FPS, self._DESIREDFPS)

    def read(self) -> Frame | None:
        """
        Read the next frame from the video source.

        Returns:
            np.ndarray | None: The next frame, or None if the video has ended.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame = frame
        return frame

    def testRead(self) -> Frame | None:
        """
        Read frames in a loop for testing. Loops back to the start if the video ends.

        Returns:
            np.ndarray | None: The next frame, or None if no frame is available.
        """
        if self.cap:
            _, frame = self.cap.read()
        else:
            return None
        if frame is None:
            return None

        return frame

    def release(self) -> None:
        """
        Release video capture resources.
        """
        if self.cap:
            self.cap.release()
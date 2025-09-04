import cv2 as cv
import numpy as np
from .Camera import Camera
from typing import Final

class VideoCamera(Camera):
    """
    Camera subclass that reads frames from a webcam, camera device, or video file.
    """
    _DESIREDFPS: Final[int] = 24

    def __init__(self, source: int | str = 0) -> None:
        self.cap = cv.VideoCapture(source)
        self.frame: np.ndarray | None = None
        self.source = source

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self._WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._HEIGHT)
        self.cap.set(cv.CAP_PROP_FPS, self._DESIREDFPS)

    def read(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame = frame
        return frame

    def testRead(self) -> np.ndarray | None:
        """
        Read a single frame from the video source. but loops

        Returns:
            A tuple containing a boolean success flag and the frame (numpy array or None).
        """
        if self.cap:
            _, frame = self.cap.read()
        else:
            return None
        if frame is None:
            return None

        return frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
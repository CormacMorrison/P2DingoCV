from typing import Final
import cv2 as cv
import numpy as np


class Camera:
    """
    Camera class for reading video frames from a camera device or video file using OpenCV.

    Attributes:
        source (int | str): Identifier for the video input source. Can be an integer
            index for a camera device (e.g., 0 for the default webcam) or a string
            representing a file path or stream URL.
        cap (cv2.VideoCapture): OpenCV video capture object that handles the video
            stream or camera input and provides methods for frame retrieval and release.
    """

    _HEIGHT: Final[int] = 1080
    _WIDTH: Final[int] = 1920
    _DESIREDFPS: Final[int] = 60

    def __init__(self, source: int | str = 0) -> None:
        """
        Initialize the camera or video capture object.

        Args:
            source (int | str): Camera device index (e.g., 0 for default webcam),
                                file path to video, or video stream URL.

        Raises:
            ValueError: If the video source cannot be opened.
        """
        self.source: int | str = source
        self.cap: cv.VideoCapture = cv.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera or video source: {source}")

        # Optionally set resolution
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self._WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._HEIGHT)
        self.cap.set(cv.CAP_PROP_FPS, self._DESIREDFPS)

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read a single frame from the video source.

        Returns:
            A tuple containing a boolean success flag and the frame (numpy array or None).
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame
    
    def mock_read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read a single frame from the video source.

        Returns:
            A tuple containing a boolean success flag and the frame (numpy array or None).
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
        # Try to loop only if source is a file (not int)
            if isinstance(self.source, str):
                total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        return False, None
            else:
                return False, None

        return True, frame

    def stream(self) -> None:
        """
        Continuously stream video frames until 'q' is pressed.
        Displays the frames in a window named 'Camera Stream'.
        """
        while True:
            ret, frame = self.read_frame()
            if not ret or frame is None:
                print("Failed to grab frame or stream ended.")
                break

            cv.imshow("Camera Stream", frame)

            # Press 'q' to quit streaming
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        self.release()

    def release(self) -> None:
        """
        Release the video capture object and destroy all OpenCV windows.
        """
        self.cap.release()
        cv.destroyAllWindows()

def mockStream(self) -> None:
    """
    Continuously stream video frames from a video file in a loop until 'q' is pressed.
    Displays the frames in a window named 'Camera Stream'.
    """
    while True:
        ret, frame = self.read_frame()
        if not ret or frame is None:
            print("Looping to start of video.")
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        cv.imshow("Camera Stream", frame)

        # Press 'q' to quit streaming
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    self.release()

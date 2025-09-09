import os
from .Camera import Camera
from .ImageCamera import ImageCamera
from .VideoCamera import VideoCamera

class CameraFactory:
    """
    Factory class to create appropriate Camera instances based on the given input source.

    This class abstracts the logic of selecting the correct Camera subclass
    (VideoCamera or ImageCamera) based on whether the input is a device index,
    a single image file, a video file, or a directory of images.

    Methods:
        create(source: int | str) -> Camera:
            Returns a Camera instance corresponding to the given source.
    """    """
    Factory class to create appropriate Camera instances based on input.
    """

    @staticmethod
    def create(source: int | str) -> Camera:
        """
        Create a Camera instance based on the input source.

        Args:
            source (int | str): The input source. Can be:
                - int: Device index for a webcam or capture device.
                - str: Path to a video file, an image file, or a directory of images.

        Returns:
            Camera: An instance of VideoCamera or ImageCamera.

        Raises:
            ValueError: If the source type or file extension is unsupported.
        """
        if isinstance(source, int):
            # Treat as a device index (e.g., webcam)
            return VideoCamera(source)

        if not isinstance(source, str):
            raise ValueError(f"Unsupported source type: {type(source)}")

        if os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                return VideoCamera(source)
            elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                return ImageCamera(source)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        if os.path.isdir(source):
            return ImageCamera(source)

        raise ValueError(f"Invalid camera source: {source}")
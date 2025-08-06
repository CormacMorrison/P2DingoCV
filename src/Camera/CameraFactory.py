import os
from .Camera import Camera
from .ImageCamera import ImageCamera
from .VideoCamera import VideoCamera

class CameraFactory:
    """
    Factory class to create appropriate Camera instances based on input.
    """

    @staticmethod
    def create(source: int | str) -> Camera:
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
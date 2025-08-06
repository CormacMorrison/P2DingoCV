import os
import cv2 as cv
import numpy as np
from .Camera import Camera

class ImageCamera(Camera):
    """
    Camera subclass that reads either a single image or a sequence of images from a folder.
    """

    def __init__(self, path: str) -> None:
        self.imagePaths: list[str]
        self.index: int = 0
        self.frame: np.ndarray 

        if os.path.isdir(path):
            # It's a directory: load all image files
            self.imagePaths = sorted([
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            ])

            if not self.imagePaths:
                raise ValueError(f"No images found in folder: {path}")

        elif os.path.isfile(path):
            # It's a file: verify it's a valid image extension
            if not path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                raise ValueError(f"Unsupported image format: {path}")
            self.imagePaths = [path]

        else:
            raise ValueError(f"Invalid path: {path}")

    def read(self) -> np.ndarray | None:
        # Image paths must exist at this point
        if self.index >= len(self.imagePaths):
            return None
        
        imagePath = self.imagePaths[self.index]
        image = cv.imread(imagePath)
        image = cv.resize(image, (self._WIDTH, self._HEIGHT), interpolation=cv.INTER_LINEAR)
        self.frame = image
        self.index += 1

        if image is None:
            raise IOError(f"Failed to read image: {imagePath}")

        return image
    
    def testRead(self) -> np.ndarray | None:
        if self.index >= len(self.imagePaths):
            self.index = 0

        imagePath = self.imagePaths[self.index]
        image = cv.imread(imagePath)
        self.frame = image
        self.index += 1

        if image is None:
            raise IOError(f"Failed to read image: {imagePath}")

        return image


    def release(self) -> None:
        # No persistent resources to release
        pass

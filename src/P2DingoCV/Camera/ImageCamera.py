import os
import cv2 as cv
import numpy as np
from .Camera import Camera

from ..Types.Types import *

class ImageCamera(Camera):
    """
    Camera subclass that reads either a single image or a sequence of images from a folder.

    Attributes:
        imagePaths (list[str]): Ordered list of image file paths to read.
        index (int): Current index in the imagePaths list.
        frame (np.ndarray): Last frame read from the camera.
    """    """
    Camera subclass that reads either a single image or a sequence of images from a folder.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the ImageCamera with a file or directory path.

        Args:
            path (str): Path to an image file or a directory containing images.

        Raises:
            ValueError: If the path is invalid or contains no supported images.
        """
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
        """
        Read the next image in the sequence.

        Returns:
            np.ndarray | None: The next image as a NumPy array, or None if at the end.

        Raises:
            IOError: If the image cannot be read.
        """
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
    
    def testRead(self) -> Frame | None:
        """
        Read images in a loop for testing purposes. Resets to the first image when the end is reached.

        Returns:
            np.ndarray | None: The next image as a NumPy array.

        Raises:
            IOError: If the image cannot be read.
        """
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
        """
        Release resources associated with the camera.

        No persistent resources to release for ImageCamera.
        """
        pass

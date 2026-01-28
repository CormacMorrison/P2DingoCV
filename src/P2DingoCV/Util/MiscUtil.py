import sys
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import logging
from pathlib import Path
import json
import uuid

class MiscUtil:
    pass
    @staticmethod
    def iqrOutliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        med = np.median(data)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = data[(data < lower) | (data > upper)]
        return outliers, (lower, upper)

    @staticmethod
    def angleDiff(a, b) -> float:
        d = abs(a - b)
        return min(d, 180 - d)
    
    @staticmethod
    def safeInt(x, tol=1e-6) -> int:
        if abs(x - round(x)) < tol:
            return int(round(x))
        else:
            raise ValueError("Detected grid spacings are not integers.")

    @staticmethod
    def getMode(arr: np.ndarray) -> NDArray[np.integer]:
        arr = np.asarray(arr, dtype=float)  # force numeric
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]
    
    @staticmethod
    def setupLogger(name: str, logDir: str | None) -> logging.Logger:
        logger = logging.getLogger(name)
        if logger.handlers:
            # Logger already exists; return it
            return logger

        logger.setLevel(logging.INFO)
        logger.propagate = False  # prevent logs going to root logger

        if logDir:
            log_path = Path(logDir).expanduser().resolve()
            log_path.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path / f"{name}.log")
        else:
            handler = logging.StreamHandler()  # fallback

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    
    @staticmethod
    def loadConfig(path: str | Path) -> dict:
        path = Path(path)
        with path.open("r") as f:
            return json.load(f)
        
    @staticmethod
    def boxesOverlap(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
        """
        Check if two rectangular bounding boxes overlap.

        Parameters
        ----------
        b1 : tuple[int, int, int, int]
            Bounding box 1 specified as (x, y, width, height).
        b2 : tuple[int, int, int, int]
            Bounding box 2 specified as (x, y, width, height).

        Returns
        -------
        bool
            True if the bounding boxes overlap (intersect), False otherwise.

        Notes
        -----
        - Overlap is determined based on the intersection of the boxes' areas.
        - Boxes that touch edges but do not share any interior area are considered non-overlapping.
        """
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)

        return (xb > xa) and (yb > ya)
    
    @staticmethod
    def maskContains(mask_a, mask_b):
        """
        Check if one binary mask is completely contained within another.

        This function returns True if every non-zero pixel in `mask_b` falls
        within a non-zero region of `mask_a`. In other words, `mask_b` is
        entirely "inside" `mask_a`.

        Parameters
        ----------
        mask_a : np.ndarray
            Binary mask (0 or 255) representing the "container" region.
        mask_b : np.ndarray
            Binary mask (0 or 255) to check for containment within `mask_a`.

        Returns
        -------
        bool
            True if all non-zero pixels of `mask_b` are also non-zero in `mask_a`,
            False otherwise.

        Notes
        -----
        - Both masks must have the same shape.
        - This function works for binary masks where foreground pixels are non-zero
            (typically 255) and background is zero.
        """
        return np.all((mask_a == 0) | (mask_b == 0))
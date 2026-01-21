import sys
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

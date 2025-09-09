import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class VisualUtils:
    """Utility class for visualizing and saving images, histograms, and tables.

    Provides methods to plot image histograms, save frames, plot frequency arrays, 
    and pretty-print tables. All outputs are saved to the specified directory.
    """
    def __init__(self, pathToVisuals: str = "visuals"):
        """Initialize the VisualUtils object and create output directory.

        Args:
            pathToVisuals (str): Path to directory where visuals will be saved. Defaults to "visuals".
        """
        self.outputPath = pathToVisuals
        os.makedirs(self.outputPath, exist_ok=True)
    

    def plotHistogram(self, frame: np.ndarray) -> np.ndarray | None:
        """Plot and save the histogram of an image.

        Supports grayscale and BGR images. Saves the histogram as a PNG in the output directory.

        Args:
            frame (np.ndarray): Image array (grayscale or BGR).

        Returns:
            np.ndarray | None: The input frame, also displayed using OpenCV.
        
        Raises:
            AssertionError: If frame is None.
            ValueError: If the image format is unsupported.
        """
        assert frame is not None, "file could not be read, check with os.path.exists()"

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            # Grayscale
            print("Detected grayscale image")
            gray = frame if len(frame.shape) == 2 else frame[:, :, 0]
            gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])

            plt.figure(figsize=(8, 4))
            plt.title("Grayscale Histogram")
            plt.plot(gray_hist, color="k")
            plt.xlim([0, 256])
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            filepath = os.path.join(self.outputPath, "histogram_gray.png")
            plt.savefig(filepath)
            plt.close()

        elif frame.shape[2] == 3:
            # BGR
            print("Detected BGR image")
            colors = ("b", "g", "r")
            plt.figure(figsize=(8, 4))
            plt.title("BGR Histogram")
            for i, col in enumerate(colors):
                hist = cv.calcHist([frame], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            filepath = os.path.join(self.outputPath, "histogram_bgr.png")
            plt.savefig(filepath)
            plt.close()
        else:
            raise ValueError("Unsupported image format")

        cv.imshow("frame", frame)
        return frame

    def plotFreqArray(self, arr: np.ndarray, title: str) -> None:
        """Plot a 1D array as a bar chart and save it as a PNG.

        Args:
            arr (np.ndarray): 1D array of values to plot.
            title (str): Title of the plot, also used as filename.
        """
        plots_dir = os.path.join(self.outputPath, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure()
        plt.bar(range(len(arr)), arr)
        plt.xlabel("Index")
        plt.ylabel("Value")

        filepath = os.path.join(plots_dir, f"{title}.png")
        plt.savefig(filepath)
        plt.close()

    def saveFrame(self, frame: np.ndarray, tag: str, frameCount: int, folder: str = "frames") -> None:
        """Save an image frame with a tag and frame count.

        Args:
            frame (np.ndarray): Image frame to save.
            tag (str): Descriptive tag for the frame.
            frameCount (int): Frame number to include in the filename.
            folder (str): Subfolder under the output path to save frames. Defaults to "frames".
        """
        folder_path = os.path.join(self.outputPath, folder)
        os.makedirs(folder_path, exist_ok=True)

        filename = os.path.join(
            folder_path, f"frames/frame_{frameCount}_{tag}.png"
        )
        cv.imwrite(filename, frame)
        print(f"Frame saved to {filename}")

    @staticmethod
    def printTable(data, headers=None, precision=2) -> None:
        """Pretty-print a list of lists (or tuples) as a table with borders.

        Numeric values are formatted to the specified precision. Nested tuples/lists are also formatted.

        Args:
            data (list[list|tuple]): 2D list or list of tuples containing table data.
            headers (list[str] | None): Optional list of column headers.
            precision (int): Number of decimal places for numeric values. Defaults to 2.
        """
        str_data = []
        col_widths = []

        for row in data:
            str_row = []
            for val in row:
                if isinstance(val, (tuple, list)):
                    val_str = "(" + ", ".join(
                        f"{v:.{precision}f}" if isinstance(v, (int, float)) else str(v)
                        for v in val
                    ) + ")"
                elif isinstance(val, (int, float)):
                    val_str = f"{val:.{precision}f}"
                else:
                    val_str = str(val)
                str_row.append(val_str)
            str_data.append(str_row)

        num_cols = len(str_data[0])
        if headers:
            for i in range(num_cols):
                max_data_len = max(len(str_data[r][i]) for r in range(len(str_data)))
                col_widths.append(max(max_data_len, len(headers[i])))
        else:
            for i in range(num_cols):
                max_data_len = max(len(str_data[r][i]) for r in range(len(str_data)))
                col_widths.append(max_data_len)

        def print_row(row):
            line = "| " + " | ".join(
                f"{val:{col_widths[i]}}" for i, val in enumerate(row)
            ) + " |"
            print(line)

        if headers:
            sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
            print(sep)
            print_row(headers)
            print(sep)

        for row in str_data:
            print_row(row)
            if headers:
                print(sep)

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ..Types.Types import *
from logging import Logger


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
    

    def plotHistogram(self, frame: Frame) -> Frame | None:
        """Plot and save the histogram of an image.

        Supports grayscale and BGR images. Saves the histogram as a PNG in the output directory.

        Args:
            frame (Frame): Image array (grayscale or BGR).

        Returns:
            Frame | None: The input frame, also displayed using OpenCV.
        
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

    
    def saveFrame(self, frame, tag: str, frameCount: int, logger: Logger, folder: str = "frames",) -> None:
        """Save an image frame with a tag and frame count.

        Args:
            frame (Frame): Image frame to save.
            tag (str): Descriptive tag for the frame.
            frameCount (int): Frame number to include in the filename.
            folder (str): Subfolder under the output path to save frames. Defaults to "frames".
        """
        folder_path = os.path.join(self.outputPath, folder)  # Create folder path
        os.makedirs(folder_path, exist_ok=True)  # Ensure folder exists

        # Construct the file path without redundant "frames/" string
        filename = os.path.join(folder_path, f"frame_{frameCount}_{tag}.png")
        
        # Save the frame
        success = cv.imwrite(filename, frame)
        if success:
            logger.info(f"Frame saved to {filename}")
        else:
            logger.info(f"Error saving frame to {filename}")
    
    def savePanelArray(self, panels: list[Frame], frameCount: int, logger: Logger, folder: str = "cells") -> None:
        """Save an array of panel images to disk.

        Args:
            panels (list[Frame]): List of panel image frames to save.
            frameCount (int): Frame number to include in the filenames.
            logger (Logger): Logger object for logging messages.
            folder (str): Subfolder under the output path to save panels. Defaults to "frames".
        """
        folder_path = os.path.join(self.outputPath, folder)  # Create folder path
        os.makedirs(folder_path, exist_ok=True)  # Ensure folder exists

        for idx, panel in enumerate(panels):
            filename = os.path.join(folder_path, f"frame_{frameCount}_panel_{idx}.png")
            success = cv.imwrite(filename, panel)
            if success:
                logger.info(f"Panel {idx} saved to {filename}")
            else:
                logger.info(f"Error saving panel {idx} to {filename}")

    def drawFrameCountours(self, frame: Frame, componentMask: Frame, cx: int, cy: int, lbl: int) -> None:
        """
        Draws the contours of a mask on the given frame and places a label at the specified centroid position.

        Args:
            frame (numpy.ndarray): The image frame on which to draw the contours and label.
            componentMask (numpy.ndarray): A binary mask image used to extract contours (non-zero pixels represent the object of interest).
            cx (int): The x-coordinate of the centroid where the label should be placed.
            cy (int): The y-coordinate of the centroid where the label should be placed.
            lbl (float): The label value to be displayed at the centroid position.

        Returns:
            None: The function modifies the `frame` in place by drawing the contours and adding the label.

        Notes:
            - The function assumes that the mask contains a single object (only the first contour is drawn).
            - The contour is drawn in green `(0, 255, 0)` with a thickness of 1 pixel.
            - The label is drawn in red `(0, 0, 255)` using the font `cv.FONT_HERSHEY_SIMPLEX`.
            - The label is displayed with 2 decimal points.
        """
        cv.drawContours(
            frame,
            [
                cv.findContours(
                    componentMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
                )[0][0]
            ],
            -1,
            (0, 255, 0),
            1,
        )
        cv.putText(
            frame,
            f"{lbl:.2f}",
            (cx, cy),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv.LINE_AA,
        )
    
    @staticmethod
    def drawLines(
        frame: np.ndarray, lines: np.ndarray | None, colour: str
    ) -> np.ndarray:
        if lines is not None and lines.shape[1] == 5:
            # remove count column for drawing
            lines = np.hstack((lines[:, :4], lines[:, 5:]))
        if lines is None:
            return frame
            # Prepare BGR output from binary input
        b = 0
        g = 0
        r = 0
        if colour == "g":
            g = 255
        elif colour == "r":
            r = 255
        elif colour == "b":
            b = 255
        if len(frame.shape) == 2:
            output = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            output = frame.copy()
        # Draw detected lines in green
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv.line(output, (x1, y1), (x2, y2), (b, g, r), 2)

        return output
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
    @staticmethod            
    def drawSpacingLines(frame: np.ndarray, rows: int, cols: int, height: int, width: int) -> np.ndarray:
        """
        Draws a rows x cols grid over the image with thick blue lines
        and a blue outer outline.
        """
        if len(frame.shape) == 2:
            output = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            output = frame.copy()

        h, w = height, width

        row_spacing = h / rows
        col_spacing = w / cols

        line_color = (255, 0, 0)  # Blue (BGR)
        line_thickness = 3

        # ---- Internal horizontal grid lines ----
        for i in range(1, rows):
            y = int(i * row_spacing)
            cv.line(output, (0, y), (w - 1, y), line_color, line_thickness, cv.LINE_AA)

        # ---- Internal vertical grid lines ----
        for i in range(1, cols):
            x = int(i * col_spacing)
            cv.line(output, (x, 0), (x, h - 1), line_color, line_thickness, cv.LINE_AA)

        # ---- Outer outline ----
        cv.rectangle(
            output, (0, 0), (w - 1, h - 1), line_color, line_thickness + 3, cv.LINE_AA
        )

        return output

    @staticmethod
    def splitImageToGrid(image: Frame, h_cells: int, v_cells: int):
        """
        Split an image into a grid of equally sized cells.

        The image is divided into `h_cells` rows and `v_cells` columns.
        Any remainder pixels (if the image is not perfectly divisible)
        are included in the last row/column cells.

        Args:
            image (Frame): Input image as a NumPy array of shape (H, W, C).
            h_cells (int): Number of horizontal grid cells (rows).
            v_cells (int): Number of vertical grid cells (columns).

        Returns:
            list[Frame]: A list of image cells in row-major order
                        (top-left to bottom-right), each as a Frame.
        """
        # height: int; width: int; cell_height: int; cell_width: int
        # height, width, _ = image.shape
        # cell_height = height // h_cells
        # cell_width = width // v_cells
        # cells: list[Frame] = []

        # for i in range(h_cells):
        #     for j in range(v_cells):
        #         y_start: int = i * cell_height
        #         y_end: int = (i + 1) * cell_height if i < h_cells - 1 else height
        #         x_start: int = j * cell_width
        #         x_end: int = (j + 1) * cell_width if j < v_cells - 1 else width

        #         cell_img: Frame = image[y_start:y_end, x_start:x_end, :]
        #         cells.append(cell_img)

        # return cells
        
        height, width, _ = image.shape
        cell_height = height // h_cells
        cell_width = width // v_cells
        cells = []

        for i in range(h_cells):
            for j in range(v_cells):
                y_start = i * cell_height
                y_end = (i + 1) * cell_height if i < h_cells - 1 else height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width if j < v_cells - 1 else width

                cell_img = image[y_start:y_end, x_start:x_end, :]
                cells.append(cell_img)

        return cells

    @staticmethod
    def stretchToAspectRatio(image: Frame, targetHeight: int, targetRatio: float):
        """
        Resize (stretch) an image to a target aspect ratio and height.

        This function changes the pixel dimensions of the image to match
        the desired aspect ratio by stretching (not cropping). New pixels
        are interpolated.

        Args:
            image (Frame): Input image as a NumPy array of shape (H, W, C).
            targetHeight (int): Desired output height in pixels.
            targetRatio (float): Desired width/height ratio.

        Returns:
            Frame: Resized image with shape (targetHeight, targetWidth, C).
        """
        targetWidth = int(targetHeight* targetRatio)
        stretched = cv.resize(image, (targetWidth, targetHeight), interpolation=cv.INTER_LINEAR)
        return stretched

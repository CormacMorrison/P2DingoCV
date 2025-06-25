import cv2 as cv
import numpy as np

def showImage(image_path: str, window_name: str ='Image'):
    """
    Load and display an image using OpenCV.

    Parameters
    ----------
    image_path : str
        Path to the image file (e.g., 'photo.jpg').
    window_name : str, optional
        Name of the window in which the image will be displayed. Default is 'Image'.

    Returns
    -------
    None
        Displays the image in a window until any key is pressed.

    Raises
    ------
    FileNotFoundError
        If the image file cannot be loaded.

    Notes
    -----
    Press any key while the image window is focused to close it.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Error: Could not read image from '{image_path}'")

    cv2.imshow(window_name, image)
    print("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def changeLiveRes(width, height):
    capture.set(3,width)
    capture.set(4,height)

def playVideo(source: int | str = 0, wait_time: int = 25, exit_key: str ='q', scale: float = 1):
    """
    Play a video from a file or webcam using OpenCV.

    Parameters
    ----------
    source : int or str, optional
        The video source. Use 0 (default) for the primary webcam, or a string 
        path to a video file (e.g., 'video.mp4').
    wait_time : int, optional
        Delay between frames in milliseconds. Use a smaller value (e.g., 1) 
        for real-time webcam playback. Default is 25.
    exit_key : str, optional
        The key to press in the video window to exit playback. Default is 'q'.

    Returns
    -------
    None
        Displays video frames in a window until the video ends or the exit key is pressed.

    Raises
    ------
    FileNotFoundError
        If the video source cannot be opened (e.g., invalid file path or webcam index).

    Notes
    -----
    This function uses `cv.imshow()` to display video frames and waits for
    `exit_key` to be pressed. It releases video resources and closes the display
    window on exit.
    """
    cap = cv.VideoCapture(source)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: '{source}'")

    print(f"Press '{exit_key}' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resize = rescaleFrame(frame) 

        cv.imshow('Video Playback', frame_resize)

        if cv.waitKey(wait_time) & 0xFF == ord(exit_key):
            break

    cap.release()
    cv.destroyAllWindows()

def rescaleFrame(frame: np.ndarray, scale: float = 0.75) -> np.ndarray:
    """
    Resize an image or video frame by a given scale factor.

    Parameters
    ----------
    frame : np.ndarray
        The input image or video frame as a NumPy array.
    scale : float, optional
        Scaling factor for resizing. Values less than 1 shrink the image,
        values greater than 1 enlarge it. Default is 0.75 (75% of original size).

    Returns
    -------
    np.ndarray
        The resized image or frame as a NumPy array.

    Notes
    -----
    Uses OpenCV's `cv.resize` with `INTER_AREA` interpolation, which is suitable
    for shrinking images while preserving quality.
    """
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


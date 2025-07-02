from Camera.Camera import Camera
from ImageProcessor.ImageProcecssor import ImageProcessor
from DisplayUnit import DisplayUnit

if __name__ == "__main__":
    cam = Camera('/home/cormac/P2Dingo/P2DingoCV/testImages/video.mp4')
    processor = ImageProcessor(cam._HEIGHT, cam._WIDTH)
    display = DisplayUnit(cam, processor)
    display.mock_run()

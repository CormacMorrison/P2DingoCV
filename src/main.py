from Camera.Camera import Camera
from ImageProcessor.ImageProcecssor import ImageProcessor
from DisplayUnit import DisplayUnit

if __name__ == "__main__":
    cam = Camera('/home/cormac/P2Dingo/P2DingoCV/testImages/Underside Solar Panels/Thermal Still.mp4')
    processor = ImageProcessor()
    display = DisplayUnit(cam, processor)
    display.mock_run()

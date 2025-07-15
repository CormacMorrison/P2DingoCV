from Camera.Camera import Camera
from ImageProcessor.ImageProcecssor import ImageProcessor
from DisplayUnit import DisplayUnit

if __name__ == "__main__":
    cam = Camera(0)
    processor = ImageProcessor(cam._HEIGHT, cam._WIDTH)
    display = DisplayUnit(cam, processor)
    display.runVisuals()

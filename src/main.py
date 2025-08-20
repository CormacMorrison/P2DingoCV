from Camera.Camera import Camera
from Camera.CameraFactory import CameraFactory
from ImageProcessor.ImageProcessor import ImageProcessor
from DisplayUnit import DisplayUnit
from ImageProcessor.HotspotDetector import HotspotDetector

if __name__ == "__main__":
    cam: Camera = CameraFactory.create("/home/cormac/P2Dingo/P2DingoCV/testImages/testNeg.png")
    processor: ImageProcessor = ImageProcessor(cam._HEIGHT, cam._WIDTH)
    hotspot: HotspotDetector = HotspotDetector(cam, processor)
    hotspot.execute()
    display = DisplayUnit(cam, processor)
    display.runVisuals()

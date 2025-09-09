
class TempDetectionFailed(Exception):
    """Exception raised when temperature detection fails.

    This exception can be used in image processing or thermal analysis pipelines
    to signal that temperature could not be reliably measured for a frame or component.

    Args:
        message (str, optional): Custom error message. Defaults to 
            "Temperature detection failed".
    """
    def __init__(self, message="Temperature detection failed"):
        super().__init__(message)

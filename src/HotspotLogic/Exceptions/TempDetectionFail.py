
class TempDetectionFailed(Exception):
    def __init__(self, message="Temperature detection failed"):
        super().__init__(message)

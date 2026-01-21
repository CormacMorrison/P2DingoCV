import numpy as np
from typing import TypeAlias
from numpy.typing import NDArray, ArrayLike
from typing import Sequence
from cv2.typing import MatLike

Frame: TypeAlias = np.ndarray
FloatFrame: TypeAlias = NDArray[np.float32]
ChannelList: TypeAlias = Sequence[MatLike]
Channel: TypeAlias = np.ndarray



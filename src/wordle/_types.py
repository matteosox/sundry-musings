"""Types for Wordle package"""

from typing import Union

import numpy as np
from numpy import typing as npt
from typing_extensions import TypeAlias

UIntType: TypeAlias = Union[np.uint8, np.uint16, np.uint32]
UIntArrayType: TypeAlias = npt.NDArray[UIntType]
IntArrayType: TypeAlias = npt.NDArray[np.int64]
FloatArrayType: TypeAlias = npt.NDArray[np.float64]

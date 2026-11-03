"""AudioSource プロトコル。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class AudioSource(Protocol):
    """音声ブロックを供給する入力源。

    ブロックは float32・形状 (frames, channels)、値域はおおよそ [-1, 1]。
    """

    @property
    def sample_rate(self) -> int: ...

    @property
    def channels(self) -> int: ...

    def __iter__(self) -> Iterator[NDArray[np.float32]]: ...

    def close(self) -> None: ...

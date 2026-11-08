"""音声の取得スレッド(AudioSource から読み、モノラル int16 / 16kHz に変換して渡す)。

`start` の文字起こしと `record` の録音で共有する。
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from loopscribe.audio.convert import AudioConverter
from loopscribe.audio.source import AudioSource
from loopscribe.constants import SAMPLE_RATE


class AudioCapture:
    """capture スレッドで AudioSource を読み、変換したサンプルを `on_samples` に渡す。

    取得に失敗したら、警告を出して `stop_event` をセットし、`failed` を立てる。
    スレッドの終わりにソースを閉じて、`on_finish` を呼ぶ。
    """

    def __init__(
        self,
        source: AudioSource,
        on_samples: Callable[[NDArray[np.int16]], None],
        *,
        stop_event: threading.Event,
        on_warning: Callable[[str], None],
        on_finish: Callable[[], None] | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._source = source
        self._on_samples = on_samples
        self._stop_event = stop_event
        self._on_warning = on_warning
        self._on_finish = on_finish
        self._sample_rate = sample_rate
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._source_closed = False
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def close_source_if_stopped(self) -> None:
        """停止の指示があれば、ソースを1回だけ閉じる(メインスレッドから呼ぶ)。

        無音時の WASAPI はブロックを返さず、capture 側が停止に気づけないため。
        """
        if not self._stop_event.is_set() or self._source_closed:
            return
        self._source_closed = True
        with contextlib.suppress(Exception):
            self._source.close()

    def _run(self) -> None:
        converter = AudioConverter(
            self._source.sample_rate,
            self._source.channels,
            out_rate=self._sample_rate,
        )
        try:
            for block in self._source:
                if self._stop_event.is_set():
                    break
                self._emit(converter.convert(block))
            # 停止または EOF: リサンプラをフラッシュ
            self._emit(
                converter.convert(
                    np.empty((0, self._source.channels), dtype=np.float32),
                    last=True,
                )
            )
        except Exception as exc:
            self._failed = True
            self._on_warning(f"音声の取得に失敗したため録音を停止します: {exc}")
            self._stop_event.set()
        finally:
            with contextlib.suppress(Exception):
                self._source.close()
            if self._on_finish is not None:
                self._on_finish()

    def _emit(self, samples: NDArray[np.int16]) -> None:
        if samples.size == 0:
            return
        self._on_samples(samples)

"""WASAPI ループバックによるシステム音声取得(Windows 専用)。

`pyaudiowpatch` の import はこのモジュール内でのみ行い、かつ遅延させる。
WSL では本モジュールを import しても依存欠落で落ちないようにする
(実際のデバイス操作時に ImportError / FatalStartupError になる)。
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rich.console import Console

from loopscribe.constants import WASAPI_FRAMES_PER_BUFFER, WASAPI_QUEUE_POLL_SECONDS
from loopscribe.errors import DeviceNotFoundError, FatalStartupError

# イテレータ終了用の番兵
_SENTINEL = object()


@dataclass(frozen=True)
class OutputDeviceInfo:
    """ループバック取得の対象になる出力デバイス。"""

    index: int
    name: str
    sample_rate: int
    channels: int
    is_default: bool
    loopback_index: int
    loopback_name: str
    loopback_sample_rate: int
    loopback_channels: int


def _import_pyaudio() -> Any:
    """pyaudiowpatch を遅延 import する。未導入なら FatalStartupError。"""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise FatalStartupError(
            "pyaudiowpatch を import できません。"
            "Windows で `uv sync` / `uv tool install` 済みか確認してください。"
            f" ({exc})"
        ) from exc
    return pyaudio


def _wasapi_host_api(pyaudio_mod: Any, pa: Any) -> dict[str, Any]:
    try:
        return pa.get_host_api_info_by_type(pyaudio_mod.paWASAPI)
    except OSError as exc:
        raise FatalStartupError(
            "WASAPI が利用できません。Windows 上で実行しているか確認してください。"
        ) from exc


def _build_output_devices(pa: Any, pyaudio_mod: Any) -> list[OutputDeviceInfo]:
    """WASAPI 出力デバイスのうち、ループバック対応があるものを列挙する。"""
    wasapi_info = _wasapi_host_api(pyaudio_mod, pa)
    default_out = int(wasapi_info["defaultOutputDevice"])
    devices: list[OutputDeviceInfo] = []

    for device in pa.get_device_info_generator_by_host_api(host_api_type=pyaudio_mod.paWASAPI):
        if device.get("isLoopbackDevice"):
            continue
        if int(device.get("maxOutputChannels", 0)) < 1:
            continue
        try:
            loopback = pa.get_wasapi_loopback_analogue_by_dict(device)
        except (LookupError, ValueError, KeyError):
            continue
        index = int(device["index"])
        devices.append(
            OutputDeviceInfo(
                index=index,
                name=str(device["name"]),
                sample_rate=int(device["defaultSampleRate"]),
                channels=int(device["maxOutputChannels"]),
                is_default=(index == default_out),
                loopback_index=int(loopback["index"]),
                loopback_name=str(loopback["name"]),
                loopback_sample_rate=int(loopback["defaultSampleRate"]),
                loopback_channels=int(loopback["maxInputChannels"]),
            )
        )
    return devices


def list_output_devices() -> list[OutputDeviceInfo]:
    """ループバック対象の出力デバイスを一覧する。"""
    pyaudio_mod = _import_pyaudio()
    with pyaudio_mod.PyAudio() as pa:
        devices = _build_output_devices(pa, pyaudio_mod)
    if not devices:
        raise DeviceNotFoundError(
            "ループバック可能な出力デバイスが見つかりません。"
            "`loopscribe devices` で一覧を確認してください。"
        )
    return devices


def resolve_device(
    selector: str | None,
    devices: Sequence[OutputDeviceInfo],
) -> OutputDeviceInfo:
    """`--device` の指定(番号または名前の一部)を解決する。None なら既定デバイス。"""
    if not devices:
        raise DeviceNotFoundError(
            "ループバック可能な出力デバイスが見つかりません。"
            "`loopscribe devices` で一覧を確認してください。"
        )

    if selector is None or selector.strip() == "":
        for device in devices:
            if device.is_default:
                return device
        # 既定フラグが無い場合は先頭を使う(異常系のフォールバック)
        return devices[0]

    text = selector.strip()
    if text.isdigit():
        index = int(text)
        matches = [d for d in devices if d.index == index or d.loopback_index == index]
        if not matches:
            raise DeviceNotFoundError(
                f"デバイス番号 {index} は見つかりません。"
                "`loopscribe devices` で一覧を確認してください。"
            )
        return matches[0]

    needle = text.casefold()
    matches = [
        d for d in devices if needle in d.name.casefold() or needle in d.loopback_name.casefold()
    ]
    if not matches:
        raise DeviceNotFoundError(
            f"名前に '{selector}' を含むデバイスが見つかりません。"
            "`loopscribe devices` で一覧を確認してください。"
        )
    if len(matches) > 1:
        listed = ", ".join(f"[{d.index}] {d.name}" for d in matches)
        raise DeviceNotFoundError(
            f"名前に '{selector}' を含むデバイスが複数あります: {listed}。番号で指定してください。"
        )
    return matches[0]


def print_device_list(
    devices: Sequence[OutputDeviceInfo],
    *,
    console: Console,
) -> None:
    """デバイス一覧を rich.Console に表示する。"""
    if not devices:
        console.print("ループバック可能な出力デバイスはありません。")
        return
    console.print("出力デバイス(ループバック対象):")
    for device in devices:
        mark = "*" if device.is_default else " "
        console.print(
            f"  {mark} [{device.index:>3}] {device.name}"
            f"  {device.sample_rate}Hz  {device.channels}ch"
            f"  (loopback [{device.loopback_index}])"
        )
    console.print("  (* = 既定の出力デバイス。--device には番号または名前の一部を指定)")


def open_loopback_source(
    device: str | None = None,
    *,
    frames_per_buffer: int = WASAPI_FRAMES_PER_BUFFER,
) -> WasapiLoopbackSource:
    """`--device` 指定を解決し、ループバック用 AudioSource を返す。

    ストリームは `__iter__` 開始時に開く(モデル読み込み中に録音しないため)。
    """
    devices = list_output_devices()
    info = resolve_device(device, devices)
    return WasapiLoopbackSource(info, frames_per_buffer=frames_per_buffer)


class WasapiLoopbackSource:
    """WASAPI ループバックをブロック供給する AudioSource。

    コールバックは `queue.SimpleQueue` に生バイトを積むだけにし、
    float32 への変換は capture スレッド側(`__iter__`)で行う。
    """

    def __init__(
        self,
        device: OutputDeviceInfo,
        *,
        frames_per_buffer: int = WASAPI_FRAMES_PER_BUFFER,
        queue_poll_seconds: float = WASAPI_QUEUE_POLL_SECONDS,
    ) -> None:
        if frames_per_buffer <= 0:
            raise ValueError("frames_per_buffer は 1 以上であること")
        if queue_poll_seconds <= 0:
            raise ValueError("queue_poll_seconds は正の数であること")
        if device.loopback_channels < 1:
            raise DeviceNotFoundError(
                f"ループバックデバイスのチャンネル数が不正です: {device.loopback_name}"
            )
        self._device = device
        self._frames_per_buffer = frames_per_buffer
        self._queue_poll_seconds = queue_poll_seconds
        self._sample_rate = device.loopback_sample_rate
        self._channels = device.loopback_channels
        self._queue: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._closed = False
        self._close_lock = threading.Lock()
        self._pyaudio_mod: Any | None = None
        self._pa: Any | None = None
        self._stream: Any | None = None
        self._started = False
        self._stream_error: BaseException | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def device(self) -> OutputDeviceInfo:
        return self._device

    def __iter__(self) -> Iterator[NDArray[np.float32]]:
        self._ensure_started()
        while True:
            try:
                item = self._queue.get(timeout=self._queue_poll_seconds)
            except queue.Empty:
                if self._closed:
                    break
                if self._stream_error is not None:
                    raise RuntimeError(
                        f"音声の取得に失敗しました: {self._stream_error}"
                    ) from self._stream_error
                if self._stream is not None and not self._stream.is_active():
                    raise RuntimeError(
                        "音声ストリームが停止しました(デバイス切断の可能性)"
                    ) from None
                continue

            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise RuntimeError(f"音声の取得に失敗しました: {item}") from item
            if not isinstance(item, (bytes, bytearray)):
                continue
            if not item:
                continue
            yield self._bytes_to_block(bytes(item))

        if self._stream_error is not None:
            raise RuntimeError(
                f"音声の取得に失敗しました: {self._stream_error}"
            ) from self._stream_error

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        stream = self._stream
        if stream is not None:
            with contextlib.suppress(Exception):
                if stream.is_active():
                    stream.stop_stream()
            with contextlib.suppress(Exception):
                stream.close()
            self._stream = None
        pa = self._pa
        if pa is not None:
            with contextlib.suppress(Exception):
                pa.terminate()
            self._pa = None
        self._queue.put(_SENTINEL)

    def _ensure_started(self) -> None:
        if self._started:
            return
        if self._closed:
            raise ValueError("close 済みの WasapiLoopbackSource は読めない")
        pyaudio_mod = _import_pyaudio()
        self._pyaudio_mod = pyaudio_mod
        pa = pyaudio_mod.PyAudio()
        self._pa = pa
        try:
            stream = pa.open(
                format=pyaudio_mod.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                frames_per_buffer=self._frames_per_buffer,
                input=True,
                input_device_index=self._device.loopback_index,
                stream_callback=self._callback,
            )
        except Exception as exc:
            with contextlib.suppress(Exception):
                pa.terminate()
            self._pa = None
            raise FatalStartupError(
                f"ループバックストリームを開けませんでした"
                f"([{self._device.loopback_index}] {self._device.loopback_name}): {exc}"
            ) from exc
        self._stream = stream
        self._started = True

    def _callback(
        self,
        in_data: bytes | None,
        _frame_count: int,
        _time_info: object,
        _status: int,
    ) -> tuple[None, int]:
        pyaudio_mod = self._pyaudio_mod
        if pyaudio_mod is None:
            return (None, 1)  # paComplete 相当のフォールバック
        if self._closed:
            return (None, pyaudio_mod.paComplete)
        # _status 非ゼロはオーバーフロー等。切断は is_active 側で検知する。
        if in_data:
            self._queue.put(in_data)
        return (None, pyaudio_mod.paContinue)

    def _bytes_to_block(self, raw: bytes) -> NDArray[np.float32]:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        samples /= 32768.0
        frames = samples.size // self._channels
        if frames * self._channels != samples.size:
            raise RuntimeError("ループバック音声のフレーム境界が不正です")
        return samples.reshape(frames, self._channels)

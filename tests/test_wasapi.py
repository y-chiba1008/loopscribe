"""wasapi の単体テスト(実デバイスは使わず、解決ロジックとモックで検証)。"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console

from loopscribe.audio.wasapi import (
    OutputDeviceInfo,
    WasapiLoopbackSource,
    open_loopback_source,
    print_device_list,
    resolve_device,
)
from loopscribe.errors import DeviceNotFoundError, FatalStartupError


def _device(
    *,
    index: int,
    name: str,
    is_default: bool = False,
    loopback_index: int | None = None,
) -> OutputDeviceInfo:
    lb = index + 100 if loopback_index is None else loopback_index
    return OutputDeviceInfo(
        index=index,
        name=name,
        sample_rate=48_000,
        channels=2,
        is_default=is_default,
        loopback_index=lb,
        loopback_name=f"{name} [Loopback]",
        loopback_sample_rate=48_000,
        loopback_channels=2,
    )


def test_resolve_default_device() -> None:
    devices = [
        _device(index=1, name="Headphones"),
        _device(index=5, name="Speakers", is_default=True),
    ]
    assert resolve_device(None, devices).index == 5
    assert resolve_device("", devices).index == 5


def test_resolve_by_index_and_loopback_index() -> None:
    devices = [_device(index=5, name="Speakers", loopback_index=15)]
    assert resolve_device("5", devices).index == 5
    assert resolve_device("15", devices).index == 5


def test_resolve_by_name_substring() -> None:
    devices = [
        _device(index=1, name="Headphones (USB)"),
        _device(index=5, name="Speakers (Realtek)", is_default=True),
    ]
    assert resolve_device("realtek", devices).index == 5
    assert resolve_device("Head", devices).index == 1


def test_resolve_ambiguous_name_raises() -> None:
    devices = [
        _device(index=1, name="USB Speakers A"),
        _device(index=2, name="USB Speakers B"),
    ]
    with pytest.raises(DeviceNotFoundError, match="複数"):
        resolve_device("USB", devices)


def test_resolve_missing_raises() -> None:
    devices = [_device(index=5, name="Speakers", is_default=True)]
    with pytest.raises(DeviceNotFoundError, match="99"):
        resolve_device("99", devices)
    with pytest.raises(DeviceNotFoundError, match="HDMI"):
        resolve_device("HDMI", devices)
    with pytest.raises(DeviceNotFoundError, match="見つかりません"):
        resolve_device(None, [])


def test_print_device_list_marks_default() -> None:
    console = Console(record=True, width=120)
    devices = [
        _device(index=1, name="Headphones"),
        _device(index=5, name="Speakers", is_default=True),
    ]
    print_device_list(devices, console=console)
    text = console.export_text()
    assert "[  5] Speakers" in text
    assert "*" in text
    assert "loopback [105]" in text


def test_bytes_to_block_shape() -> None:
    info = _device(index=5, name="Speakers", is_default=True)
    source = WasapiLoopbackSource(info)
    # 2ch × 2 frames の int16 little-endian
    raw = np.array([1000, -1000, 2000, -2000], dtype="<i2").tobytes()
    block = source._bytes_to_block(raw)
    assert block.shape == (2, 2)
    assert block.dtype == np.float32
    assert abs(block[0, 0] - 1000 / 32768.0) < 1e-6


def test_wasapi_source_iterates_queued_blocks() -> None:
    from loopscribe.audio import wasapi as wasapi_mod

    info = _device(index=5, name="Speakers", is_default=True)
    source = WasapiLoopbackSource(info, queue_poll_seconds=0.05)
    source._ensure_started = lambda: None  # type: ignore[method-assign]
    source._started = True
    source._stream = MagicMock(is_active=MagicMock(return_value=True))

    pcm = np.array([1000, -1000], dtype="<i2").tobytes()
    source._queue.put(pcm)
    source._queue.put(wasapi_mod._SENTINEL)

    blocks = list(source)
    assert len(blocks) == 1
    assert blocks[0].shape == (1, 2)


def test_wasapi_source_close_from_other_thread_ends_silent_iteration() -> None:
    """無音でキューが空のままでも、別スレッドからの close で反復が終わる。"""
    info = _device(index=5, name="Speakers", is_default=True)
    source = WasapiLoopbackSource(info, queue_poll_seconds=0.05)
    source._ensure_started = lambda: None  # type: ignore[method-assign]
    source._started = True
    source._stream = MagicMock(is_active=MagicMock(return_value=True))

    blocks: list[np.ndarray] = []
    reader = threading.Thread(target=lambda: blocks.extend(source), daemon=True)
    reader.start()
    time.sleep(0.2)
    assert reader.is_alive()

    source.close()
    reader.join(timeout=2.0)
    assert not reader.is_alive()
    assert blocks == []


def test_wasapi_source_raises_when_stream_stops() -> None:
    info = _device(index=5, name="Speakers", is_default=True)
    source = WasapiLoopbackSource(info, queue_poll_seconds=0.05)
    source._ensure_started = lambda: None  # type: ignore[method-assign]
    source._started = True
    source._stream = MagicMock(is_active=MagicMock(return_value=False))

    with pytest.raises(RuntimeError, match="ストリームが停止"):
        next(iter(source))


def test_list_output_devices_builds_from_pyaudio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speakers = {
        "index": 5,
        "name": "Speakers",
        "defaultSampleRate": 48000.0,
        "maxOutputChannels": 2,
        "maxInputChannels": 0,
        "isLoopbackDevice": False,
    }
    loopback = {
        "index": 15,
        "name": "Speakers [Loopback]",
        "defaultSampleRate": 48000.0,
        "maxInputChannels": 2,
        "isLoopbackDevice": True,
    }
    no_loopback = {
        "index": 7,
        "name": "HDMI",
        "defaultSampleRate": 48000.0,
        "maxOutputChannels": 2,
        "maxInputChannels": 0,
        "isLoopbackDevice": False,
    }

    class FakePa:
        def __enter__(self) -> FakePa:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_host_api_info_by_type(self, _t: object) -> dict[str, int]:
            return {"defaultOutputDevice": 5}

        def get_device_info_generator_by_host_api(self, **_kwargs: object):
            yield speakers
            yield no_loopback
            yield loopback

        def get_wasapi_loopback_analogue_by_dict(
            self,
            info: dict[str, object],
        ) -> dict[str, object]:
            if info["name"] == "Speakers":
                return loopback
            raise LookupError("none")

    fake_mod = type(
        "M",
        (),
        {"paWASAPI": 13, "PyAudio": staticmethod(lambda: FakePa())},
    )()
    monkeypatch.setattr("loopscribe.audio.wasapi._import_pyaudio", lambda: fake_mod)

    from loopscribe.audio.wasapi import list_output_devices

    devices = list_output_devices()
    assert len(devices) == 1
    assert devices[0].index == 5
    assert devices[0].is_default is True
    assert devices[0].loopback_index == 15


def test_open_loopback_source_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [
        _device(index=1, name="Headphones"),
        _device(index=5, name="Speakers", is_default=True),
    ]
    monkeypatch.setattr(
        "loopscribe.audio.wasapi.list_output_devices",
        lambda: devices,
    )
    source = open_loopback_source("Speakers")
    assert isinstance(source, WasapiLoopbackSource)
    assert source.device.index == 5
    assert source.sample_rate == 48_000
    assert source.channels == 2


def test_import_pyaudio_wraps_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyaudiowpatch":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    from loopscribe.audio.wasapi import _import_pyaudio

    with pytest.raises(FatalStartupError, match="pyaudiowpatch"):
        _import_pyaudio()

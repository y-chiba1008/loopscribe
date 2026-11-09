"""cli の単体テスト。"""

from __future__ import annotations

import re
import sys
import threading
import wave
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from loopscribe.audio.wasapi import OutputDeviceInfo
from loopscribe.audio.wavfile import WavFileSource
from loopscribe.cli import app, run_file_transcription
from loopscribe.config import read_raw
from loopscribe.errors import DeviceNotFoundError, ModelUnavailableError
from loopscribe.models.kotoba import KotobaModel
from loopscribe.models.small import SmallModel
from loopscribe.transcriber import FakeTranscriber, Segment

runner = CliRunner()


def _fake_device() -> OutputDeviceInfo:
    return OutputDeviceInfo(
        index=5,
        name="Speakers",
        sample_rate=48_000,
        channels=2,
        is_default=True,
        loopback_index=15,
        loopback_name="Speakers [Loopback]",
        loopback_sample_rate=48_000,
        loopback_channels=2,
    )


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "loopscribe" in result.stdout
    assert "1.1.0" in result.stdout


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.stdout
    assert "record" in result.stdout
    assert "transcribe" in result.stdout
    assert "download-model" in result.stdout
    assert "devices" in result.stdout


def test_start_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    result = runner.invoke(app, ["start"])
    assert result.exit_code == 1
    assert "Windows" in result.stdout


def test_devices_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 1
    assert "Windows" in result.stdout


def test_start_help_shows_options() -> None:
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--model" in result.stdout


def test_download_model_help() -> None:
    result = runner.invoke(app, ["download-model", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.stdout


def test_download_model_defaults_to_small(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        SmallModel, "download", lambda self: called.append(self.name) or Path("/cache/small")
    )
    result = runner.invoke(app, ["download-model"])
    assert result.exit_code == 0
    assert called == ["small"]
    assert "取得完了" in result.stdout


def test_download_model_kotoba(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        KotobaModel, "download", lambda self: called.append(self.name) or Path("/cache/kotoba")
    )
    result = runner.invoke(app, ["download-model", "--model", "kotoba"])
    assert result.exit_code == 0
    assert called == ["kotoba"]
    assert "kotoba-tech/kotoba-whisper-v2.0-faster" in result.stdout


def test_download_model_failure_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(self: SmallModel) -> Path:
        raise ModelUnavailableError("モデル 'small' の取得に失敗しました: network down")

    monkeypatch.setattr(SmallModel, "download", _raise)
    result = runner.invoke(app, ["download-model"])
    assert result.exit_code == 1
    assert "エラー" in result.stdout


def test_download_model_rejects_unknown_model() -> None:
    result = runner.invoke(app, ["download-model", "--model", "large"])
    assert result.exit_code == 2


def test_devices_lists_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    devices = [_fake_device()]
    printed: list[tuple[object, object]] = []

    fake_wasapi = SimpleNamespace(
        list_output_devices=lambda: devices,
        print_device_list=lambda devs, console=None: printed.append((devs, console)),
    )
    monkeypatch.setattr("loopscribe.cli._load_wasapi", lambda: fake_wasapi)

    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 0
    assert printed
    assert printed[0][0] == devices
    assert isinstance(printed[0][1], Console)


def test_devices_device_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    def _raise() -> list[OutputDeviceInfo]:
        raise DeviceNotFoundError("ループバック可能な出力デバイスが見つかりません。")

    fake_wasapi = SimpleNamespace(
        list_output_devices=_raise,
        print_device_list=lambda *_a, **_k: None,
    )
    monkeypatch.setattr("loopscribe.cli._load_wasapi", lambda: fake_wasapi)

    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 1
    assert "エラー" in result.stdout


def test_start_opens_loopback_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    opened: list[str | None] = []

    class FakeSource:
        sample_rate = 48_000
        channels = 2

        def __iter__(self):
            return iter(())

        def close(self) -> None:
            return None

    fake_wasapi = SimpleNamespace(
        open_loopback_source=lambda device: opened.append(device) or FakeSource(),
    )
    monkeypatch.setattr("loopscribe.cli._load_wasapi", lambda: fake_wasapi)
    models: list[str] = []
    monkeypatch.setattr(
        "loopscribe.cli.run_transcription",
        lambda source, console, output_dir, model_name, output_path: models.append(model_name) or 0,
    )

    result = runner.invoke(app, ["start", "--device", "Speakers"])
    assert result.exit_code == 0
    assert opened == ["Speakers"]
    assert models == ["small"]

    result = runner.invoke(app, ["start", "--model", "kotoba"])
    assert result.exit_code == 0
    assert models == ["small", "kotoba"]


def test_start_rejects_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    result = runner.invoke(app, ["start", "--model", "large"])
    assert result.exit_code == 2


def test_start_help_shows_output_name() -> None:
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code == 0
    assert "--output-name" in result.stdout


def test_start_rejects_path_in_output_name_before_opening_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    opened: list[str | None] = []
    fake_wasapi = SimpleNamespace(
        open_loopback_source=lambda device: opened.append(device) or _EmptySource(),
    )
    monkeypatch.setattr("loopscribe.cli._load_wasapi", lambda: fake_wasapi)

    result = runner.invoke(app, ["start", "-n", "notes/meeting"])
    assert result.exit_code == 1
    assert "エラー" in result.stdout
    assert opened == []


class _EmptySource:
    sample_rate = 16_000
    channels = 1

    def __iter__(self):
        return iter(())

    def close(self) -> None:
        return None


class _FakeSpec:
    repo_id = "fake/model"

    def __init__(
        self,
        on_ensure: Callable[[], None] | None = None,
        transcriber: FakeTranscriber | None = None,
    ) -> None:
        self._on_ensure = on_ensure
        self._transcriber = transcriber or FakeTranscriber()

    def ensure(self) -> Path:
        if self._on_ensure is not None:
            self._on_ensure()
        return Path("/cache/fake")

    def load(self, model_path: Path) -> FakeTranscriber:
        return self._transcriber


def _patch_start(monkeypatch: pytest.MonkeyPatch, spec: _FakeSpec) -> list[str | None]:
    monkeypatch.setattr(sys, "platform", "win32")
    opened: list[str | None] = []
    fake_wasapi = SimpleNamespace(
        open_loopback_source=lambda device: opened.append(device) or _EmptySource(),
    )
    monkeypatch.setattr("loopscribe.cli._load_wasapi", lambda: fake_wasapi)
    monkeypatch.setattr("loopscribe.cli.get_model", lambda name: spec)
    return opened


@pytest.mark.parametrize("name", ["meeting", "meeting.txt"])
def test_start_writes_output_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    _patch_start(monkeypatch, _FakeSpec())
    result = runner.invoke(app, ["start", "-o", str(tmp_path), "-n", name])
    assert result.exit_code == 0
    assert [p.name for p in tmp_path.iterdir()] == ["meeting.txt"]
    assert "警告" not in result.stdout


def test_start_default_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_start(monkeypatch, _FakeSpec())
    result = runner.invoke(app, ["start", "-o", str(tmp_path)])
    assert result.exit_code == 0
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert re.fullmatch(r"loopscribe_\d{8}_\d{6}\.txt", files[0].name)


def test_start_warns_existing_name_before_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "meeting.txt").write_text("既存", encoding="utf-8")
    opened = _patch_start(monkeypatch, _FakeSpec())

    result = runner.invoke(app, ["start", "-o", str(tmp_path), "-n", "meeting"])
    assert result.exit_code == 0
    assert opened == [None]
    assert (tmp_path / "meeting.txt").read_text(encoding="utf-8") == "既存"
    assert (tmp_path / "meeting_2.txt").is_file()
    out = result.stdout
    assert out.count("警告") == 1
    assert out.index("meeting_2.txt に保存します") < out.index("モデルの読み込みが完了しました")


def test_start_numbers_file_created_while_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _create_same_name() -> None:
        (tmp_path / "meeting.txt").write_text("読み込み中に作られた", encoding="utf-8")

    _patch_start(monkeypatch, _FakeSpec(on_ensure=_create_same_name))

    result = runner.invoke(app, ["start", "-o", str(tmp_path), "-n", "meeting"])
    assert result.exit_code == 0
    assert (tmp_path / "meeting.txt").read_text(encoding="utf-8") == "読み込み中に作られた"
    assert (tmp_path / "meeting_2.txt").is_file()
    assert "meeting_2.txt に保存します" in result.stdout


def _patch_record(monkeypatch: pytest.MonkeyPatch, source_factory=_EmptySource) -> list[str | None]:
    monkeypatch.setattr(sys, "platform", "win32")
    opened: list[str | None] = []
    fake_wasapi = SimpleNamespace(
        open_loopback_source=lambda device: opened.append(device) or source_factory(),
    )
    monkeypatch.setattr("loopscribe.cli._load_wasapi", lambda: fake_wasapi)
    monkeypatch.setattr(
        "loopscribe.cli.get_model",
        lambda name: pytest.fail("record はモデルを読み込まない"),
    )
    return opened


def test_record_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    result = runner.invoke(app, ["record"])
    assert result.exit_code == 1
    assert "Windows" in result.stdout


def test_record_help_shows_options() -> None:
    result = runner.invoke(app, ["record", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--output-name" in result.stdout
    assert "--model" not in result.stdout


@pytest.mark.parametrize("name", ["meeting", "meeting.wav", "meeting.WAV"])
def test_record_writes_output_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    opened = _patch_record(monkeypatch)
    result = runner.invoke(app, ["record", "--device", "2", "-o", str(tmp_path), "-n", name])
    assert result.exit_code == 0
    assert opened == ["2"]
    assert [p.name for p in tmp_path.iterdir()] == ["meeting.wav"]
    with wave.open(str(tmp_path / "meeting.wav"), "rb") as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16_000)
    assert "警告" not in result.stdout


def test_record_default_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_record(monkeypatch)
    result = runner.invoke(app, ["record", "-o", str(tmp_path)])
    assert result.exit_code == 0
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert re.fullmatch(r"loopscribe_\d{8}_\d{6}\.wav", files[0].name)


def test_record_numbers_existing_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "meeting.wav").write_bytes(b"existing")
    _patch_record(monkeypatch)
    result = runner.invoke(app, ["record", "-o", str(tmp_path), "-n", "meeting"])
    assert result.exit_code == 0
    assert (tmp_path / "meeting.wav").read_bytes() == b"existing"
    assert (tmp_path / "meeting_2.wav").is_file()
    assert result.stdout.count("警告") == 1


def test_record_rejects_path_in_output_name_before_opening_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened = _patch_record(monkeypatch)
    result = runner.invoke(app, ["record", "-o", str(tmp_path), "-n", "notes\\meeting"])
    assert result.exit_code == 1
    assert "エラー" in result.stdout
    assert opened == []
    assert list(tmp_path.iterdir()) == []


def test_record_device_not_found_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _raise() -> _EmptySource:
        raise DeviceNotFoundError("出力デバイスが見つかりません: X")

    _patch_record(monkeypatch, source_factory=_raise)
    result = runner.invoke(app, ["record", "-o", str(tmp_path), "--device", "X"])
    assert result.exit_code == 1
    assert "loopscribe devices" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_record_capture_failure_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FailingSource(_EmptySource):
        def __iter__(self):
            raise OSError("device lost")

    _patch_record(monkeypatch, source_factory=FailingSource)
    result = runner.invoke(app, ["record", "-o", str(tmp_path), "-n", "meeting"])
    assert result.exit_code == 1
    assert "device lost" in result.stdout
    assert (tmp_path / "meeting.wav").is_file()


def _write_wav(path: Path, *, seconds: float, sample_rate: int = 16_000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(seconds * sample_rate) * channels)


def _patch_transcribe(monkeypatch: pytest.MonkeyPatch, spec: _FakeSpec | None = None) -> list[str]:
    """Windows 以外として、モデルを偽物にする。読み込んだモデル名を返す。"""
    monkeypatch.setattr(sys, "platform", "linux")
    loaded: list[str] = []
    fake_spec = spec or _FakeSpec()

    def _get_model(name: str) -> _FakeSpec:
        loaded.append(name)
        return fake_spec

    monkeypatch.setattr("loopscribe.cli.get_model", _get_model)
    return loaded


def test_transcribe_help_shows_options() -> None:
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "FILENAME" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--output-name" in result.stdout
    assert "--model" in result.stdout
    assert "--device" not in result.stdout


def test_transcribe_writes_txt_named_after_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wav = tmp_path / "meeting.wav"
    _write_wav(wav, seconds=2.0, sample_rate=48_000, channels=2)
    out_dir = tmp_path / "out"
    fake = FakeTranscriber([[Segment(start=0.5, end=1.0, text="こんにちは")]])
    spec = _FakeSpec(transcriber=fake)
    loaded = _patch_transcribe(monkeypatch, spec)

    result = runner.invoke(app, ["transcribe", str(wav), "-o", str(out_dir)])
    assert result.exit_code == 0
    assert loaded == ["small"]
    assert [p.name for p in out_dir.iterdir()] == ["meeting.txt"]
    lines = (out_dir / "meeting.txt").read_text(encoding="utf-8").splitlines()
    assert re.fullmatch(
        r"# loopscribe 入力: meeting\.wav / 処理開始: [\d-]+ [\d:]+ / モデル: fake/model", lines[0]
    )
    assert lines[1:] == ["[00:00:00] こんにちは"]
    assert "[00:00:00] こんにちは" in result.stdout
    assert "警告" not in result.stdout


def test_transcribe_output_name_and_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wav = tmp_path / "meeting.wav"
    _write_wav(wav, seconds=1.0)
    (tmp_path / "notes.txt").write_text("既存", encoding="utf-8")
    loaded = _patch_transcribe(monkeypatch)

    result = runner.invoke(
        app, ["transcribe", str(wav), "-o", str(tmp_path), "-n", "notes.txt", "--model", "kotoba"]
    )
    assert result.exit_code == 0
    assert loaded == ["kotoba"]
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "既存"
    assert (tmp_path / "notes_2.txt").is_file()
    out = result.stdout
    assert out.count("警告") == 1
    assert out.index("notes_2.txt に保存します") < out.index("モデルの読み込みが完了しました")


def test_transcribe_missing_file_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded = _patch_transcribe(monkeypatch)
    result = runner.invoke(app, ["transcribe", str(tmp_path / "none.wav"), "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "入力ファイルがありません" in result.stdout
    assert loaded == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("content", [b"", b"not a wav file"])
def test_transcribe_invalid_wav_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: bytes
) -> None:
    wav = tmp_path / "broken.wav"
    wav.write_bytes(content)
    loaded = _patch_transcribe(monkeypatch)
    result = runner.invoke(app, ["transcribe", str(wav), "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "wav を読み込めません" in result.stdout
    assert loaded == []
    assert [p.name for p in tmp_path.iterdir()] == ["broken.wav"]


def test_transcribe_rejects_non_pcm16_wav(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wav = tmp_path / "u8.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(b"\x80" * 800)
    _patch_transcribe(monkeypatch)
    result = runner.invoke(app, ["transcribe", str(wav), "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "16bit" in result.stdout


def test_transcribe_rejects_path_in_output_name_before_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wav = tmp_path / "meeting.wav"
    _write_wav(wav, seconds=1.0)
    loaded = _patch_transcribe(monkeypatch)
    result = runner.invoke(app, ["transcribe", str(wav), "-o", str(tmp_path), "-n", "a/b"])
    assert result.exit_code == 1
    assert "エラー" in result.stdout
    assert loaded == []


def test_transcribe_model_unavailable_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wav = tmp_path / "meeting.wav"
    _write_wav(wav, seconds=1.0)

    def _raise() -> None:
        raise ModelUnavailableError("モデル 'small' の取得に失敗しました: network down")

    _patch_transcribe(monkeypatch, _FakeSpec(on_ensure=_raise))
    result = runner.invoke(app, ["transcribe", str(wav), "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "network down" in result.stdout
    assert not (tmp_path / "meeting.txt").exists()


def test_run_file_transcription_stopped_returns_130(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wav = tmp_path / "meeting.wav"
    _write_wav(wav, seconds=5.0)
    _patch_transcribe(monkeypatch)
    stop = threading.Event()
    stop.set()
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120, highlight=False)

    code = run_file_transcription(
        WavFileSource(wav),
        console=console,
        input_path=wav,
        output_path=tmp_path / "meeting.txt",
        stop_event=stop,
    )
    assert code == 130
    assert "中断しました" in stream.getvalue()
    assert (tmp_path / "meeting.txt").read_text(encoding="utf-8").startswith("# loopscribe 入力:")


def test_config_shows_defaults_when_file_missing(config_file: Path) -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert str(config_file) in result.stdout
    assert "未作成" in result.stdout
    assert "default-model = small" in result.stdout
    assert "既定" in result.stdout
    assert not config_file.exists()


def test_config_sets_and_clears_default_model(config_file: Path) -> None:
    result = runner.invoke(app, ["config", "default-model=kotoba"])
    assert result.exit_code == 0
    assert read_raw(config_file) == {"default-model": "kotoba"}
    assert "更新しました" in result.stdout

    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "default-model = kotoba" in result.stdout
    assert "設定ファイル" in result.stdout

    result = runner.invoke(app, ["config", "default-model="])
    assert result.exit_code == 0
    assert read_raw(config_file) == {}
    assert "既定に戻しました" in result.stdout


@pytest.mark.parametrize("argument", ["default-model=large", "nope=x", "default-model"])
def test_config_rejects_invalid_arguments(config_file: Path, argument: str) -> None:
    result = runner.invoke(app, ["config", argument])
    assert result.exit_code == 1
    assert "エラー" in result.stdout
    assert not config_file.exists()


def test_config_does_not_overwrite_broken_file(config_file: Path) -> None:
    config_file.write_text("broken = [\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "default-model=kotoba"])
    assert result.exit_code == 1
    assert config_file.read_text(encoding="utf-8") == "broken = [\n"


def test_config_warns_on_broken_file_and_shows_defaults(config_file: Path) -> None:
    config_file.write_text('default-model = "kotoba\n', encoding="utf-8")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "警告" in result.stdout
    assert "default-model = small" in result.stdout


def test_start_uses_config_default_model(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    fake_wasapi = SimpleNamespace(open_loopback_source=lambda device: _EmptySource())
    monkeypatch.setattr("loopscribe.cli._load_wasapi", lambda: fake_wasapi)
    models: list[str] = []
    monkeypatch.setattr(
        "loopscribe.cli.run_transcription",
        lambda source, console, output_dir, model_name, output_path: models.append(model_name) or 0,
    )
    runner.invoke(app, ["config", "default-model=kotoba"])

    result = runner.invoke(app, ["start"])
    assert result.exit_code == 0
    assert models == ["kotoba"]

    # CLI オプションは設定ファイルより優先する
    result = runner.invoke(app, ["start", "--model", "small"])
    assert result.exit_code == 0
    assert models == ["kotoba", "small"]


def test_start_warns_and_uses_small_when_config_is_invalid(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    fake_wasapi = SimpleNamespace(open_loopback_source=lambda device: _EmptySource())
    monkeypatch.setattr("loopscribe.cli._load_wasapi", lambda: fake_wasapi)
    models: list[str] = []
    monkeypatch.setattr(
        "loopscribe.cli.run_transcription",
        lambda source, console, output_dir, model_name, output_path: models.append(model_name) or 0,
    )
    config_file.write_text('default-model = "large"\n', encoding="utf-8")

    result = runner.invoke(app, ["start"])
    assert result.exit_code == 0
    assert models == ["small"]
    assert "警告" in result.stdout


def test_transcribe_uses_config_default_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_file: Path
) -> None:
    wav = tmp_path / "meeting.wav"
    _write_wav(wav, seconds=1.0)
    loaded = _patch_transcribe(monkeypatch)
    config_file.write_text('default-model = "kotoba"\n', encoding="utf-8")

    result = runner.invoke(app, ["transcribe", str(wav), "-o", str(tmp_path)])
    assert result.exit_code == 0
    assert loaded == ["kotoba"]


def test_download_model_uses_config_default_model(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        KotobaModel, "download", lambda self: called.append(self.name) or Path("/cache/kotoba")
    )
    config_file.write_text('default-model = "kotoba"\n', encoding="utf-8")

    result = runner.invoke(app, ["download-model"])
    assert result.exit_code == 0
    assert called == ["kotoba"]

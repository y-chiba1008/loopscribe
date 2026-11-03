"""errors の単体テスト。"""

from __future__ import annotations

from loopscribe.errors import (
    DeviceNotFoundError,
    FatalStartupError,
    LoopscribeError,
    ModelUnavailableError,
)


def test_exception_hierarchy() -> None:
    assert issubclass(FatalStartupError, LoopscribeError)
    assert issubclass(DeviceNotFoundError, FatalStartupError)
    assert issubclass(ModelUnavailableError, FatalStartupError)


def test_exceptions_are_raisable() -> None:
    for exc in (
        DeviceNotFoundError("device missing"),
        ModelUnavailableError("model missing"),
    ):
        try:
            raise exc
        except LoopscribeError as caught:
            assert str(caught) == str(exc)

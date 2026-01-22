from __future__ import annotations

from types import FrameType
from typing import Any

import pytest
from _pytest.capture import CaptureFixture

from signal_handler import SignalHandler


def test_signal_handler_sets_flag_and_prints(capsys: CaptureFixture[str]) -> None:
    handler = SignalHandler()
    assert handler.shutdown_requested is False

    handler(15, None)
    captured = capsys.readouterr()

    assert "Shutdown signal 15 received" in captured.out
    assert handler.shutdown_requested is True

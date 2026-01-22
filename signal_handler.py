from __future__ import annotations

from types import FrameType


class SignalHandler:
    """A class to handle shutdown signals gracefully."""

    def __init__(self) -> None:
        self.shutdown_requested: bool = False

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        print(f"\n[Main] Shutdown signal {signum} received. Finishing current tasks and saving...")
        self.shutdown_requested = True

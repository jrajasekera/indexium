"""Type stubs for ffmpeg-python library."""

from typing import Any

class Error(Exception):
    """FFmpeg error."""

    stdout: bytes
    stderr: bytes

    def __init__(
        self, cmd: str, stdout: bytes | None = None, stderr: bytes | None = None
    ) -> None: ...

class Stream:
    """Represents an ffmpeg stream."""

    def output(
        self,
        filename: str,
        **kwargs: Any,
    ) -> Stream: ...
    def overwrite_output(self) -> Stream: ...
    def run(
        self,
        cmd: str = "ffmpeg",
        capture_stdout: bool = False,
        capture_stderr: bool = False,
        input: bytes | None = None,
        quiet: bool = False,
        overwrite_output: bool = False,
    ) -> tuple[bytes, bytes]: ...
    def run_async(
        self,
        cmd: str = "ffmpeg",
        pipe_stdin: bool = False,
        pipe_stdout: bool = False,
        pipe_stderr: bool = False,
        quiet: bool = False,
        overwrite_output: bool = False,
    ) -> Any: ...
    def filter(self, filter_name: str, *args: Any, **kwargs: Any) -> Stream: ...
    def split(self) -> tuple[Stream, Stream]: ...
    def concat(self, *streams: Stream, v: int = 1, a: int = 0) -> Stream: ...

def probe(
    filename: str,
    cmd: str = "ffprobe",
    timeout: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]: ...
def input(
    filename: str,
    **kwargs: Any,
) -> Stream: ...
def output(
    stream: Stream,
    filename: str,
    **kwargs: Any,
) -> Stream: ...
def run(
    stream: Stream,
    cmd: str = "ffmpeg",
    capture_stdout: bool = False,
    capture_stderr: bool = False,
    input: bytes | None = None,
    quiet: bool = False,
    overwrite_output: bool = False,
) -> tuple[bytes, bytes]: ...
def overwrite_output(stream: Stream) -> Stream: ...
def merge_outputs(*streams: Stream) -> Stream: ...
def filter(
    stream: Stream,
    filter_name: str,
    *args: Any,
    **kwargs: Any,
) -> Stream: ...
def concat(*streams: Stream, v: int = 1, a: int = 0) -> Stream: ...

"""Type stubs for pytesseract library."""

from typing import Any

from PIL import Image

__file__: str

class TesseractError(Exception):
    status: int
    message: str
    def __init__(self, status: int, message: str) -> None: ...

class TesseractNotFoundError(TesseractError): ...

def get_tesseract_version() -> str: ...
def image_to_string(
    image: Image.Image | str | Any,
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    output_type: int = ...,
    timeout: int = 0,
) -> str: ...
def image_to_boxes(
    image: Image.Image | str | Any,
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    output_type: int = ...,
    timeout: int = 0,
) -> str: ...
def image_to_data(
    image: Image.Image | str | Any,
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    output_type: int = ...,
    timeout: int = 0,
) -> str | dict[str, Any]: ...
def image_to_osd(
    image: Image.Image | str | Any,
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    output_type: int = ...,
    timeout: int = 0,
) -> str | dict[str, Any]: ...
def image_to_pdf_or_hocr(
    image: Image.Image | str | Any,
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    extension: str = "pdf",
    timeout: int = 0,
) -> bytes: ...
def run_and_get_output(
    image: Image.Image | str | Any,
    extension: str = "",
    lang: str | None = None,
    config: str = "",
    nice: int = 0,
    timeout: int = 0,
) -> str: ...

# Output type constants
class Output:
    BYTES: int
    DATAFRAME: int
    DICT: int
    STRING: int

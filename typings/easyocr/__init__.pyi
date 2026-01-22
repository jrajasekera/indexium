"""Type stubs for easyocr library."""

from typing import Any, Sequence

import numpy.typing as npt

__file__: str

# OCR result: (bounding_box, text, confidence)
# bounding_box is list of 4 corner points, each as [x, y]
OCRResult = tuple[list[list[int]], str, float]

class Reader:
    def __init__(
        self,
        lang_list: list[str],
        gpu: bool = False,
        model_storage_directory: str | None = None,
        user_network_directory: str | None = None,
        recog_network: str = "standard",
        download_enabled: bool = True,
        detector: bool = True,
        recognizer: bool = True,
        verbose: bool = True,
        quantize: bool = True,
        cudnn_benchmark: bool = False,
    ) -> None: ...
    def readtext(
        self,
        image: str | bytes | npt.NDArray[Any],
        decoder: str = "greedy",
        beamWidth: int = 5,
        batch_size: int = 1,
        workers: int = 0,
        allowlist: str | None = None,
        blocklist: str | None = None,
        detail: int = 1,
        rotation_info: Sequence[int] | None = None,
        paragraph: bool = False,
        min_size: int = 20,
        contrast_ths: float = 0.1,
        adjust_contrast: float = 0.5,
        filter_ths: float = 0.003,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        y_ths: float = 0.5,
        x_ths: float = 1.0,
        add_margin: float = 0.1,
        output_format: str = "standard",
    ) -> list[OCRResult]: ...
    def detect(
        self,
        image: str | bytes | npt.NDArray[Any],
        min_size: int = 20,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        add_margin: float = 0.1,
        reformat: bool = True,
        optimal_num_chars: int | None = None,
    ) -> tuple[list[list[list[int]]], list[list[list[int]]]]: ...
    def recognize(
        self,
        image: str | bytes | npt.NDArray[Any],
        horizontal_list: list[list[int]] | None = None,
        free_list: list[list[list[int]]] | None = None,
        decoder: str = "greedy",
        beamWidth: int = 5,
        batch_size: int = 1,
        workers: int = 0,
        allowlist: str | None = None,
        blocklist: str | None = None,
        detail: int = 1,
        rotation_info: Sequence[int] | None = None,
        paragraph: bool = False,
        contrast_ths: float = 0.1,
        adjust_contrast: float = 0.5,
        filter_ths: float = 0.003,
        y_ths: float = 0.5,
        x_ths: float = 1.0,
        reformat: bool = True,
        output_format: str = "standard",
    ) -> list[OCRResult]: ...

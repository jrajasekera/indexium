"""Type stubs for face_recognition library."""

from typing import Literal

import numpy as np
import numpy.typing as npt

# Type aliases
FaceLocation = tuple[int, int, int, int]  # (top, right, bottom, left)
FaceEncoding = npt.NDArray[np.float64]  # 128-dimensional face encoding

def face_locations(
    img: npt.NDArray[np.uint8],
    number_of_times_to_upsample: int = 1,
    model: str = "hog",
) -> list[FaceLocation]: ...
def face_encodings(
    face_image: npt.NDArray[np.uint8],
    known_face_locations: list[FaceLocation] | None = None,
    num_jitters: int = 1,
    model: Literal["small", "large"] = "small",
) -> list[FaceEncoding]: ...
def face_distance(
    face_encodings: list[FaceEncoding],
    face_to_compare: FaceEncoding,
) -> npt.NDArray[np.float64]: ...
def compare_faces(
    known_face_encodings: list[FaceEncoding],
    face_encoding_to_check: FaceEncoding,
    tolerance: float = 0.6,
) -> list[bool]: ...
def load_image_file(
    file: str,
    mode: Literal["RGB", "L"] = "RGB",
) -> npt.NDArray[np.uint8]: ...

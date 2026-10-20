"""Unicode-safe filesystem helpers used by the SfM pipeline.

OpenCV's filename based ``imread``/``imwrite`` APIs use narrow strings in
some Windows builds.  Reading and writing the encoded bytes through Python's
filesystem APIs keeps the path handling in Python's Unicode-aware layer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import cv2
import numpy as np


PathLike = Union[str, os.PathLike]


def imread(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image without passing the filename to OpenCV.

    The return value intentionally matches ``cv2.imread``: unreadable or empty
    files return ``None`` instead of raising an ``OSError``.
    """
    try:
        with open(path, "rb") as stream:
            encoded = np.frombuffer(stream.read(), dtype=np.uint8)
    except (OSError, TypeError, ValueError):
        return None

    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def imwrite(
    path: PathLike,
    image: np.ndarray,
    params: Optional[Sequence[int]] = None,
) -> bool:
    """Write an image without passing the filename to OpenCV.

    Encoding is still performed by OpenCV, while Python writes the resulting
    bytes to the Unicode path.  Parent directories are deliberately not
    created so the behavior stays close to ``cv2.imwrite``.
    """
    extension = Path(os.fspath(path)).suffix
    if not extension:
        raise ValueError(f"Image output path has no file extension: {path}")

    encode_params = list(params) if params is not None else []
    success, encoded = cv2.imencode(extension, image, encode_params)
    if not success:
        return False

    try:
        with open(path, "wb") as stream:
            stream.write(encoded.tobytes())
    except (OSError, TypeError, ValueError):
        return False
    return True


def validate_colmap_ascii_image_names(
    image_paths: Iterable[PathLike],
    image_root: PathLike,
) -> None:
    """Reject non-ASCII image names before they enter native COLMAP.

    The absolute root may contain arbitrary Unicode.  Only names relative to
    ``image_root`` are restricted because the bundled Windows COLMAP stores
    such names using the active ANSI code page rather than stable UTF-8.
    """
    root = os.fspath(image_root)
    invalid = []
    for image_path in image_paths:
        relative = Path(os.path.relpath(os.fspath(image_path), root)).as_posix()
        if not relative.isascii():
            invalid.append(relative)
            if len(invalid) >= 10:
                break

    if invalid:
        examples = ", ".join(repr(name) for name in invalid)
        raise ValueError(
            "COLMAP image names relative to the image root must contain only "
            "ASCII characters on Windows. The image root itself may contain "
            f"Chinese characters. Invalid name(s): {examples}"
        )

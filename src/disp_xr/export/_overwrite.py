"""One rule for existing outputs: refuse them unless ``overwrite=True``.

API review question 3 (decided 2026-10-11): every writer takes
``overwrite: bool = False``. ``to_icechunk`` keeps ``mode``, which also covers
appending to a repository.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Optional

ZarrMode = Literal["w", "w-"]


def zarr_mode(mode: Optional[ZarrMode], overwrite: bool, writer: str) -> ZarrMode:
    """Return the zarr write mode for ``overwrite``; ``mode`` is deprecated."""
    if mode is None:
        return "w" if overwrite else "w-"
    warnings.warn(
        f"{writer}(mode=...) is deprecated; use overwrite=True to replace an "
        "existing store (removed in 1.0)",
        DeprecationWarning,
        stacklevel=3,
    )
    return mode


def refuse_existing(paths: Iterable[Path], overwrite: bool) -> None:
    """Raise ``FileExistsError`` for the first of ``paths`` that exists."""
    if overwrite:
        return
    for path in paths:
        if path.exists():
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

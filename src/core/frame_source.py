"""Video frame source protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FrameSource(Protocol):
    """Threaded capture with latest-frame buffer."""

    @property
    def source_type(self) -> str:
        ...

    @property
    def source_label(self) -> str:
        ...

    @property
    def is_running(self) -> bool:
        ...

    @property
    def error(self) -> str | None:
        ...

    def start(self) -> bool:
        ...

    def stop(self) -> None:
        ...

    def get_latest(self) -> tuple[np.ndarray | None, int]:
        ...

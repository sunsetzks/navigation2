from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Path:
    x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    yaws: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def reset(self, size: int) -> None:
        self.x = np.zeros(size, dtype=np.float32)
        self.y = np.zeros(size, dtype=np.float32)
        self.yaws = np.zeros(size, dtype=np.float32)

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trajectories:
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    yaws: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    def reset(self, batch_size: int, time_steps: int) -> None:
        shape = (batch_size, time_steps)
        self.x = np.zeros(shape, dtype=np.float32)
        self.y = np.zeros(shape, dtype=np.float32)
        self.yaws = np.zeros(shape, dtype=np.float32)

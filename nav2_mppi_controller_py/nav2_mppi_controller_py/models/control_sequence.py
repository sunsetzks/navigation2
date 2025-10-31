from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Control:
    """Single control sample."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass
class ControlSequence:
    """Control sequence along the horizon."""

    vx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    vy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    wz: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def reset(self, time_steps: int) -> None:
        self.vx = np.zeros(time_steps, dtype=np.float32)
        self.vy = np.zeros(time_steps, dtype=np.float32)
        self.wz = np.zeros(time_steps, dtype=np.float32)

    def ensure_shape(self, time_steps: int) -> None:
        """Ensure the control sequence arrays have a given temporal length."""
        if self.vx.shape != (time_steps,):
            self.reset(time_steps)

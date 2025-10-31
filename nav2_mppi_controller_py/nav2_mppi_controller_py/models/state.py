from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..messages import Pose, Twist


@dataclass
class State:
    vx: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    vy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    wz: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    cvx: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    cvy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    cwz: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    pose: Pose = field(default_factory=Pose)
    speed: Twist = field(default_factory=Twist)

    def reset(self, batch_size: int, time_steps: int) -> None:
        shape = (batch_size, time_steps)
        self.vx = np.zeros(shape, dtype=np.float32)
        self.vy = np.zeros(shape, dtype=np.float32)
        self.wz = np.zeros(shape, dtype=np.float32)
        self.cvx = np.zeros(shape, dtype=np.float32)
        self.cvy = np.zeros(shape, dtype=np.float32)
        self.cwz = np.zeros(shape, dtype=np.float32)

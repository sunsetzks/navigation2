from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlConstraints:
    vx_max: float = 0.0
    vx_min: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass
class SamplingStd:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

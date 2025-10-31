from __future__ import annotations

from dataclasses import dataclass, field

from .constraints import ControlConstraints, SamplingStd


@dataclass
class OptimizerSettings:
    base_constraints: ControlConstraints = field(default_factory=ControlConstraints)
    constraints: ControlConstraints = field(default_factory=ControlConstraints)
    sampling_std: SamplingStd = field(default_factory=SamplingStd)
    model_dt: float = 0.0
    temperature: float = 0.0
    gamma: float = 0.0
    batch_size: int = 0
    time_steps: int = 0
    iteration_count: int = 0
    shift_control_sequence: bool = False
    retry_attempt_limit: int = 0

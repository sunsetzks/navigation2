from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from .base import CriticFunction


class VelocityDeadbandCritic(CriticFunction):
    """Penalizes velocities below deadband thresholds."""

    def __init__(
        self,
        weight: float = 35.0,
        power: int = 1,
        deadband_velocities: list[float] | tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__(weight=weight)
        self.power = power

        if deadband_velocities is None:
            deadband_velocities = [0.0, 0.0, 0.0]
        self.deadband_velocities = np.array(deadband_velocities, dtype=np.float32)

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        state = data.state
        vx = state.vx
        wz = state.wz

        # Check if holonomic
        if data.motion_model and data.motion_model.is_holonomic():
            vy = state.vy
            deadband_penalty = (
                np.maximum(np.abs(self.deadband_velocities[0]) - np.abs(vx), 0)
                + np.maximum(np.abs(self.deadband_velocities[1]) - np.abs(vy), 0)
                + np.maximum(np.abs(self.deadband_velocities[2]) - np.abs(wz), 0)
            )
        else:
            deadband_penalty = (
                np.maximum(np.abs(self.deadband_velocities[0]) - np.abs(vx), 0)
                + np.maximum(np.abs(self.deadband_velocities[2]) - np.abs(wz), 0)
            )

        # Integrate over time and apply weight and power
        cost_sum = np.sum(deadband_penalty * data.model_dt, axis=1)
        if self.power > 1:
            costs += np.power(cost_sum * self.weight, self.power)
        else:
            costs += cost_sum * self.weight


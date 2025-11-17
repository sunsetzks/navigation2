from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from ..tools.utils import within_position_goal_tolerance
from .base import CriticFunction


class TwirlingCritic(CriticFunction):
    """Penalizes high angular velocities to reduce twirling motion."""

    def __init__(self, weight: float = 10.0, power: int = 1) -> None:
        super().__init__(weight=weight)
        self.power = power

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        # Don't apply when near goal
        if within_position_goal_tolerance(data.goal_checker, data.state.pose, data.goal):
            return

        # Penalize absolute angular velocity
        wz_abs = np.abs(data.state.wz)

        # Mean over time and apply weight and power
        wz_mean = np.mean(wz_abs, axis=1)
        if self.power > 1:
            costs += np.power(wz_mean * self.weight, self.power)
        else:
            costs += wz_mean * self.weight


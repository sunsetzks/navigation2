from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from ..tools.utils import within_position_goal_tolerance_scalar
from .base import CriticFunction


class PreferForwardCritic(CriticFunction):
    """Penalizes backward motion to prefer forward movement."""

    def __init__(self, weight: float = 5.0, power: int = 1, threshold_to_consider: float = 0.5) -> None:
        super().__init__(weight=weight)
        self.power = power
        self.threshold_to_consider = threshold_to_consider

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        # Don't apply close to goal, let the goal critics take over
        if within_position_goal_tolerance_scalar(
            self.threshold_to_consider, data.state.pose, data.goal
        ):
            return

        # Penalize backward motion (negative vx)
        backward_motion = np.maximum(-data.state.vx, 0)

        # Integrate over time and apply weight and power
        cost_sum = np.sum(backward_motion * data.model_dt, axis=1)
        if self.power > 1:
            costs += np.power(cost_sum * self.weight, self.power)
        else:
            costs += cost_sum * self.weight


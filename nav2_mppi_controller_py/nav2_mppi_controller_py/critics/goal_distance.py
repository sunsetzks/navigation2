from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from .base import CriticFunction


class GoalDistanceCritic(CriticFunction):
    """Encourages trajectories to terminate near the goal position."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight=weight)

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        goal = data.goal
        trajectories = data.trajectories
        final_x = trajectories.x[:, -1]
        final_y = trajectories.y[:, -1]

        dx = final_x - goal.position.x
        dy = final_y - goal.position.y
        dist_sq = dx * dx + dy * dy
        costs += self.weight * dist_sq

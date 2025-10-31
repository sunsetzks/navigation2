from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from ..messages import quaternion_from_yaw, yaw_from_quaternion
from ..tools.utils import normalize_angles
from .base import CriticFunction


class GoalHeadingCritic(CriticFunction):
    """Penalises heading error at the end of the trajectory."""

    def __init__(self, weight: float = 1.0, goal_yaw: float | None = None) -> None:
        super().__init__(weight=weight)
        self.goal_yaw = goal_yaw

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        goal_yaw = self.goal_yaw
        if goal_yaw is None:
            goal_yaw = yaw_from_quaternion(data.goal.orientation)

        trajectories = data.trajectories
        final_yaw = trajectories.yaws[:, -1]
        yaw_error = normalize_angles(final_yaw - goal_yaw)
        costs += self.weight * (yaw_error * yaw_error)

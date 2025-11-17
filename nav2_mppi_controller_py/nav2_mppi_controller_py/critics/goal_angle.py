from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from ..tools.utils import (
    shortest_angular_distance,
    within_position_goal_tolerance_scalar,
)
from .base import CriticFunction


class GoalHeadingCritic(CriticFunction):
    """Penalises heading error at the end of the trajectory."""

    def __init__(
        self,
        weight: float = 3.0,
        power: int = 1,
        threshold_to_consider: float = 0.5,
    ) -> None:
        super().__init__(weight=weight)
        self.power = power
        self.threshold_to_consider = threshold_to_consider

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        # Only apply when within threshold distance to goal
        if not within_position_goal_tolerance_scalar(
            self.threshold_to_consider, data.state.pose, data.goal
        ):
            return

        # Get goal yaw from path's last point (consistent with C++ implementation)
        if data.path.yaws.shape[0] == 0:
            return
        goal_idx = data.path.yaws.shape[0] - 1
        goal_yaw = data.path.yaws[goal_idx]

        # Calculate angular distance for all trajectory points
        yaw_errors = np.abs(
            shortest_angular_distance(data.trajectories.yaws, goal_yaw)
        )

        # Mean angle error over time steps for each trajectory
        yaw_error_mean = np.mean(yaw_errors, axis=1)

        # Apply weight and power

        costs += np.power(yaw_error_mean * self.weight, self.power)
    

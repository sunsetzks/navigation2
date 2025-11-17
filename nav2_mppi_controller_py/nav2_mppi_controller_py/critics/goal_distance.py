from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from ..tools.utils import within_position_goal_tolerance_scalar
from .base import CriticFunction


class GoalDistanceCritic(CriticFunction):
    """Encourages trajectories to terminate near the goal position.
    
    Args:
        weight: Weight for the cost function
        power: Power to apply to the cost (default: 1)
        threshold_to_consider: Distance threshold to consider this critic (default: 1.4)
        use_all_points: If True, uses mean distance over all trajectory points.
                       If False, uses only the last point of each trajectory.
                       Default: True (MPPI standard behavior)
    """

    def __init__(
        self,
        weight: float = 5.0,
        power: int = 1,
        threshold_to_consider: float = 1.4,
        use_all_points: bool = True,
    ) -> None:
        super().__init__(weight=weight)
        self.power = power
        self.threshold_to_consider = threshold_to_consider
        self.use_all_points = use_all_points

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        # Only apply when within threshold distance to goal
        if not within_position_goal_tolerance_scalar(
            self.threshold_to_consider, data.state.pose, data.goal
        ):
            return

        goal_x = data.goal.position.x
        goal_y = data.goal.position.y

        # Calculate distance from trajectory points to goal
        dx = data.trajectories.x - goal_x
        dy = data.trajectories.y - goal_y
        dists = np.sqrt(dx * dx + dy * dy)

        # Choose evaluation method based on use_all_points flag
        if self.use_all_points:
            # Mean distance over time steps for each trajectory
            dist_mean = np.mean(dists, axis=1)
        else:
            # Use only the last point of each trajectory
            dist_mean = dists[:, -1]

        # Apply weight and power    
        costs += np.power(dist_mean * self.weight, self.power)

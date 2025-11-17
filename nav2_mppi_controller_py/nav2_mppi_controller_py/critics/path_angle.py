from __future__ import annotations

import math

import numpy as np

from ..critic_data import CriticData
from ..tools.utils import (
    normalize_angles,
    pose_point_angle,
    set_path_furthest_point_if_not_set,
    shortest_angular_distance,
    within_position_goal_tolerance_scalar,
)
from .base import CriticFunction


class PathAngleCritic(CriticFunction):
    """Penalizes angle mismatch between trajectory heading and path direction."""

    def __init__(
        self,
        weight: float = 2.0,
        power: int = 1,
        offset_from_furthest: int = 4,
        threshold_to_consider: float = 0.5,
        max_angle_to_furthest: float = 1.2,
        forward_preference: bool = True,
        vx_min: float = -0.35,
    ) -> None:
        super().__init__(weight=weight)
        self.power = power
        self.offset_from_furthest = offset_from_furthest
        self.threshold_to_consider = threshold_to_consider
        self.max_angle_to_furthest = max_angle_to_furthest

        # Determine if reversing is allowed
        if abs(vx_min) < 1e-6:
            self.reversing_allowed = False
        elif vx_min < 0.0:
            self.reversing_allowed = True
        else:
            self.reversing_allowed = False

        if not self.reversing_allowed:
            self.forward_preference = True
        else:
            self.forward_preference = forward_preference

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        # Don't apply close to goal
        if within_position_goal_tolerance_scalar(
            self.threshold_to_consider, data.state.pose, data.goal
        ):
            return

        # Find furthest reached point if not set
        set_path_furthest_point_if_not_set(data)

        if data.furthest_reached_path_point is None:
            return

        # Calculate offset index
        offset_idx = min(
            data.furthest_reached_path_point + self.offset_from_furthest,
            data.path.x.shape[0] - 1,
        )

        goal_x = data.path.x[offset_idx]
        goal_y = data.path.y[offset_idx]

        # Check if angle is already acceptable
        if (
            pose_point_angle(
                data.state.pose, goal_x, goal_y, self.forward_preference
            )
            < self.max_angle_to_furthest
        ):
            return

        # Calculate yaw from trajectories to goal point
        yaws_between_points = np.arctan2(
            goal_y - data.trajectories.y, goal_x - data.trajectories.x
        )

        # Calculate angle differences
        yaws = np.abs(
            shortest_angular_distance(data.trajectories.yaws, yaws_between_points)
        )

        # Handle reversing case
        if self.reversing_allowed and not self.forward_preference:
            # Check if angle > 90 degrees, if so consider reverse direction
            yaws_between_points_corrected = np.where(
                yaws < math.pi / 2,
                yaws_between_points,
                normalize_angles(yaws_between_points + math.pi),
            )
            corrected_yaws = np.abs(
                shortest_angular_distance(
                    data.trajectories.yaws, yaws_between_points_corrected
                )
            )
            yaw_mean = np.mean(corrected_yaws, axis=1)
        else:
            yaw_mean = np.mean(yaws, axis=1)

        # Apply weight and power
        if self.power > 1:
            costs += np.power(yaw_mean * self.weight, self.power)
        else:
            costs += yaw_mean * self.weight


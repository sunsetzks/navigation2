from __future__ import annotations

import math

import numpy as np

from ..critic_data import CriticData
from ..tools.utils import (
    find_path_trajectory_initial_point,
    normalize_angles,
    set_path_furthest_point_if_not_set,
    shortest_angular_distance,
    within_position_goal_tolerance_scalar,
)
from .base import CriticFunction


class PathAlignLegacyCritic(CriticFunction):
    """Aligns trajectories to the path using closest point search (legacy method)."""

    def __init__(
        self,
        weight: float = 10.0,
        power: int = 1,
        max_path_occupancy_ratio: float = 0.07,
        offset_from_furthest: int = 20,
        trajectory_point_step: int = 4,
        threshold_to_consider: float = 0.5,
        use_path_orientations: bool = False,
    ) -> None:
        super().__init__(weight=weight)
        self.power = power
        self.max_path_occupancy_ratio = max_path_occupancy_ratio
        self.offset_from_furthest = offset_from_furthest
        self.trajectory_point_step = trajectory_point_step
        self.threshold_to_consider = threshold_to_consider
        self.use_path_orientations = use_path_orientations

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        # Don't apply close to goal, let the goal critics take over
        if within_position_goal_tolerance_scalar(
            self.threshold_to_consider, data.state.pose, data.goal
        ):
            return

        # Don't apply when first getting bearing w.r.t. the path
        set_path_furthest_point_if_not_set(data)

        if data.furthest_reached_path_point is None:
            return

        if data.furthest_reached_path_point < self.offset_from_furthest:
            return

        # Don't apply when dynamic obstacles are blocking significant proportions of the local path
        # Note: path_pts_valid might be None if costmap is not available
        closest_initial_path_point = find_path_trajectory_initial_point(data)
        invalid_ctr = 0
        range_size = float(data.furthest_reached_path_point - closest_initial_path_point)

        if data.path_pts_valid is not None:
            for i in range(closest_initial_path_point, data.furthest_reached_path_point):
                if not data.path_pts_valid[i]:
                    invalid_ctr += 1
                    if (
                        float(invalid_ctr) / range_size > self.max_path_occupancy_ratio
                        and invalid_ctr > 2
                    ):
                        return

        T_x = data.trajectories.x
        T_y = data.trajectories.y
        T_yaw = data.trajectories.yaws

        # Path points (exclude last one as per C++ code)
        P_x = data.path.x[:-1]  # path points
        P_y = data.path.y[:-1]  # path points
        P_yaw = data.path.yaws[:-1]  # path points

        batch_size = T_x.shape[0]
        time_steps = T_x.shape[1]
        traj_pts_eval = int(math.floor(time_steps / self.trajectory_point_step))
        path_segments_count = data.path.x.shape[0] - 1

        if path_segments_count < 1:
            return

        cost = np.zeros(batch_size, dtype=np.float32)

        # Process each trajectory
        for t in range(batch_size):
            summed_dist = 0.0

            for p in range(self.trajectory_point_step, time_steps, self.trajectory_point_step):
                min_dist_sq = float("inf")
                min_s = 0

                # Find closest path segment to the trajectory point
                for s in range(path_segments_count - 1):
                    dx = P_x[s] - T_x[t, p]
                    dy = P_y[s] - T_y[t, p]

                    if self.use_path_orientations:
                        dyaw = shortest_angular_distance(
                            np.array([P_yaw[s]]), np.array([T_yaw[t, p]])
                        )[0]
                        dist_sq = dx * dx + dy * dy + dyaw * dyaw
                    else:
                        dist_sq = dx * dx + dy * dy

                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        min_s = s

                # The nearest path point to align to needs to be not in collision
                # If path_pts_valid is None, treat all points as valid (costmap not available)
                if data.path_pts_valid is None:
                    is_valid = min_s != 0
                elif min_s < len(data.path_pts_valid):
                    is_valid = min_s != 0 and data.path_pts_valid[min_s]
                else:
                    is_valid = min_s != 0

                if is_valid:
                    summed_dist += math.sqrt(min_dist_sq)

            cost[t] = summed_dist / traj_pts_eval

        # Apply weight and power
        if self.power > 1:
            costs += np.power(cost * self.weight, self.power)
        else:
            costs += cost * self.weight


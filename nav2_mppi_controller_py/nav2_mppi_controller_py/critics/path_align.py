from __future__ import annotations

import math

import numpy as np

from ..critic_data import CriticData
from ..tools.utils import (
    find_closest_path_pt,
    find_path_trajectory_initial_point,
    normalize_angles,
    set_path_furthest_point_if_not_set,
    shortest_angular_distance,
    within_position_goal_tolerance_scalar,
)
from .base import CriticFunction


class PathAlignCritic(CriticFunction):
    """Aligns trajectories to the path using integrated distance along path."""

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

        path_segments_count = data.furthest_reached_path_point  # up to furthest only
        if path_segments_count < self.offset_from_furthest:
            return

        # Don't apply when dynamic obstacles are blocking significant proportions of the local path
        # Note: path_pts_valid might be None if costmap is not available
        # In that case, we treat all points as valid
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

        # Path points (exclude last one as per C++ code)
        P_x = data.path.x[:-1]  # path points
        P_y = data.path.y[:-1]  # path points
        P_yaw = data.path.yaws[:-1]  # path points

        batch_size = data.trajectories.x.shape[0]
        time_steps = data.trajectories.x.shape[1]
        cost = np.zeros(batch_size, dtype=np.float32)

        # Find integrated distance in the path
        path_integrated_distances = np.zeros(path_segments_count, dtype=np.float32)
        for i in range(1, path_segments_count):
            dx = P_x[i] - P_x[i - 1]
            dy = P_y[i] - P_y[i - 1]
            curr_dist = math.sqrt(dx * dx + dy * dy)
            path_integrated_distances[i] = path_integrated_distances[i - 1] + curr_dist

        # Process each trajectory
        for t in range(batch_size):
            traj_integrated_distance = 0.0
            summed_path_dist = 0.0
            num_samples = 0.0
            path_pt = 0

            T_x = data.trajectories.x[t, :]
            T_y = data.trajectories.y[t, :]

            for p in range(self.trajectory_point_step, time_steps, self.trajectory_point_step):
                Tx = T_x[p]
                Ty = T_y[p]
                dx = Tx - T_x[p - self.trajectory_point_step]
                dy = Ty - T_y[p - self.trajectory_point_step]
                traj_integrated_distance += math.sqrt(dx * dx + dy * dy)

                path_pt = find_closest_path_pt(path_integrated_distances, traj_integrated_distance, path_pt)

                # The nearest path point to align to needs to be not in collision
                # If path_pts_valid is None, treat all points as valid (costmap not available)
                if data.path_pts_valid is None:
                    is_valid = True
                elif path_pt < len(data.path_pts_valid):
                    is_valid = data.path_pts_valid[path_pt]
                else:
                    is_valid = True

                if is_valid:
                    dx = P_x[path_pt] - Tx
                    dy = P_y[path_pt] - Ty
                    num_samples += 1.0
                    if self.use_path_orientations:
                        T_yaw = data.trajectories.yaws[t, :]
                        dyaw = shortest_angular_distance(
                            np.array([P_yaw[path_pt]]), np.array([T_yaw[p]])
                        )[0]
                        summed_path_dist += math.sqrt(dx * dx + dy * dy + dyaw * dyaw)
                    else:
                        summed_path_dist += math.sqrt(dx * dx + dy * dy)

            if num_samples > 0:
                cost[t] = summed_path_dist / num_samples
            else:
                cost[t] = 0.0

        # Apply weight and power
        if self.power > 1:
            costs += np.power(cost * self.weight, self.power)
        else:
            costs += cost * self.weight


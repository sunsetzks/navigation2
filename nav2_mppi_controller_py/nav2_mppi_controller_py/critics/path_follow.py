from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from .base import CriticFunction


class PathFollowCritic(CriticFunction):
    """Penalises deviation from the reference path along the horizon."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight=weight)

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        path = data.path
        trajectories = data.trajectories

        if path.x.shape[0] == 0:
            return

        target_count = min(trajectories.x.shape[1], path.x.shape[0])
        path_x = path.x[:target_count]
        path_y = path.y[:target_count]

        traj_x = trajectories.x[:, :target_count]
        traj_y = trajectories.y[:, :target_count]

        dx = traj_x - path_x
        dy = traj_y - path_y
        dist_sq = dx * dx + dy * dy
        costs += self.weight * np.mean(dist_sq, axis=1)

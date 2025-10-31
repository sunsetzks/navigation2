from __future__ import annotations

from .messages import Path, Pose, Twist
from .optimizer import Optimizer


class MPPIController:
    """High-level wrapper around the MPPI optimiser."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self._plan = Path()
        self._goal = Pose()

    def set_plan(self, plan: Path) -> None:
        self._plan = plan
        if plan.poses:
            self._goal = plan.poses[-1].pose

    def reset(self) -> None:
        self.optimizer.reset()

    def set_speed_limit(self, speed_limit: float, percentage: bool = True) -> None:
        self.optimizer.set_speed_limit(speed_limit, percentage)

    def compute_velocity_command(self, pose: Pose, speed: Twist):
        return self.optimizer.eval_control(pose, speed, self._plan, self._goal)

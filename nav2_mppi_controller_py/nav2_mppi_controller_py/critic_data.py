from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .models import Path, State, Trajectories
from .motion_models import MotionModel
from .messages import Pose, Twist


@dataclass
class CriticData:
    state: State
    trajectories: Trajectories
    path: Path
    goal: Pose
    costs: np.ndarray
    model_dt: float

    fail_flag: bool = False
    goal_checker: Optional["GoalChecker"] = None
    motion_model: Optional[MotionModel] = None
    path_pts_valid: Optional[List[bool]] = None
    furthest_reached_path_point: Optional[int] = None


@dataclass
class GoalChecker:
    """Minimal goal checker replacement for standalone use."""

    position_tolerance: float = 0.05
    velocity_tolerance: float = 0.05

    def get_tolerances(self, pose_tolerance: Pose, velocity_tolerance: Twist) -> None:
        pose_tolerance.position.x = self.position_tolerance
        pose_tolerance.position.y = self.position_tolerance
        pose_tolerance.position.z = 0.0
        velocity_tolerance.linear.x = self.velocity_tolerance
        velocity_tolerance.linear.y = self.velocity_tolerance
        velocity_tolerance.angular.z = self.velocity_tolerance

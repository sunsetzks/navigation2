from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from ..motion_models import AckermannMotionModel
from .base import CriticFunction


class ConstraintCritic(CriticFunction):
    """Enforces feasible velocity constraints."""

    def __init__(
        self,
        weight: float = 4.0,
        power: int = 1,
        vx_max: float = 0.5,
        vy_max: float = 0.0,
        vx_min: float = -0.35,
    ) -> None:
        super().__init__(weight=weight)
        self.power = power

        # Calculate velocity constraints
        min_sgn = 1.0 if vx_min > 0.0 else -1.0
        self.max_vel = np.sqrt(vx_max * vx_max + vy_max * vy_max)
        self.min_vel = min_sgn * np.sqrt(vx_min * vx_min + vy_max * vy_max)

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        state = data.state

        # Calculate total velocity with sign
        sgn = np.where(state.vx > 0.0, 1.0, -1.0)
        vel_total = sgn * np.sqrt(state.vx * state.vx + state.vy * state.vy)

        # Check for violations
        out_of_max_bounds_motion = np.maximum(vel_total - self.max_vel, 0)
        out_of_min_bounds_motion = np.maximum(self.min_vel - vel_total, 0)

        # Check if Ackermann model and add turning radius constraint
        if isinstance(data.motion_model, AckermannMotionModel):
            vx = state.vx
            wz = state.wz
            min_turning_r = data.motion_model.get_min_turning_radius()

            # Avoid divide by zero
            wz_abs = np.abs(wz)
            wz_abs_safe = np.where(wz_abs > 1e-6, wz_abs, 1e-6)
            out_of_turning_rad_motion = np.maximum(
                min_turning_r - (np.abs(vx) / wz_abs_safe), 0.0
            )

            constraint_violations = (
                out_of_max_bounds_motion + out_of_min_bounds_motion + out_of_turning_rad_motion
            )
        else:
            constraint_violations = out_of_max_bounds_motion + out_of_min_bounds_motion

        # Integrate over time and apply weight and power
        cost_sum = np.sum(constraint_violations * data.model_dt, axis=1)
        if self.power > 1:
            costs += np.power(cost_sum * self.weight, self.power)
        else:
            costs += cost_sum * self.weight

    def get_max_vel_constraint(self) -> float:
        """Get maximum velocity constraint."""
        return float(self.max_vel)

    def get_min_vel_constraint(self) -> float:
        """Get minimum velocity constraint."""
        return float(self.min_vel)


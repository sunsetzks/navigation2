from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from .base import CriticFunction


class JerkCritic(CriticFunction):
    """Penalizes the rate of change of acceleration (jerk).
    
    This critic penalizes the second derivative of control inputs, which represents
    jerk (rate of change of acceleration). It encourages even smoother control
    sequences by penalizing rapid changes in acceleration.
    
    Args:
        weight: Weight for the cost function (default: 1.0)
        power: Power to apply to the cost (default: 1)
    """

    def __init__(
        self,
        weight: float = 1.0,
        power: int = 1,
    ) -> None:
        super().__init__(weight=weight)
        self.power = power

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        state = data.state
        dt = data.model_dt
        
        # Calculate acceleration (first derivative of control inputs)
        # acceleration = (control[t] - control[t-1]) / dt
        accel_vx = (state.cvx[:, 1:] - state.cvx[:, :-1]) / dt
        accel_vy = (state.cvy[:, 1:] - state.cvy[:, :-1]) / dt
        accel_wz = (state.cwz[:, 1:] - state.cwz[:, :-1]) / dt
        
        # Calculate jerk (second derivative of control inputs)
        # jerk = (acceleration[t] - acceleration[t-1]) / dt
        # Need at least 2 time steps for acceleration, so jerk needs at least 3 time steps
        if accel_vx.shape[1] < 2:
            return
        
        jerk_vx = (accel_vx[:, 1:] - accel_vx[:, :-1]) / dt
        jerk_vy = (accel_vy[:, 1:] - accel_vy[:, :-1]) / dt
        jerk_wz = (accel_wz[:, 1:] - accel_wz[:, :-1]) / dt
        
        # Calculate squared jerk magnitude
        jerk_squared = jerk_vx * jerk_vx + jerk_vy * jerk_vy + jerk_wz * jerk_wz
        
        # Mean jerk over time steps for each trajectory
        jerk_mean = np.mean(jerk_squared, axis=1)
        
        # Apply weight and power
        if self.power > 1:
            costs += np.power(jerk_mean * self.weight, self.power)
        else:
            costs += jerk_mean * self.weight


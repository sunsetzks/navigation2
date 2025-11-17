from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from .base import CriticFunction


class AccelerationRateCritic(CriticFunction):
    """Penalizes the rate of change of control inputs (acceleration).
    
    This critic penalizes the first derivative of control inputs (cvx, cvy, cwz),
    which represents acceleration. It encourages smoother control sequences by
    penalizing large changes between consecutive time steps.
    
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
        # Use slicing to compute differences between consecutive time steps
        accel_vx = (state.cvx[:, 1:] - state.cvx[:, :-1]) / dt
        accel_vy = (state.cvy[:, 1:] - state.cvy[:, :-1]) / dt
        accel_wz = (state.cwz[:, 1:] - state.cwz[:, :-1]) / dt
        
        # Calculate squared acceleration magnitude
        accel_squared = accel_vx * accel_vx + accel_vy * accel_vy + accel_wz * accel_wz
        
        # Mean acceleration over time steps for each trajectory
        accel_mean = np.mean(accel_squared, axis=1)
        
        # Apply weight and power
        if self.power > 1:
            costs += np.power(accel_mean * self.weight, self.power)
        else:
            costs += accel_mean * self.weight


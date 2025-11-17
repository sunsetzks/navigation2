from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from .base import CriticFunction


class SteeringRateCritic(CriticFunction):
    """Penalizes the rate of change of angular velocity (steering rate).
    
    This critic penalizes the rate of change of angular velocity (cwz), which
    represents steering rate for Ackermann motion models. It encourages smoother
    steering maneuvers by penalizing rapid changes in angular velocity.
    
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
        
        # Calculate steering rate (rate of change of angular velocity)
        # steering_rate = (cwz[t] - cwz[t-1]) / dt
        steering_rate = (state.cwz[:, 1:] - state.cwz[:, :-1]) / dt
        
        # Calculate squared steering rate
        steering_rate_squared = steering_rate * steering_rate
        
        # Mean steering rate over time steps for each trajectory
        steering_rate_mean = np.mean(steering_rate_squared, axis=1)
        
        # Apply weight and power
        if self.power > 1:
            costs += np.power(steering_rate_mean * self.weight, self.power)
        else:
            costs += steering_rate_mean * self.weight


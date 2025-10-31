from __future__ import annotations

import numpy as np

from ..critic_data import CriticData
from .base import CriticFunction


class ControlEffortCritic(CriticFunction):
    """Adds a quadratic penalty on high control magnitudes."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight=weight)

    def initialize(self) -> None:  # noqa: D401
        pass

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        state = data.state
        quadratic = (
            state.cvx * state.cvx
            + state.cvy * state.cvy
            + state.cwz * state.cwz
        )
        costs += self.weight * np.mean(quadratic, axis=1)

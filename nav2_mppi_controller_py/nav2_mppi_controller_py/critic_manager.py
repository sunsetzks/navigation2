from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .critic_data import CriticData
from .critics import CriticFunction


class CriticManager:
    """Simple manager that iterates over registered critics."""

    def __init__(self, critics: Iterable[CriticFunction]) -> None:
        self.critics: List[CriticFunction] = list(critics)
        for critic in self.critics:
            critic.initialize()

    def eval_scores(self, data: CriticData, costs: np.ndarray) -> None:
        for critic in self.critics:
            critic.score(data, costs)

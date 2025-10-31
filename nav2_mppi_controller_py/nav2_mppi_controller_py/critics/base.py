from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..critic_data import CriticData


@dataclass
class CriticFunction:
    """Base class for Python critic implementations."""

    weight: float = 1.0
    enabled: bool = True

    def initialize(self) -> None:
        """Optional hook for critic initialisation."""

    def score(self, data: CriticData, costs: np.ndarray) -> None:
        if not self.enabled:
            return
        self.do_score(data, costs)

    def do_score(self, data: CriticData, costs: np.ndarray) -> None:
        raise NotImplementedError

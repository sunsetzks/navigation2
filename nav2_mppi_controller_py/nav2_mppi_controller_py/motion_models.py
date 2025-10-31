from __future__ import annotations

import numpy as np

from .models import ControlSequence, State


class MotionModel:
    """Base motion model for the MPPI controller."""

    def __init__(self) -> None:
        self._is_holonomic = False

    def predict(self, state: State) -> None:
        """Populate state velocities using the commanded controls."""
        if state.cvx.size == 0:
            return

        state.vx[:, 1:] = state.cvx[:, :-1]
        state.wz[:, 1:] = state.cwz[:, :-1]

        if self.is_holonomic():
            state.vy[:, 1:] = state.cvy[:, :-1]

    def is_holonomic(self) -> bool:
        return self._is_holonomic

    def apply_constraints(self, control_sequence: ControlSequence) -> None:
        """Apply hard constraints to the control sequence."""
        # Base class does nothing; subclasses may clamp controls.
        return


class AckermannMotionModel(MotionModel):
    """Ackermann motion model that enforces a minimum turning radius."""

    def __init__(self, min_turning_radius: float = 0.2) -> None:
        super().__init__()
        self.min_turning_radius = min_turning_radius

    def apply_constraints(self, control_sequence: ControlSequence) -> None:
        vx = control_sequence.vx
        wz = control_sequence.wz
        denom = np.abs(wz)
        mask = denom > 0.0
        ratio = np.empty_like(wz)
        ratio[mask] = np.abs(vx[mask]) / denom[mask]
        ratio[~mask] = np.inf

        violation = ratio < self.min_turning_radius
        if np.any(violation):
            # Avoid divide by zero for zero angular velocity
            vx_violation = vx[violation]
            wz[violation] = np.sign(wz[violation]) * np.abs(vx_violation) / max(
                self.min_turning_radius, 1e-6
            )


class DiffDriveMotionModel(MotionModel):
    """Differential drive model (non-holonomic)."""

    def __init__(self) -> None:
        super().__init__()


class OmniMotionModel(MotionModel):
    """Holonomic motion model."""

    def __init__(self) -> None:
        super().__init__()
        self._is_holonomic = True

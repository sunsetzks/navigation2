from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from ..models import ControlSequence, OptimizerSettings, State


class NoiseGenerator:
    """Generates control noise samples for MPPI rollouts."""

    def __init__(self) -> None:
        self._settings: Optional[OptimizerSettings] = None
        self._is_holonomic: bool = False
        self._regenerate_noises: bool = False

        self._noises_vx: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._noises_vy: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._noises_wz: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._active = False
        self._ready = False
        self._thread: Optional[threading.Thread] = None

    def initialize(self, settings: OptimizerSettings, is_holonomic: bool, name: str, param_handler) -> None:
        self._settings = settings
        self._is_holonomic = is_holonomic

        getter = param_handler.get_param_getter(name)
        self._regenerate_noises = getter(
            "regenerate_noises",
            False,
            on_change=self._on_regenerate_changed,
        )

        self._active = True
        if self._regenerate_noises:
            self._thread = threading.Thread(target=self._noise_thread, daemon=True)
            self._thread.start()
        else:
            self._generate_noised_controls()

    def shutdown(self) -> None:
        with self._lock:
            self._active = False
            self._ready = True
            self._cv.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None

    def generate_next_noises(self) -> None:
        with self._lock:
            if not self._regenerate_noises:
                return
            self._ready = True
            self._cv.notify_all()

    def sample(self) -> None:
        with self._lock:
            self._generate_noised_controls()

    def set_noised_controls(self, state: State, control_sequence: ControlSequence) -> None:
        with self._lock:
            if self._settings is None:
                return
            self._ensure_noise_arrays()
            state.cvx = control_sequence.vx + self._noises_vx
            state.cvy = control_sequence.vy + self._noises_vy
            state.cwz = control_sequence.wz + self._noises_wz

    def reset(self, settings: OptimizerSettings, is_holonomic: bool) -> None:
        self._settings = settings
        self._is_holonomic = is_holonomic
        with self._lock:
            self._allocate_noise_arrays()
            self._ready = True
            if self._regenerate_noises:
                self._cv.notify_all()
            else:
                self._generate_noised_controls()

    def _on_regenerate_changed(self, value: bool) -> None:
        self._regenerate_noises = bool(value)

    def _noise_thread(self) -> None:
        while True:
            with self._lock:
                self._cv.wait_for(lambda: self._ready or not self._active)
                if not self._active:
                    return
                self._ready = False
            self._generate_noised_controls()

    def _allocate_noise_arrays(self) -> None:
        if self._settings is None:
            return
        shape = (self._settings.batch_size, self._settings.time_steps)
        self._noises_vx = np.zeros(shape, dtype=np.float32)
        self._noises_wz = np.zeros(shape, dtype=np.float32)
        self._noises_vy = np.zeros(shape, dtype=np.float32)

    def _ensure_noise_arrays(self) -> None:
        if self._settings is None:
            return
        shape = (self._settings.batch_size, self._settings.time_steps)
        if self._noises_vx.shape != shape:
            self._allocate_noise_arrays()

    def _generate_noised_controls(self) -> None:
        if self._settings is None:
            return
        self._ensure_noise_arrays()
        s = self._settings

        self._noises_vx = np.random.normal(0.0, s.sampling_std.vx, size=self._noises_vx.shape).astype(np.float32)
        self._noises_wz = np.random.normal(0.0, s.sampling_std.wz, size=self._noises_wz.shape).astype(np.float32)
        if self._is_holonomic:
            self._noises_vy = np.random.normal(0.0, s.sampling_std.vy, size=self._noises_vy.shape).astype(np.float32)
        else:
            self._noises_vy.fill(0.0)

    @property
    def noises_vx(self) -> np.ndarray:
        return self._noises_vx

    @property
    def noises_vy(self) -> np.ndarray:
        return self._noises_vy

    @property
    def noises_wz(self) -> np.ndarray:
        return self._noises_wz

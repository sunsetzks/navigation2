from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .critic_data import CriticData, GoalChecker
from .critic_manager import CriticManager
from .critics import CriticFunction
from .messages import Path, Pose, Twist, TwistStamped, quaternion_from_yaw, yaw_from_quaternion
from .models import Control, ControlSequence, OptimizerSettings, Path as PathTensor, State, Trajectories
from .motion_models import MotionModel
from .tools import utils
from .tools.noise_generator import NoiseGenerator
from .tools.parameters_handler import ParametersHandler


@dataclass
class OptimizerState:
    state: State
    trajectories: Trajectories
    generated: Trajectories
    path: utils.PathTensor
    control_sequence: ControlSequence
    control_history: list[Control]
    costs: np.ndarray


class Optimizer:
    """Pure Python MPPI optimiser."""

    def __init__(
        self,
        settings: OptimizerSettings,
        motion_model: MotionModel,
        critics: Iterable[CriticFunction],
        parameters_handler: Optional[ParametersHandler] = None,
        noise_generator: Optional[NoiseGenerator] = None,
    ) -> None:
        self.settings = settings
        self.motion_model = motion_model
        self.parameters_handler = parameters_handler or ParametersHandler()
        self.noise_generator = noise_generator or NoiseGenerator()
        self.critic_manager = CriticManager(critics)
        self.goal_checker: GoalChecker = GoalChecker()

        self._init_internal_state()
        self.noise_generator.reset(self.settings, self.motion_model.is_holonomic())

    def _init_internal_state(self) -> None:
        batch = self.settings.batch_size
        time_steps = self.settings.time_steps

        state = State()
        state.reset(batch, time_steps)

        trajectories = Trajectories()
        trajectories.reset(batch, time_steps)

        generated = Trajectories()
        generated.reset(batch, time_steps)

        path = PathTensor()
        path.reset(0)

        control_sequence = ControlSequence()
        control_sequence.reset(time_steps)

        control_history = [Control() for _ in range(4)]
        costs = np.zeros(batch, dtype=np.float32)

        self.opt_state = OptimizerState(
            state=state,
            trajectories=trajectories,
            generated=generated,
            path=path,
            control_sequence=control_sequence,
            control_history=control_history,
            costs=costs,
        )

        self._optimized_index = 0
        self._path_sample_indices = np.zeros(time_steps, dtype=np.int32)
        self._goal_pose = Pose()

    def reset(self) -> None:
        self._init_internal_state()
        self.noise_generator.reset(self.settings, self.motion_model.is_holonomic())

    def set_speed_limit(self, speed_limit: float, percentage: bool) -> None:
        constraints = self.settings.constraints
        if percentage:
            constraints.vx_max = self.settings.base_constraints.vx_max * speed_limit
            constraints.vx_min = self.settings.base_constraints.vx_min * speed_limit
            constraints.vy = self.settings.base_constraints.vy * speed_limit
            constraints.wz = self.settings.base_constraints.wz * speed_limit
        else:
            constraints.vx_max = min(speed_limit, self.settings.base_constraints.vx_max)
            constraints.vx_min = max(-speed_limit, self.settings.base_constraints.vx_min)

    def eval_control(
        self,
        robot_pose: Pose,
        robot_speed: Twist,
        plan: Path,
        goal: Optional[Pose] = None,
    ) -> TwistStamped:
        if goal is None:
            goal = plan.poses[-1].pose if plan.poses else robot_pose
        self._goal_pose = goal

        self._prepare(robot_pose, robot_speed, plan)

        for _ in range(self.settings.iteration_count):
            self._generate_rollouts()
            self._score_rollouts()
            self._update_control_sequence()

        # Apply Savitzky-Golay filter after all iterations are complete
        utils.savitsky_golay_filter(
            self.opt_state.control_sequence,
            self.opt_state.control_history,
            self.settings,
        )

        if self.settings.shift_control_sequence:
            self._shift_control_sequence()

        return self._control_from_sequence()

    def get_generated_trajectories(self) -> Trajectories:
        return self.opt_state.generated

    def get_optimized_trajectory(self) -> np.ndarray:
        idx = self._optimized_index
        traj = np.stack(
            [
                self.opt_state.trajectories.x[idx],
                self.opt_state.trajectories.y[idx],
                self.opt_state.trajectories.yaws[idx],
            ],
            axis=1,
        )
        return traj

    # Internal helpers -----------------------------------------------------

    def _prepare(self, pose: Pose, speed: Twist, plan: Path) -> None:
        s = self.opt_state
        s.state.pose = pose
        s.state.speed = speed

        s.path = utils.path_to_tensor(plan)
        self._path_sample_indices = utils.sample_path_indices(s.path, self.settings.time_steps)

        batch = self.settings.batch_size
        time_steps = self.settings.time_steps

        s.state.reset(batch, time_steps)
        s.trajectories.reset(batch, time_steps)
        s.generated.reset(batch, time_steps)
        s.costs.fill(0.0)

        if s.control_sequence.vx.shape[0] != time_steps:
            s.control_sequence.reset(time_steps)

    def _generate_rollouts(self) -> None:
        self.noise_generator.sample()
        s = self.opt_state
        settings = self.settings

        self.noise_generator.set_noised_controls(s.state, s.control_sequence)

        # Ensure first step matches current speed
        s.state.cvx[:, 0] = s.state.speed.linear.x
        s.state.cwz[:, 0] = s.state.speed.angular.z
        if self.motion_model.is_holonomic():
            s.state.cvy[:, 0] = s.state.speed.linear.y
        else:
            s.state.cvy[:, :] = 0.0

        self._apply_constraints(s.state.cvx, s.state.cvy, s.state.cwz)
        self._propagate_trajectories()
        s.generated = s.trajectories

    def _apply_constraints(self, vx: np.ndarray, vy: np.ndarray, wz: np.ndarray) -> None:
        constraints = self.settings.constraints
        np.clip(vx, constraints.vx_min, constraints.vx_max, out=vx)
        max_vy = abs(constraints.vy)
        np.clip(vy, -max_vy, max_vy, out=vy)
        np.clip(wz, -constraints.wz, constraints.wz, out=wz)

    def _propagate_trajectories(self) -> None:
        s = self.opt_state
        batch = self.settings.batch_size
        time_steps = self.settings.time_steps
        dt = self.settings.model_dt

        pose = s.state.pose
        x = np.full(batch, pose.position.x, dtype=np.float32)
        y = np.full(batch, pose.position.y, dtype=np.float32)
        yaw = np.full(batch, yaw_from_quaternion(pose.orientation), dtype=np.float32)

        s.trajectories.x[:, 0] = x
        s.trajectories.y[:, 0] = y
        s.trajectories.yaws[:, 0] = yaw

        for t in range(1, time_steps):
            vx = s.state.cvx[:, t - 1]
            vy = s.state.cvy[:, t - 1]
            wz = s.state.cwz[:, t - 1]

            yaw = yaw + wz * dt
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            x = x + (vx * cos_yaw - vy * sin_yaw) * dt
            y = y + (vx * sin_yaw + vy * cos_yaw) * dt

            s.trajectories.x[:, t] = x
            s.trajectories.y[:, t] = y
            s.trajectories.yaws[:, t] = yaw

    def _score_rollouts(self) -> None:
        s = self.opt_state

        # Build truncated path tensor aligned with the horizon
        indices = self._path_sample_indices
        sample_count = len(indices)
        truncated = PathTensor()
        truncated.reset(sample_count)
        truncated.x[:] = s.path.x[indices]
        truncated.y[:] = s.path.y[indices]
        truncated.yaws[:] = s.path.yaws[indices]

        data = CriticData(
            state=s.state,
            trajectories=s.trajectories,
            path=truncated,
            goal=self._goal_pose,
            costs=s.costs,
            model_dt=self.settings.model_dt,
            goal_checker=self.goal_checker,
            motion_model=self.motion_model,
        )
        s.costs.fill(0.0)
        self.critic_manager.eval_scores(data, s.costs)
        self._optimized_index = int(np.argmin(s.costs))

    def _update_control_sequence(self) -> None:
        s = self.opt_state
        settings = self.settings

        # Calculate bounded noises (control differences)
        bounded_noises_vx = s.state.cvx - s.control_sequence.vx
        bounded_noises_wz = s.state.cwz - s.control_sequence.wz

        # Add gamma cost term for control noise penalty
        if settings.gamma > 0:
            gamma_cost_vx = (
                settings.gamma
                / (settings.sampling_std.vx * settings.sampling_std.vx)
                * np.sum(
                    s.control_sequence.vx[None, :] * bounded_noises_vx, axis=1
                )
            )
            gamma_cost_wz = (
                settings.gamma
                / (settings.sampling_std.wz * settings.sampling_std.wz)
                * np.sum(
                    s.control_sequence.wz[None, :] * bounded_noises_wz, axis=1
                )
            )
            s.costs += gamma_cost_vx + gamma_cost_wz

            if self.motion_model.is_holonomic():
                bounded_noises_vy = s.state.cvy - s.control_sequence.vy
                gamma_cost_vy = (
                    settings.gamma
                    / (settings.sampling_std.vy * settings.sampling_std.vy)
                    * np.sum(
                        s.control_sequence.vy[None, :] * bounded_noises_vy, axis=1
                    )
                )
                s.costs += gamma_cost_vy

        # Calculate softmax weights
        temperature = max(settings.temperature, 1e-5)
        costs_normalized = s.costs - np.min(s.costs)
        exponents = np.exp(-1.0 / temperature * costs_normalized)
        softmaxes = exponents / (np.sum(exponents) + 1e-9)

        # Update control sequence using weighted average (like C++ version)
        s.control_sequence.vx = np.sum(s.state.cvx * softmaxes[:, None], axis=0)
        s.control_sequence.wz = np.sum(s.state.cwz * softmaxes[:, None], axis=0)

        if self.motion_model.is_holonomic():
            s.control_sequence.vy = np.sum(s.state.cvy * softmaxes[:, None], axis=0)
        else:
            s.control_sequence.vy.fill(0.0)

        # Apply constraints (including motion model constraints)
        self._apply_constraints(
            s.control_sequence.vx[None, :],
            s.control_sequence.vy[None, :],
            s.control_sequence.wz[None, :],
        )
        # Apply motion model specific constraints (e.g., Ackermann turning radius)
        self.motion_model.apply_constraints(s.control_sequence)

    def _shift_control_sequence(self) -> None:
        seq = self.opt_state.control_sequence
        for arr in (seq.vx, seq.vy, seq.wz):
            arr[:-1] = arr[1:]
            arr[-1] = arr[-2]

    def _control_from_sequence(self) -> TwistStamped:
        seq = self.opt_state.control_sequence
        # Use offset=1 if shift_control_sequence is enabled (like C++ version)
        offset = 1 if self.settings.shift_control_sequence else 0
        return utils.to_twist_stamped(
            seq.vx[offset], seq.vy[offset], seq.wz[offset], frame="base_link"
        )

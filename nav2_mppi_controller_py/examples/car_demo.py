from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from nav2_mppi_controller_py import (
    ControlConstraints,
    ControlEffortCritic,
    DiffDriveMotionModel,
    GoalDistanceCritic,
    GoalHeadingCritic,
    MPPIController,
    Optimizer,
    OptimizerSettings,
    PathFollowCritic,
    SamplingStd,
    Pose,
    Twist,
    build_path_from_xy,
    quaternion_from_yaw,
    yaw_from_quaternion,
)


def create_settings(
    dt: float,
    batch_size: int,
    horizon_steps: int,
    iterations: int,
    temperature: float,
) -> OptimizerSettings:
    settings = OptimizerSettings()
    settings.model_dt = dt
    settings.gamma = 0.02
    settings.temperature = temperature
    settings.batch_size = batch_size
    settings.time_steps = horizon_steps
    settings.iteration_count = iterations
    settings.shift_control_sequence = True
    settings.retry_attempt_limit = 1

    base_constraints = ControlConstraints(vx_max=0.8, vx_min=-0.2, vy=0.0, wz=1.5)
    settings.base_constraints = base_constraints
    settings.constraints = ControlConstraints(**vars(base_constraints))
    settings.sampling_std = SamplingStd(vx=0.25, vy=0.0, wz=0.35)
    return settings


def prune_path(plan, pose: Pose, look_ahead: int, dt: float = 0.05, v_ref: float = 0.5) -> object:
    """Return a pruned copy of the plan keeping poses within a time horizon ahead of the robot."""
    if not plan.poses:
        return plan

    positions = np.array([[p.pose.position.x, p.pose.position.y] for p in plan.poses])
    robot_xy = np.array([pose.position.x, pose.position.y])
    dists = np.linalg.norm(positions - robot_xy, axis=1)
    idx = int(np.argmin(dists))

    # Calculate distances between consecutive poses
    if len(positions) > 1:
        deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        # Assume constant reference speed to estimate time per pose
        dt_per_pose = deltas / v_ref
        cumulative_time = np.cumsum(dt_per_pose)
        # Find how many poses fit within look_ahead time (look_ahead here is time in seconds)
        max_time = look_ahead  # look_ahead is now time horizon
        pose_indices = np.where(cumulative_time <= max_time)[0]
        if len(pose_indices) > 0:
            end_idx = idx + pose_indices[-1] + 1  # +1 because cumsum starts from idx+1
        else:
            end_idx = idx + 1
    else:
        end_idx = idx + 1

    trimmed = plan.poses[idx:end_idx]
    if not trimmed:
        trimmed = [plan.poses[idx]]  # At least keep current pose

    new_plan = type(plan)()
    new_plan.header = plan.header
    new_plan.poses = list(trimmed)
    return new_plan


def build_reference_path(shape: str, samples: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    if shape == "lemniscate":
        denom = np.sin(t) * np.sin(t) + 1.0
        xs = radius * np.sin(t) / denom
        ys = radius * np.sin(t) * np.cos(t) / denom
    elif shape == "line":
        xs = np.linspace(-radius, radius, samples)
        ys = np.zeros_like(xs)
    else:  # default to circle
        xs = radius * np.cos(t)
        ys = radius * np.sin(t)
    return xs, ys


def simulate(
    controller: MPPIController,
    path,
    pose: Pose,
    twist: Twist,
    steps: int,
    dt: float,
    look_ahead: float,
    v_ref: float = 0.5,
    horizon_steps: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    positions: list[Tuple[float, float]] = []
    headings: list[float] = []
    velocities: list[float] = []
    predicted_trajectories: list[np.ndarray] = []
    plan_windows: list[list] = []

    for _ in range(steps):
        plan_window = prune_path(path, pose, look_ahead=look_ahead, dt=dt, v_ref=v_ref)
        controller.set_plan(plan_window)
        cmd = controller.compute_velocity_command(pose, twist)

        vx = cmd.twist.linear.x
        wz = cmd.twist.angular.z

        # Simulate predicted trajectory assuming constant control for horizon
        pred_positions = []
        pred_yaw = yaw_from_quaternion(pose.orientation)
        pred_x = pose.position.x
        pred_y = pose.position.y
        for _ in range(horizon_steps):
            pred_yaw += wz * dt
            pred_x += math.cos(pred_yaw) * vx * dt
            pred_y += math.sin(pred_yaw) * vx * dt
            pred_positions.append((pred_x, pred_y))
        predicted_trajectories.append(np.array(pred_positions))

        yaw = yaw_from_quaternion(pose.orientation)
        yaw += wz * dt
        pose.orientation = quaternion_from_yaw(yaw)
        pose.position.x += math.cos(yaw) * vx * dt
        pose.position.y += math.sin(yaw) * vx * dt

        twist.linear.x = vx
        twist.angular.z = wz

        positions.append((pose.position.x, pose.position.y))
        headings.append(yaw)
        velocities.append(vx)

        plan_windows.append([(p.pose.position.x, p.pose.position.y) for p in plan_window.poses])

    return np.asarray(positions), np.asarray(headings), np.asarray(velocities), predicted_trajectories, plan_windows


def build_controller(settings: OptimizerSettings) -> MPPIController:
    motion_model = DiffDriveMotionModel()
    critics = [
        PathFollowCritic(weight=12.0),
        GoalDistanceCritic(weight=20.0),
        GoalHeadingCritic(weight=5.0),
        ControlEffortCritic(weight=0.1),
    ]
    optimizer = Optimizer(settings, motion_model, critics)
    return MPPIController(optimizer)


def get_vehicle_corners(x: float, y: float, yaw: float, length: float, width: float) -> list[Tuple[float, float]]:
    """Calculate the four corners of the vehicle rectangle."""
    half_l = length / 2
    half_w = width / 2
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    corners = [
        (x + half_l * cos_yaw - half_w * sin_yaw, y + half_l * sin_yaw + half_w * cos_yaw),  # front right
        (x + half_l * cos_yaw + half_w * sin_yaw, y + half_l * sin_yaw - half_w * cos_yaw),  # front left
        (x - half_l * cos_yaw + half_w * sin_yaw, y - half_l * sin_yaw - half_w * cos_yaw),  # rear left
        (x - half_l * cos_yaw - half_w * sin_yaw, y - half_l * sin_yaw + half_w * cos_yaw),  # rear right
    ]
    return corners


def run_demo(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pure-Python MPPI car demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Controller time step [s]")
    parser.add_argument("--steps", type=int, default=360, help="Simulation steps to roll out")
    parser.add_argument("--radius", type=float, default=2.0, help="Reference path radius/scale")
    parser.add_argument(
        "--path-shape",
        choices=("circle", "lemniscate", "line"),
        default="circle",
        help="Reference path geometry",
    )
    parser.add_argument("--look-ahead", type=float, default=2.0, help="Time horizon [s] for pruning the plan")
    parser.add_argument("--batch-size", type=int, default=512, help="MPPI batch size")
    parser.add_argument("--time-steps", type=int, default=40, help="MPPI horizon length")
    parser.add_argument("--iterations", type=int, default=2, help="MPPI optimisation iterations per cycle")
    parser.add_argument("--temperature", type=float, default=0.4, help="MPPI temperature (softmax smoothing)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise sampling")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="GIF output path. Defaults to nav2_mppi_controller_py/media/python_demo.gif",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip saving the animation to disk")
    parser.add_argument("--show", action="store_true", help="Display the animation window")
    parser.add_argument("--v-ref", type=float, default=0.5, help="Reference speed [m/s] for time-based pruning")
    args = parser.parse_args(list(argv) if argv else None)

    np.random.seed(args.seed)

    settings = create_settings(
        dt=args.dt,
        batch_size=args.batch_size,
        horizon_steps=args.time_steps,
        iterations=args.iterations,
        temperature=args.temperature,
    )
    controller = build_controller(settings)

    xs, ys = build_reference_path(args.path_shape, args.steps, args.radius)
    path = build_path_from_xy(xs, ys)
    controller.set_plan(path)

    pose = Pose()
    pose.position.x = 0.0
    pose.position.y = 0.1
    initial_yaw = math.atan2(ys[1] - ys[0], xs[1] - xs[0]) if args.steps > 1 else 0.0
    pose.orientation = quaternion_from_yaw(initial_yaw)

    twist = Twist()

    positions, headings, velocities, predicted_trajectories, plan_windows = simulate(
        controller,
        path,
        pose,
        twist,
        steps=args.steps,
        dt=args.dt,
        look_ahead=args.look_ahead,
        v_ref=args.v_ref,
        horizon_steps=args.time_steps,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.plot(xs, ys, "k--", label="Reference path")
    # Mark the end position and orientation
    end_x, end_y = xs[-1], ys[-1]
    if len(xs) > 1:
        dx = xs[-1] - xs[-2]
        dy = ys[-1] - ys[-2]
        end_yaw = math.atan2(dy, dx)
        arrow_length = 0.5
        ax.arrow(end_x, end_y, arrow_length * math.cos(end_yaw), arrow_length * math.sin(end_yaw),
                 head_width=0.1, head_length=0.1, fc='purple', ec='purple', label="End orientation")
    ax.plot(end_x, end_y, 'ko', markersize=8, label="End position")
    (robot_path_plot,) = ax.plot([], [], "r-", linewidth=2, label="MPPI trajectory")
    (predicted_path_plot,) = ax.plot([], [], "b--", linewidth=1, label="Predicted trajectory")
    (plan_window_plot,) = ax.plot([], [], "g-", linewidth=1.5, label="Plan window")
    vehicle_length = 0.5
    vehicle_width = 0.3
    (vehicle_body,) = ax.fill([], [], color='red', alpha=0.5, label="Vehicle")
    ax.set_xlim(xs.min() - 1.0, xs.max() + 1.0)
    ax.set_ylim(ys.min() - 1.0, ys.max() + 1.0)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("nav2_mppi_controller_py demo")

    def init():
        robot_path_plot.set_data([], [])
        predicted_path_plot.set_data([], [])
        plan_window_plot.set_data([], [])
        vehicle_body.set_xy(np.empty((0, 2)))
        return robot_path_plot, predicted_path_plot, plan_window_plot, vehicle_body

    def update(frame):
        robot_path_plot.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
        if frame < len(predicted_trajectories):
            pred_pos = predicted_trajectories[frame]
            predicted_path_plot.set_data(pred_pos[:, 0], pred_pos[:, 1])
        if frame < len(plan_windows):
            pw_pos = np.array(plan_windows[frame])
            if len(pw_pos) > 0:
                plan_window_plot.set_data(pw_pos[:, 0], pw_pos[:, 1])
            else:
                plan_window_plot.set_data([], [])
        x = positions[frame, 0]
        y = positions[frame, 1]
        yaw = float(headings[frame])
        corners = get_vehicle_corners(x, y, yaw, vehicle_length, vehicle_width)
        corners_array = np.array(corners)
        vehicle_body.set_xy(corners_array)
        return robot_path_plot, predicted_path_plot, plan_window_plot, vehicle_body

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(positions),
        init_func=init,
        interval=max(int(args.dt * 1000.0), 10),
        blit=True,
    )

    if not args.no_save:
        output = (
            args.output.resolve()
            if args.output is not None
            else (Path(__file__).resolve().parent.parent / "media" / "python_demo.gif")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        ani.save(output, writer="pillow", fps=max(int(1 / args.dt), 1))
        print(f"Animation saved to {output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    run_demo()

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
    CarVisualizer,
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


def prune_path(plan, pose: Pose, look_ahead: float, dt: float = 0.05, v_ref: float = 0.5):
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
    path_xs: np.ndarray,
    path_ys: np.ndarray,
    v_ref: float = 0.5,
    horizon_steps: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, list, list, list, list, list, list, list]:
    positions: list[Tuple[float, float]] = []
    headings: list[float] = []
    velocities: list[float] = []
    predicted_trajectories: list[np.ndarray] = []
    plan_windows: list[list] = []
    nearest_points: list[Tuple[float, float]] = []
    steering_angles: list[float] = []
    command_velocities: list[Tuple[float, float]] = []  # (vx_cmd, wz_cmd)
    control_sequences_before_filter: list[np.ndarray] = []  # Store vx sequences before filter
    control_sequences_after_filter: list[np.ndarray] = []  # Store vx sequences after filter
    control_sequences_wz_before_filter: list[np.ndarray] = []  # Store wz sequences before filter
    control_sequences_wz_after_filter: list[np.ndarray] = []  # Store wz sequences after filter

    for _ in range(steps):
        plan_window = prune_path(path, pose, look_ahead=look_ahead, dt=dt, v_ref=v_ref)
        controller.set_plan(plan_window)
        cmd = controller.compute_velocity_command(pose, twist)

        # Get control sequences before and after filtering
        optimizer = controller.optimizer
        seq_before = optimizer.get_control_sequence_before_filter()
        seq_after = optimizer.get_control_sequence_after_filter()
        
        if seq_before is not None:
            control_sequences_before_filter.append(seq_before.vx.copy())
            control_sequences_wz_before_filter.append(seq_before.wz.copy())
        else:
            control_sequences_before_filter.append(np.array([]))
            control_sequences_wz_before_filter.append(np.array([]))
        
        control_sequences_after_filter.append(seq_after.vx.copy())
        control_sequences_wz_after_filter.append(seq_after.wz.copy())

        vx = cmd.twist.linear.x
        wz = cmd.twist.angular.z

        # Calculate steering angle
        steering_angle = math.atan(0.5 * wz / vx) if abs(vx) > 0.01 else 0.0
        steering_angles.append(steering_angle)

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
        
        # Store command velocities
        command_velocities.append((vx, wz))

    return (np.asarray(positions), np.asarray(headings), np.asarray(velocities), 
            predicted_trajectories, plan_windows, nearest_points, steering_angles, 
            command_velocities, control_sequences_before_filter, control_sequences_after_filter,
            control_sequences_wz_before_filter, control_sequences_wz_after_filter)


def build_controller(settings: OptimizerSettings) -> MPPIController:
    motion_model = DiffDriveMotionModel()
    critics = [
        PathFollowCritic(weight=12.0),
        GoalDistanceCritic(weight=10.0),
        GoalHeadingCritic(weight=0.0),
        ControlEffortCritic(weight=0.1),
    ]
    optimizer = Optimizer(settings, motion_model, critics)
    return MPPIController(optimizer)


def get_nearest_path_point(robot_pos: Tuple[float, float], path_xs: np.ndarray, path_ys: np.ndarray) -> Tuple[float, float]:
    """Find the nearest point on the reference path to the robot's current position."""
    robot_xy = np.array(robot_pos)
    path_positions = np.column_stack([path_xs, path_ys])
    dists = np.linalg.norm(path_positions - robot_xy, axis=1)
    nearest_idx = int(np.argmin(dists))
    return path_xs[nearest_idx], path_ys[nearest_idx]


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
    parser.add_argument("--look-ahead", type=float, default=1.0, help="Time horizon [s] for pruning the plan")
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
    parser.add_argument("--no-show", action="store_true", help="Skip displaying the animation window")
    parser.add_argument("--save", action="store_true", help="Save the animation to disk")
    parser.add_argument("--v-ref", type=float, default=0.5, help="Reference speed [m/s] for time-based pruning")
    parser.add_argument("--car-style", type=str, default="default", 
                       choices=["default", "sedan", "truck", "sports", "compact"],
                       help="Car visualization style")
    parser.add_argument("--wheelbase-length", type=float, default=None,
                       help="Wheelbase length [m] (distance between front/rear axles). If not specified, uses style default.")
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

    (positions, headings, velocities, predicted_trajectories, plan_windows, nearest_points, 
     steering_angles, command_velocities, control_sequences_before_filter, 
     control_sequences_after_filter, control_sequences_wz_before_filter, 
     control_sequences_wz_after_filter) = simulate(
        controller,
        path,
        pose,
        twist,
        steps=args.steps,
        dt=args.dt,
        look_ahead=args.look_ahead,
        path_xs=xs,
        path_ys=ys,
        v_ref=args.v_ref,
        horizon_steps=args.time_steps,
    )

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax_main = fig.add_subplot(gs[0, :])
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_steering = fig.add_subplot(gs[1, 1])
    ax_filter_vx = fig.add_subplot(gs[2, 0])
    ax_filter_wz = fig.add_subplot(gs[2, 1])
    ax = ax_main  # Keep ax as alias for backward compatibility
    ax.set_aspect("equal")
    ax.plot(xs, ys, "k--", label="Reference path", alpha=0.5)
    # Mark the end position and orientation
    end_x, end_y = xs[-1], ys[-1]
    if len(xs) > 1:
        dx = xs[-1] - xs[-2]
        dy = ys[-1] - ys[-2]
        end_yaw = math.atan2(dy, dx)
        arrow_length = 0.5
        ax.arrow(end_x, end_y, arrow_length * math.cos(end_yaw), arrow_length * math.sin(end_yaw),
                 head_width=0.1, head_length=0.1, fc='purple', ec='purple', label="End orientation", alpha=0.7)
    ax.plot(end_x, end_y, 'ko', markersize=8, label="End position", alpha=0.7)
    (robot_path_plot,) = ax.plot([], [], "r-", linewidth=2, label="MPPI trajectory", alpha=0.8)
    (predicted_path_plot,) = ax.plot([], [], "b--", linewidth=1, label="Predicted trajectory", alpha=0.5)
    (plan_window_plot,) = ax.plot([], [], "g-", linewidth=1.5, label="Plan window", alpha=0.6)
    (nearest_point_plot,) = ax.plot([], [], "mo", markersize=8, label="Nearest path point", alpha=0.9)
    car_visualizer = CarVisualizer(style=args.car_style, wheelbase_length=args.wheelbase_length)
    car_visualizer.initialize_plot(ax)
    ax.set_xlim(xs.min() - 1.0, xs.max() + 1.0)
    ax.set_ylim(ys.min() - 1.0, ys.max() + 1.0)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("nav2_mppi_controller_py demo")

    # Set up velocity subplot
    ax_vel.set_title("Command Velocities")
    ax_vel.set_xlabel("Time [s]")
    ax_vel.set_ylabel("Velocity")
    time_steps = np.arange(len(command_velocities)) * args.dt
    vx_cmds = [cmd[0] for cmd in command_velocities]
    wz_cmds = [cmd[1] for cmd in command_velocities]
    
    (vx_plot,) = ax_vel.plot([], [], "b-", linewidth=2, label="Linear velocity (vx)", alpha=0.8)
    (wz_plot,) = ax_vel.plot([], [], "r-", linewidth=2, label="Angular velocity (wz)", alpha=0.8)
    ax_vel.legend()
    ax_vel.grid(True, alpha=0.3)
    
    # Set velocity plot limits
    ax_vel.set_xlim(0, time_steps[-1])
    vx_margin = 0.1 * (max(vx_cmds) - min(vx_cmds)) if vx_cmds else 0.1
    wz_margin = 0.1 * (max(wz_cmds) - min(wz_cmds)) if wz_cmds else 0.1
    ax_vel.set_ylim(min(min(vx_cmds) - vx_margin, min(wz_cmds) - wz_margin), 
                    max(max(vx_cmds) + vx_margin, max(wz_cmds) + wz_margin))

    # Set up steering angle subplot
    ax_steering.set_title("Steering Wheel Angle")
    ax_steering.set_xlabel("Time [s]")
    ax_steering.set_ylabel("Steering Angle [rad]")
    
    (steering_plot,) = ax_steering.plot([], [], "g-", linewidth=2, label="Steering angle", alpha=0.8)
    ax_steering.legend()
    ax_steering.grid(True, alpha=0.3)
    
    # Set steering angle plot limits
    ax_steering.set_xlim(0, time_steps[-1])
    steering_margin = 0.1 * (max(steering_angles) - min(steering_angles)) if steering_angles else 0.1
    ax_steering.set_ylim(min(steering_angles) - steering_margin, max(steering_angles) + steering_margin)

    # Set up filter comparison subplots
    ax_filter_vx.set_title("Control Sequence vx: Before vs After Filter")
    ax_filter_vx.set_xlabel("Time Step in Horizon")
    ax_filter_vx.set_ylabel("vx [m/s]")
    ax_filter_vx.grid(True, alpha=0.3)
    
    ax_filter_wz.set_title("Control Sequence wz: Before vs After Filter")
    ax_filter_wz.set_xlabel("Time Step in Horizon")
    ax_filter_wz.set_ylabel("wz [rad/s]")
    ax_filter_wz.grid(True, alpha=0.3)

    # Initialize filter comparison plots
    (filter_vx_before_plot,) = ax_filter_vx.plot([], [], "r-", linewidth=1.5, label="Before filter", alpha=0.7)
    (filter_vx_after_plot,) = ax_filter_vx.plot([], [], "b-", linewidth=1.5, label="After filter", alpha=0.7)
    ax_filter_vx.legend()
    
    (filter_wz_before_plot,) = ax_filter_wz.plot([], [], "r-", linewidth=1.5, label="Before filter", alpha=0.7)
    (filter_wz_after_plot,) = ax_filter_wz.plot([], [], "b-", linewidth=1.5, label="After filter", alpha=0.7)
    ax_filter_wz.legend()
    
    def init():
        robot_path_plot.set_data([], [])
        predicted_path_plot.set_data([], [])
        plan_window_plot.set_data([], [])
        nearest_point_plot.set_data([], [])
        vx_plot.set_data([], [])
        wz_plot.set_data([], [])
        steering_plot.set_data([], [])
        filter_vx_before_plot.set_data([], [])
        filter_vx_after_plot.set_data([], [])
        filter_wz_before_plot.set_data([], [])
        filter_wz_after_plot.set_data([], [])
        car_elements = car_visualizer.update_plot(0, 0, 0, 0)
        return [robot_path_plot, predicted_path_plot, plan_window_plot, nearest_point_plot, 
                vx_plot, wz_plot, steering_plot, filter_vx_before_plot, filter_vx_after_plot,
                filter_wz_before_plot, filter_wz_after_plot] + car_elements

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
        if frame < len(nearest_points):
            nearest_pt = nearest_points[frame]
            nearest_point_plot.set_data([nearest_pt[0]], [nearest_pt[1]])
        
        # Update velocity plots
        current_time = time_steps[:frame + 1]
        vx_plot.set_data(current_time, vx_cmds[:frame + 1])
        wz_plot.set_data(current_time, wz_cmds[:frame + 1])
        
        # Update steering angle plot
        steering_plot.set_data(current_time, steering_angles[:frame + 1])
        
        # Update filter comparison plots
        if frame < len(control_sequences_before_filter) and frame < len(control_sequences_after_filter):
            seq_before_vx = control_sequences_before_filter[frame]
            seq_after_vx = control_sequences_after_filter[frame]
            seq_before_wz = control_sequences_wz_before_filter[frame]
            seq_after_wz = control_sequences_wz_after_filter[frame]
            
            if len(seq_before_vx) > 0 and len(seq_after_vx) > 0:
                time_steps_horizon = np.arange(len(seq_after_vx))
                filter_vx_before_plot.set_data(time_steps_horizon, seq_before_vx)
                filter_vx_after_plot.set_data(time_steps_horizon, seq_after_vx)
                
                # Update axis limits for vx
                vx_min = min(seq_before_vx.min() if len(seq_before_vx) > 0 else 0, 
                            seq_after_vx.min() if len(seq_after_vx) > 0 else 0)
                vx_max = max(seq_before_vx.max() if len(seq_before_vx) > 0 else 0, 
                            seq_after_vx.max() if len(seq_after_vx) > 0 else 0)
                if vx_max > vx_min:
                    ax_filter_vx.set_xlim(0, len(seq_after_vx) - 1)
                    ax_filter_vx.set_ylim(vx_min - 0.1, vx_max + 0.1)
                
            if len(seq_before_wz) > 0 and len(seq_after_wz) > 0:
                time_steps_horizon = np.arange(len(seq_after_wz))
                filter_wz_before_plot.set_data(time_steps_horizon, seq_before_wz)
                filter_wz_after_plot.set_data(time_steps_horizon, seq_after_wz)
                
                # Update axis limits for wz
                wz_min = min(seq_before_wz.min() if len(seq_before_wz) > 0 else 0, 
                            seq_after_wz.min() if len(seq_after_wz) > 0 else 0)
                wz_max = max(seq_before_wz.max() if len(seq_before_wz) > 0 else 0, 
                            seq_after_wz.max() if len(seq_after_wz) > 0 else 0)
                if wz_max > wz_min:
                    ax_filter_wz.set_xlim(0, len(seq_after_wz) - 1)
                    ax_filter_wz.set_ylim(wz_min - 0.1, wz_max + 0.1)
        
        x = positions[frame, 0]
        y = positions[frame, 1]
        yaw = float(headings[frame])
        steering_angle = steering_angles[frame] if frame < len(steering_angles) else 0.0
        car_elements = car_visualizer.update_plot(x, y, yaw, steering_angle)
        return [robot_path_plot, predicted_path_plot, plan_window_plot, nearest_point_plot, 
                vx_plot, wz_plot, steering_plot, filter_vx_before_plot, filter_vx_after_plot,
                filter_wz_before_plot, filter_wz_after_plot] + car_elements

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(positions),
        init_func=init,
        interval=max(int(args.dt * 1000.0), 10),
        blit=True,
    )

    # Add pause button
    ax_pause = fig.add_axes((0.92, 0.05, 0.05, 0.04))
    btn_pause = Button(ax_pause, 'Pause', color='lightgoldenrodyellow', hovercolor='lightyellow')
    
    pause_state = {'paused': False, 'current_frame': 0}
    info_texts = []  # Store info text for updating
    
    def on_pause(event):
        if pause_state['paused']:
            ani.event_source.start()
            btn_pause.label.set_text('Pause')
            pause_state['paused'] = False
        else:
            ani.event_source.stop()
            btn_pause.label.set_text('Play')
            pause_state['paused'] = True
    
    def on_key(event):
        if event.key == ' ':  # Space bar to toggle pause
            on_pause(None)
        elif event.key == 'right' and pause_state['paused']:  # Right arrow to step forward
            pause_state['current_frame'] = min(pause_state['current_frame'] + 1, len(positions) - 1)
            # Clear the current plot elements and redraw
            init()
            update(pause_state['current_frame'])
            # Update info text
            for info_text in info_texts:
                info_text.set_text(f'Space: Pause/Play | ← →: Step | Step: {pause_state["current_frame"]}/{len(positions) - 1}')
            fig.canvas.draw()
        elif event.key == 'left' and pause_state['paused']:  # Left arrow to step backward
            pause_state['current_frame'] = max(pause_state['current_frame'] - 1, 0)
            # Clear the current plot elements and redraw
            init()
            update(pause_state['current_frame'])
            # Update info text
            for info_text in info_texts:
                info_text.set_text(f'Space: Pause/Play | ← →: Step | Step: {pause_state["current_frame"]}/{len(positions) - 1}')
            fig.canvas.draw()
    
    btn_pause.on_clicked(on_pause)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Add info text
    info_text = fig.text(0.5, 0.01, f'Space: Pause/Play | ← →: Step | Step: 0/{len(positions) - 1}',
                         ha='center', fontsize=9, color='gray')
    info_texts.append(info_text)

    if args.save:
        output = (
            args.output.resolve()
            if args.output is not None
            else (Path(__file__).resolve().parent.parent / "media" / "python_demo.gif")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        ani.save(output, writer="pillow", fps=max(int(1 / args.dt), 1))
        print(f"Animation saved to {output}")

    if not args.no_show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    run_demo()

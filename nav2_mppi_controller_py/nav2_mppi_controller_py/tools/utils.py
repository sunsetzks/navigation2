from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import numpy as np

from ..messages import (
    ColorRGBA,
    Header,
    Marker,
    MarkerArray,
    Path,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    Time,
    Twist,
    TwistStamped,
    Vector3,
    quaternion_from_yaw,
    yaw_from_quaternion,
)
from ..models import Control, ControlSequence, OptimizerSettings, Path as PathTensor
from ..critic_data import CriticData


def create_pose(x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> Pose:
    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    pose.orientation = quaternion_from_yaw(yaw)
    return pose


def create_scale(x: float, y: float, z: float) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def create_color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=r, g=g, b=b, a=a)


def create_marker(
    marker_id: int,
    pose: Pose,
    scale: Vector3,
    color: ColorRGBA,
    frame_id: str,
    namespace: str,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=Time(), frame_id=frame_id)
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose = pose
    marker.scale = scale
    marker.color = color
    return marker


def to_twist_stamped(vx: float, vy: float, wz: float, frame: str) -> TwistStamped:
    twist = TwistStamped()
    twist.header = Header(stamp=Time(), frame_id=frame)
    twist.twist.linear.x = float(vx)
    twist.twist.linear.y = float(vy)
    twist.twist.angular.z = float(wz)
    return twist


def path_to_tensor(path: Path) -> PathTensor:
    tensor = PathTensor()
    tensor.reset(len(path.poses))
    for idx, pose_stamped in enumerate(path.poses):
        pose = pose_stamped.pose
        tensor.x[idx] = pose.position.x
        tensor.y[idx] = pose.position.y
        tensor.yaws[idx] = yaw_from_quaternion(pose.orientation)
    return tensor


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def normalize_angles(arr: np.ndarray) -> np.ndarray:
    return np.vectorize(normalize_angle)(arr)


def shortest_angular_distance(from_angle: np.ndarray, to_angle: np.ndarray) -> np.ndarray:
    return normalize_angles(to_angle - from_angle)


def savitsky_golay_filter(
    control_sequence: ControlSequence,
    control_history: List[Control],
    settings: OptimizerSettings,
) -> None:
    coeffs = np.array([-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0], dtype=np.float32) / 231.0
    num_sequences = control_sequence.vx.shape[0] - 1
    if num_sequences < 20:
        return

    def apply_filter(samples: Sequence[float]) -> float:
        return float(np.dot(coeffs, np.asarray(samples, dtype=np.float32)))

    def apply_over_axis(sequence: np.ndarray, history: Sequence[float]) -> None:
        seq = sequence
        h0, h1, h2, h3 = history
        idx = 0
        seq[idx] = apply_filter((h0, h1, h2, h3, seq[idx], seq[idx + 1], seq[idx + 2], seq[idx + 3], seq[idx + 4]))
        idx += 1
        seq[idx] = apply_filter((h1, h2, h3, seq[idx - 1], seq[idx], seq[idx + 1], seq[idx + 2], seq[idx + 3], seq[idx + 4]))
        idx += 1
        seq[idx] = apply_filter((h2, h3, seq[idx - 2], seq[idx - 1], seq[idx], seq[idx + 1], seq[idx + 2], seq[idx + 3], seq[idx + 4]))
        idx += 1
        seq[idx] = apply_filter((h3, seq[idx - 3], seq[idx - 2], seq[idx - 1], seq[idx], seq[idx + 1], seq[idx + 2], seq[idx + 3], seq[idx + 4]))
        for idx in range(4, num_sequences - 4):
            seq[idx] = apply_filter(
                (
                    seq[idx - 4],
                    seq[idx - 3],
                    seq[idx - 2],
                    seq[idx - 1],
                    seq[idx],
                    seq[idx + 1],
                    seq[idx + 2],
                    seq[idx + 3],
                    seq[idx + 4],
                )
            )
        idx = num_sequences - 4
        seq[idx] = apply_filter(
            (
                seq[idx - 4],
                seq[idx - 3],
                seq[idx - 2],
                seq[idx - 1],
                seq[idx],
                seq[idx + 1],
                seq[idx + 2],
                seq[idx + 3],
                seq[idx + 4],
            )
        )
        idx += 1
        seq[idx] = apply_filter(
            (
                seq[idx - 4],
                seq[idx - 3],
                seq[idx - 2],
                seq[idx - 1],
                seq[idx],
                seq[idx + 1],
                seq[idx + 2],
                seq[idx + 2],
                seq[idx + 2],
            )
        )
        idx += 1
        seq[idx] = apply_filter(
            (
                seq[idx - 4],
                seq[idx - 3],
                seq[idx - 2],
                seq[idx - 1],
                seq[idx],
                seq[idx + 1],
                seq[idx + 1],
                seq[idx + 1],
                seq[idx + 1],
            )
        )
        idx += 1
        seq[idx] = apply_filter(
            (
                seq[idx - 4],
                seq[idx - 3],
                seq[idx - 2],
                seq[idx - 1],
                seq[idx],
                seq[idx],
                seq[idx],
                seq[idx],
                seq[idx],
            )
        )

    apply_over_axis(
        control_sequence.vx,
        (control_history[0].vx, control_history[1].vx, control_history[2].vx, control_history[3].vx),
    )
    apply_over_axis(
        control_sequence.vy,
        (control_history[0].vy, control_history[1].vy, control_history[2].vy, control_history[3].vy),
    )
    apply_over_axis(
        control_sequence.wz,
        (control_history[0].wz, control_history[1].wz, control_history[2].wz, control_history[3].wz),
    )

    offset = 1 if settings.shift_control_sequence else 0
    control_history[0] = control_history[1]
    control_history[1] = control_history[2]
    control_history[2] = control_history[3]
    control_history[3] = Control(
        vx=float(control_sequence.vx[offset]),
        vy=float(control_sequence.vy[offset]),
        wz=float(control_sequence.wz[offset]),
    )


def build_path_from_xy(
    xs: Sequence[float],
    ys: Sequence[float],
    yaws: Sequence[float] | None = None,
    frame_id: str = "map",
) -> Path:
    if yaws is None:
        yaws = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    path = Path()
    path.header = Header(stamp=Time(), frame_id=frame_id)
    for x, y, yaw in zip(xs, ys, yaws):
        pose_stamped = PoseStamped()
        pose_stamped.header = Header(stamp=Time(), frame_id=frame_id)
        pose_stamped.pose.position = Point(x=x, y=y, z=0.0)
        pose_stamped.pose.orientation = quaternion_from_yaw(float(yaw))
        path.poses.append(pose_stamped)
    return path


def sample_path_indices(path: PathTensor, time_steps: int) -> np.ndarray:
    if path.x.shape[0] == 0:
        return np.zeros(time_steps, dtype=np.int32)
    return np.linspace(0, path.x.shape[0] - 1, num=time_steps).astype(np.int32)


def goal_distance(goal: Pose, position: np.ndarray) -> np.ndarray:
    dx = position[:, 0] - goal.position.x
    dy = position[:, 1] - goal.position.y
    return np.hypot(dx, dy)


def within_position_goal_tolerance(
    goal_checker: Optional["GoalChecker"], robot: Pose, goal: Pose
) -> bool:
    """Check if the robot pose is within the Goal Checker's tolerances to goal."""
    if goal_checker is not None:
        from ..messages import Pose as PoseMsg, Twist

        pose_tolerance = PoseMsg()
        velocity_tolerance = Twist()
        goal_checker.get_tolerances(pose_tolerance, velocity_tolerance)

        pose_tolerance_sq = pose_tolerance.position.x * pose_tolerance.position.x

        dx = robot.position.x - goal.position.x
        dy = robot.position.y - goal.position.y

        dist_sq = dx * dx + dy * dy

        if dist_sq < pose_tolerance_sq:
            return True

    return False


def within_position_goal_tolerance_scalar(
    pose_tolerance: float, robot: Pose, goal: Pose
) -> bool:
    """Check if the robot pose is within tolerance to the goal."""
    dist_sq = (goal.position.x - robot.position.x) ** 2 + (goal.position.y - robot.position.y) ** 2

    pose_tolerance_sq = pose_tolerance * pose_tolerance

    if dist_sq < pose_tolerance_sq:
        return True

    return False


def find_path_furthest_reached_point(data: CriticData) -> int:
    """Evaluate furthest point idx of data.path which is nearest to some trajectory in data.trajectories."""
    traj_x = data.trajectories.x[:, -1:]  # Shape: (batch_size, 1)
    traj_y = data.trajectories.y[:, -1:]  # Shape: (batch_size, 1)

    # Broadcast to compute distances
    dx = data.path.x - traj_x  # Shape: (batch_size, path_length)
    dy = data.path.y - traj_y  # Shape: (batch_size, path_length)

    dists = dx * dx + dy * dy  # Shape: (batch_size, path_length)

    max_id_by_trajectories = 0
    min_distance_by_path = float("inf")

    for i in range(dists.shape[0]):  # For each trajectory
        min_id_by_path = 0
        min_distance_by_path = float("inf")
        for j in range(dists.shape[1]):  # For each path point
            cur_dist = dists[i, j]
            if cur_dist < min_distance_by_path:
                min_distance_by_path = cur_dist
                min_id_by_path = j
        max_id_by_trajectories = max(max_id_by_trajectories, min_id_by_path)

    return max_id_by_trajectories


def find_path_trajectory_initial_point(data: CriticData) -> int:
    """Evaluate closest point idx of data.path which is nearest to the start of the trajectory in data.trajectories."""
    # First point should be the same for all trajectories from initial conditions
    dx = data.path.x - data.trajectories.x[0, 0]
    dy = data.path.y - data.trajectories.y[0, 0]
    dists = dx * dx + dy * dy

    min_distance_by_path = float("inf")
    min_id = 0
    for j in range(len(dists)):
        if dists[j] < min_distance_by_path:
            min_distance_by_path = dists[j]
            min_id = j

    return min_id


def set_path_furthest_point_if_not_set(data: CriticData) -> None:
    """Evaluate path furthest point if it is not set."""
    if data.furthest_reached_path_point is None:
        data.furthest_reached_path_point = find_path_furthest_reached_point(data)


def pose_point_angle(pose: Pose, point_x: float, point_y: float, forward_preference: bool) -> float:
    """Evaluate angle from pose (have angle) to point (no angle)."""
    pose_x = pose.position.x
    pose_y = pose.position.y
    pose_yaw = yaw_from_quaternion(pose.orientation)

    yaw = math.atan2(point_y - pose_y, point_x - pose_x)

    # If no preference for forward, return smallest angle either in heading or 180 of heading
    if not forward_preference:
        angle1 = abs(normalize_angle(yaw - pose_yaw))
        angle2 = abs(normalize_angle(yaw - normalize_angle(pose_yaw + math.pi)))
        return min(angle1, angle2)

    return abs(normalize_angle(yaw - pose_yaw))


def find_closest_path_pt(
    vec: List[float] | np.ndarray, dist: float, init: int = 0
) -> int:
    """Compare to trajectory points to find closest path point along integrated distances."""
    vec_array = np.asarray(vec)
    if init >= len(vec_array) or len(vec_array) == 0:
        return 0

    # Find the insertion point using binary search
    search_vec = vec_array[init:]
    if len(search_vec) == 0:
        return 0

    idx = np.searchsorted(search_vec, dist)

    if idx == 0:
        # TODO/FIXME: Current behavior returns 0 when idx equals init.
        # This is incorrect when init > 0 — it should return init.
        return 0

    if idx >= len(search_vec):
        # Handle case where dist is beyond the last element
        return len(vec_array) - 1

    # Check which point is closer
    if dist - search_vec[idx - 1] < search_vec[idx] - dist:
        return init + idx - 1
    return init + idx


def find_first_path_inversion(path: Path) -> int:
    """Find the iterator of the first pose at which there is an inversion on the path."""
    # At least 3 poses for a possible inversion
    if len(path.poses) < 3:
        return len(path.poses)

    # Iterating through the path to determine the position of the path inversion
    for idx in range(1, len(path.poses) - 1):
        # We have two vectors for the dot product OA and AB. Determining the vectors.
        oa_x = path.poses[idx].pose.position.x - path.poses[idx - 1].pose.position.x
        oa_y = path.poses[idx].pose.position.y - path.poses[idx - 1].pose.position.y
        ab_x = path.poses[idx + 1].pose.position.x - path.poses[idx].pose.position.x
        ab_y = path.poses[idx + 1].pose.position.y - path.poses[idx].pose.position.y

        # Checking for the existence of cusp, in the path, using the dot product.
        dot_product = oa_x * ab_x + oa_y * ab_y
        if dot_product < 0.0:
            return idx + 1

    return len(path.poses)


def remove_poses_after_first_inversion(path: Path) -> int:
    """Find and remove poses after the first inversion in the path."""
    first_after_inversion = find_first_path_inversion(path)
    if first_after_inversion == len(path.poses):
        return 0

    # Remove poses after the inversion
    path.poses = path.poses[:first_after_inversion]
    return first_after_inversion

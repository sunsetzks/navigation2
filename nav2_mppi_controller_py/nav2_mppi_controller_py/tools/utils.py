from __future__ import annotations

import math
from typing import Iterable, List, Sequence

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

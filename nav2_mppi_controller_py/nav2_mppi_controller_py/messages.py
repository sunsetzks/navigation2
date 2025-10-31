from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class Time:
    sec: int = 0
    nanosec: int = 0


@dataclass
class Header:
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def quaternion_from_yaw(yaw: float) -> Quaternion:
    half = yaw * 0.5
    return Quaternion(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half))


def yaw_from_quaternion(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))


@dataclass
class Pose:
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseStamped:
    header: Header = field(default_factory=Header)
    pose: Pose = field(default_factory=Pose)


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Twist:
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)


@dataclass
class TwistStamped:
    header: Header = field(default_factory=Header)
    twist: Twist = field(default_factory=Twist)


@dataclass
class ColorRGBA:
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    a: float = 1.0


@dataclass
class Marker:
    header: Header = field(default_factory=Header)
    ns: str = ""
    id: int = 0
    type: int = 0
    action: int = 0
    pose: Pose = field(default_factory=Pose)
    scale: Vector3 = field(default_factory=Vector3)
    color: ColorRGBA = field(default_factory=ColorRGBA)

    SPHERE = 2
    ADD = 0


@dataclass
class MarkerArray:
    markers: List[Marker] = field(default_factory=list)


@dataclass
class Path:
    header: Header = field(default_factory=Header)
    poses: List[PoseStamped] = field(default_factory=list)

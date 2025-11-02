import math
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np


class CarVisualizer:
    """Visualization class for car with wheels and steering."""

    def __init__(self, vehicle_length: float = 0.5, vehicle_width: float = 0.3,
                 wheel_width: float = 0.04, wheel_length: float = 0.12):
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.wheel_width = wheel_width
        self.wheel_length = wheel_length

        # Wheel colors: rear left, rear right, front left, front right
        self.wheel_colors = ['dimgray', 'dimgray', 'black', 'black']

        # Initialize plot elements
        self.vehicle_body = None
        self.wheels = []
        self.wheelbase_line = None
        self.steering_angle = 0.0

    def get_vehicle_corners(self, x: float, y: float, yaw: float) -> List[Tuple[float, float]]:
        """Calculate the four corners of the vehicle rectangle."""
        half_l = self.vehicle_length / 2
        half_w = self.vehicle_width / 2
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        corners = [
            (x + half_l * cos_yaw - half_w * sin_yaw, y + half_l * sin_yaw + half_w * cos_yaw),  # front right
            (x + half_l * cos_yaw + half_w * sin_yaw, y + half_l * sin_yaw - half_w * cos_yaw),  # front left
            (x - half_l * cos_yaw + half_w * sin_yaw, y - half_l * sin_yaw - half_w * cos_yaw),  # rear left
            (x - half_l * cos_yaw - half_w * sin_yaw, y - half_l * sin_yaw + half_w * cos_yaw),  # rear right
        ]
        return corners

    def get_wheel_corners(self, x: float, y: float, yaw: float, steering_angle: float = 0.0) -> List[List[Tuple[float, float]]]:
        """Calculate the corners for all four wheels.
        
        Wheel coordinate system:
        - wheel_length is along the rolling direction (forward/backward)
        - wheel_width is perpendicular to rolling direction (left/right)
        """
        half_l = self.vehicle_length / 2
        half_w = self.vehicle_width / 2
        wheel_half_w = self.wheel_width / 2
        wheel_half_l = self.wheel_length / 2

        # Wheel mounting positions relative to vehicle center (in vehicle frame)
        # These are the centers where each wheel is attached
        wheel_positions = [
            (-half_l * 0.8, -half_w * 0.9),  # rear left
            (-half_l * 0.8, half_w * 0.9),   # rear right
            (half_l * 0.8, -half_w * 0.9),   # front left
            (half_l * 0.8, half_w * 0.9),    # front right
        ]

        wheel_corners = []
        for i, (wx_rel, wy_rel) in enumerate(wheel_positions):
            # Transform wheel center from vehicle frame to world frame
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            wx_world = x + wx_rel * cos_yaw - wy_rel * sin_yaw
            wy_world = y + wx_rel * sin_yaw + wy_rel * cos_yaw

            # Determine wheel orientation: vehicle yaw + steering angle for front wheels (indices 2, 3)
            wheel_yaw = yaw + (steering_angle if i >= 2 else 0.0)
            cos_wheel = math.cos(wheel_yaw)
            sin_wheel = math.sin(wheel_yaw)

            # Calculate wheel rectangle corners around the wheel center
            # The wheel length is in the rolling direction (aligned with wheel_yaw)
            # The wheel width is perpendicular to rolling direction
            corners = [
                (wx_world + wheel_half_l * cos_wheel - wheel_half_w * sin_wheel,
                 wy_world + wheel_half_l * sin_wheel + wheel_half_w * cos_wheel),
                (wx_world + wheel_half_l * cos_wheel + wheel_half_w * sin_wheel,
                 wy_world + wheel_half_l * sin_wheel - wheel_half_w * cos_wheel),
                (wx_world - wheel_half_l * cos_wheel + wheel_half_w * sin_wheel,
                 wy_world - wheel_half_l * sin_wheel - wheel_half_w * cos_wheel),
                (wx_world - wheel_half_l * cos_wheel - wheel_half_w * sin_wheel,
                 wy_world - wheel_half_l * sin_wheel + wheel_half_w * cos_wheel),
            ]
            wheel_corners.append(corners)

        return wheel_corners

    def initialize_plot(self, ax: Any) -> None:
        """Initialize plot elements for the car visualization."""
        # Vehicle body
        self.vehicle_body = ax.fill([], [], color='red', alpha=0.5, label="Vehicle")[0]

        # Wheelbase line (dotted line connecting front and rear axles)
        self.wheelbase_line, = ax.plot([], [], 'b:', linewidth=2, label="Wheelbase", alpha=0.7)

        # Wheels
        self.wheels = []
        for color in self.wheel_colors:
            wheel = ax.fill([], [], color=color, alpha=0.8, label=f"Wheel {len(self.wheels)+1}")
            self.wheels.append(wheel[0])

    def update_plot(self, x: float, y: float, yaw: float, steering_angle: float = 0.0) -> List:
        """Update the car visualization for the given position and orientation."""
        # Update vehicle body
        corners = self.get_vehicle_corners(x, y, yaw)
        corners_array = np.array(corners)
        self.vehicle_body.set_xy(corners_array)

        # Update wheelbase line (from rear axle to front axle)
        half_l = self.vehicle_length / 2
        rear_axle_x = x - (half_l * 0.8) * math.cos(yaw)
        rear_axle_y = y - (half_l * 0.8) * math.sin(yaw)
        front_axle_x = x + (half_l * 0.8) * math.cos(yaw)
        front_axle_y = y + (half_l * 0.8) * math.sin(yaw)
        self.wheelbase_line.set_data([rear_axle_x, front_axle_x], [rear_axle_y, front_axle_y])

        # Update wheels
        wheel_corners = self.get_wheel_corners(x, y, yaw, steering_angle)
        plot_elements = [self.vehicle_body, self.wheelbase_line]

        for wheel, corners in zip(self.wheels, wheel_corners):
            corners_array = np.array(corners)
            wheel.set_xy(corners_array)
            plot_elements.append(wheel)

        return plot_elements

    def calculate_steering_angle(self, wz: float, vx: float, wheelbase: float = 0.5) -> float:
        """Calculate steering angle from angular velocity and linear velocity."""
        if abs(vx) < 1e-6:
            return 0.0
        # Steering angle = atan(wheelbase * wz / vx)
        return math.atan(wheelbase * wz / vx)
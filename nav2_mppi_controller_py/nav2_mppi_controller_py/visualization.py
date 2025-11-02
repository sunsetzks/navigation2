import math
from typing import List, Tuple, Any, Dict

import matplotlib.pyplot as plt
import numpy as np


# Car style configurations
CAR_STYLES = {
    "default": {
        "vehicle_length": 0.5,
        "vehicle_width": 0.3,
        "wheel_width": 0.04,
        "wheel_length": 0.12,
        "wheelbase_fraction": 0.8,
        "track_fraction": 0.9,
        "body_color": "red",
        "body_alpha": 0.5,
        "rear_wheel_color": "dimgray",
        "front_wheel_color": "black",
        "wheel_alpha": 0.8,
        "wheelbase_color": "blue",
        "wheelbase_style": "--",
        "wheelbase_alpha": 0.7,
        "wheelbase_linewidth": 2.5,
    },
    "sedan": {
        "vehicle_length": 0.45,
        "vehicle_width": 0.25,
        "wheel_width": 0.035,
        "wheel_length": 0.1,
        "wheelbase_fraction": 0.75,
        "track_fraction": 0.85,
        "body_color": "#1f77b4",  # steel blue
        "body_alpha": 0.6,
        "rear_wheel_color": "black",
        "front_wheel_color": "black",
        "wheel_alpha": 0.9,
        "wheelbase_color": "cyan",
        "wheelbase_style": "--",
        "wheelbase_alpha": 0.75,
        "wheelbase_linewidth": 2.5,
    },
    "truck": {
        "vehicle_length": 0.6,
        "vehicle_width": 0.35,
        "wheel_width": 0.05,
        "wheel_length": 0.14,
        "wheelbase_fraction": 0.8,
        "track_fraction": 0.95,
        "body_color": "#ff7f0e",  # orange
        "body_alpha": 0.55,
        "rear_wheel_color": "darkslategray",
        "front_wheel_color": "darkslategray",
        "wheel_alpha": 0.85,
        "wheelbase_color": "darkorange",
        "wheelbase_style": "--",
        "wheelbase_alpha": 0.7,
        "wheelbase_linewidth": 2.5,
    },
    "sports": {
        "vehicle_length": 0.48,
        "vehicle_width": 0.28,
        "wheel_width": 0.038,
        "wheel_length": 0.11,
        "wheelbase_fraction": 0.72,
        "track_fraction": 0.88,
        "body_color": "#d62728",  # red
        "body_alpha": 0.65,
        "rear_wheel_color": "black",
        "front_wheel_color": "black",
        "wheel_alpha": 0.95,
        "wheelbase_color": "red",
        "wheelbase_style": "--",
        "wheelbase_alpha": 0.75,
        "wheelbase_linewidth": 2.5,
    },
    "compact": {
        "vehicle_length": 0.35,
        "vehicle_width": 0.22,
        "wheel_width": 0.032,
        "wheel_length": 0.09,
        "wheelbase_fraction": 0.75,
        "track_fraction": 0.82,
        "body_color": "#2ca02c",  # green
        "body_alpha": 0.6,
        "rear_wheel_color": "#333333",
        "front_wheel_color": "#333333",
        "wheel_alpha": 0.88,
        "wheelbase_color": "lime",
        "wheelbase_style": "--",
        "wheelbase_alpha": 0.7,
        "wheelbase_linewidth": 2.5,
    },
}


class CarVisualizer:
    """Visualization class for car with wheels and steering.
    
    Supports multiple car styles: 'default', 'sedan', 'truck', 'sports', 'compact'
    """

    def __init__(self, style: str = "default", 
                 vehicle_length: float | None = None, vehicle_width: float | None = None,
                 wheel_width: float | None = None, wheel_length: float | None = None,
                 wheelbase_length: float | None = None):
        """Initialize the car visualizer with a specific style.
        
        Args:
            style: Car style name ('default', 'sedan', 'truck', 'sports', 'compact')
            vehicle_length: Override style's vehicle length
            vehicle_width: Override style's vehicle width
            wheel_width: Override style's wheel width
            wheel_length: Override style's wheel length
            wheelbase_length: Override style's wheelbase length (distance between front/rear axles)
        """
        if style not in CAR_STYLES:
            raise ValueError(f"Unknown car style: {style}. Available: {list(CAR_STYLES.keys())}")
        
        # Load style configuration
        config = CAR_STYLES[style].copy()
        self.style = style
        
        # Override with user-provided values
        self.vehicle_length = vehicle_length or config["vehicle_length"]
        self.vehicle_width = vehicle_width or config["vehicle_width"]
        self.wheel_width = wheel_width or config["wheel_width"]
        self.wheel_length = wheel_length or config["wheel_length"]
        
        # Calculate wheelbase length (use provided value or fraction of vehicle length)
        self.wheelbase_length = wheelbase_length or (self.vehicle_length * config["wheelbase_fraction"])
        
        # Style parameters
        self.wheelbase_fraction = config["wheelbase_fraction"]  # Keep for backward compatibility
        self.track_fraction = config["track_fraction"]
        self.body_color = config["body_color"]
        self.body_alpha = config["body_alpha"]
        self.rear_wheel_color = config["rear_wheel_color"]
        self.front_wheel_color = config["front_wheel_color"]
        self.wheel_alpha = config["wheel_alpha"]
        self.wheelbase_color = config["wheelbase_color"]
        self.wheelbase_style = config["wheelbase_style"]
        self.wheelbase_alpha = config["wheelbase_alpha"]
        self.wheelbase_linewidth = config.get("wheelbase_linewidth", 2.5)
        
        # Wheel colors: rear left, rear right, front left, front right
        self.wheel_colors = [self.rear_wheel_color, self.rear_wheel_color,
                            self.front_wheel_color, self.front_wheel_color]

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
            (-self.wheelbase_length / 2, -half_w * self.track_fraction),  # rear left
            (-self.wheelbase_length / 2, half_w * self.track_fraction),   # rear right
            (self.wheelbase_length / 2, -half_w * self.track_fraction),   # front left
            (self.wheelbase_length / 2, half_w * self.track_fraction),    # front right
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
        self.vehicle_body = ax.fill([], [], color=self.body_color, alpha=self.body_alpha, label="Vehicle")[0]

        # Wheelbase line (dash-dot line connecting front and rear axles)
        self.wheelbase_line, = ax.plot([], [], color=self.wheelbase_color, linestyle=self.wheelbase_style, 
                                       linewidth=self.wheelbase_linewidth, label="Wheelbase", alpha=self.wheelbase_alpha)

        # Wheels
        self.wheels = []
        for i, color in enumerate(self.wheel_colors):
            wheel = ax.fill([], [], color=color, alpha=self.wheel_alpha, label=f"Wheel {i+1}")
            self.wheels.append(wheel[0])

    def update_plot(self, x: float, y: float, yaw: float, steering_angle: float = 0.0) -> List:
        """Update the car visualization for the given position and orientation."""
        # Update vehicle body
        corners = self.get_vehicle_corners(x, y, yaw)
        corners_array = np.array(corners)
        self.vehicle_body.set_xy(corners_array)

        # Update wheelbase line (from rear axle to front axle)
        rear_axle_x = x - (self.wheelbase_length / 2) * math.cos(yaw)
        rear_axle_y = y - (self.wheelbase_length / 2) * math.sin(yaw)
        front_axle_x = x + (self.wheelbase_length / 2) * math.cos(yaw)
        front_axle_y = y + (self.wheelbase_length / 2) * math.sin(yaw)
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
#!/usr/bin/env python3
"""Test script to visualize steering wheel rotation."""

import matplotlib.pyplot as plt
import numpy as np
from nav2_mppi_controller_py.visualization import CarVisualizer

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Test different steering angles (positive = left turn)
steering_angles = [0.0, 0.4, -0.4]
titles = ["No Steering (0°)", "Left Turn +23° (Positive)", "Right Turn -23° (Negative)"]

for ax, steering_angle, title in zip(axes, steering_angles, titles):
    car_viz = CarVisualizer(vehicle_length=0.5, vehicle_width=0.3,
                           wheel_width=0.04, wheel_length=0.12)
    
    # Initialize the plot with the visualizer
    car_viz.initialize_plot(ax)
    
    # Vehicle at origin, pointing up (yaw=pi/2)
    x, y, yaw = 0.0, 0.0, np.pi/2
    
    # Update the visualization (this will draw vehicle, wheels, and wheelbase line)
    car_viz.update_plot(x, y, yaw, steering_angle)
    
    # Mark vehicle center and direction
    ax.plot(x, y, 'r*', markersize=15, label='Vehicle Center')
    ax.arrow(x, y, 0.15*np.cos(yaw), 0.15*np.sin(yaw), 
             head_width=0.04, head_length=0.04, fc='red', ec='red', linewidth=2)
    
    # Draw wheel orientation lines for front wheels
    wheel_corners = car_viz.get_wheel_corners(x, y, yaw, steering_angle)
    for i in range(2, 4):  # front wheels only (indices 2, 3)
        corners_array = np.array(wheel_corners[i])
        center_x = np.mean(corners_array[:, 0])
        center_y = np.mean(corners_array[:, 1])
        wheel_yaw = yaw + steering_angle
        line_len = 0.06
        ax.plot([center_x, center_x + line_len * np.cos(wheel_yaw)],
               [center_y, center_y + line_len * np.sin(wheel_yaw)],
               'r-', linewidth=2, alpha=0.8)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

plt.suptitle('Steering Visualization Test: Positive = Left Turn, Negative = Right Turn', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

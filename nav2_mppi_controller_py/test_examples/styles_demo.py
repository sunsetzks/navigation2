#!/usr/bin/env python3
"""Demo script to show all available car styles."""

import matplotlib.pyplot as plt
import numpy as np
from nav2_mppi_controller_py.visualization import CarVisualizer, CAR_STYLES

# Create figure with subplots for each style
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

styles = ["default", "sedan", "truck", "sports", "compact"]
empty_ax = axes[-1]

for idx, (ax, style) in enumerate(zip(axes[:-1], styles)):
    car_viz = CarVisualizer(style=style)
    car_viz.initialize_plot(ax)
    
    # Vehicle at origin, pointing up (yaw=pi/2)
    x, y, yaw = 0.0, 0.0, np.pi/2
    
    # Show 3 steering angles
    steering_angles = [0.0, 0.35, -0.35]
    colors_trans = [1.0, 0.6, 0.6]
    
    for steering_angle, alpha_factor in zip(steering_angles, colors_trans):
        # Create temporary visualizer for each steering angle
        viz_temp = CarVisualizer(style=style)
        
        # Get the geometry but don't use initialize_plot
        vehicle_corners = viz_temp.get_vehicle_corners(x, y, yaw)
        vehicle_array = np.array(vehicle_corners)
        ax.fill(vehicle_array[:, 0], vehicle_array[:, 1], 
               color=viz_temp.body_color, alpha=viz_temp.body_alpha * alpha_factor)
        
        # Draw wheelbase
        half_l = viz_temp.vehicle_length / 2
        rear_axle_x = x - (half_l * viz_temp.wheelbase_fraction) * np.cos(yaw)
        rear_axle_y = y - (half_l * viz_temp.wheelbase_fraction) * np.sin(yaw)
        front_axle_x = x + (half_l * viz_temp.wheelbase_fraction) * np.cos(yaw)
        front_axle_y = y + (half_l * viz_temp.wheelbase_fraction) * np.sin(yaw)
        ax.plot([rear_axle_x, front_axle_x], [rear_axle_y, front_axle_y],
               color=viz_temp.wheelbase_color, linestyle=viz_temp.wheelbase_style,
               linewidth=2, alpha=viz_temp.wheelbase_alpha * alpha_factor)
        
        # Draw wheels
        wheel_corners = viz_temp.get_wheel_corners(x, y, yaw, steering_angle)
        for i, (corners, wheel_color) in enumerate(zip(wheel_corners, viz_temp.wheel_colors)):
            corners_array = np.array(corners)
            ax.fill(corners_array[:, 0], corners_array[:, 1], 
                   color=wheel_color, alpha=viz_temp.wheel_alpha * alpha_factor)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{style.upper()}\nL={car_viz.vehicle_length:.2f}m, W={car_viz.vehicle_width:.2f}m", 
                fontsize=11, fontweight='bold')
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)

# Add legend in the empty subplot
ax = empty_ax
ax.axis('off')
legend_text = "Car Style Comparison\n\n"
legend_text += "Style Features:\n"
for style in styles:
    config = CAR_STYLES[style]
    legend_text += f"\n{style.upper()}:\n"
    legend_text += f"  Length: {config['vehicle_length']:.2f}m\n"
    legend_text += f"  Width: {config['vehicle_width']:.2f}m\n"
    legend_text += f"  Body: {config['body_color']}\n"
    legend_text += f"  Wheels: {config['rear_wheel_color']}\n"

ax.text(0.1, 0.95, legend_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Car Visualization Styles: Full Steering Range (-35° to +35°)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

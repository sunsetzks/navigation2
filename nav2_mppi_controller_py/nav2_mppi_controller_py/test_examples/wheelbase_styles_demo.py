#!/usr/bin/env python3
"""Demo to compare different wheelbase line styles."""

import matplotlib.pyplot as plt
import numpy as np
from nav2_mppi_controller_py.visualization import CarVisualizer

# Different line styles to test
line_styles = {
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dash-dot": "-.",
    "thick solid": "-",
    "wide dashed": "--",
}

line_widths = {
    "solid": 2,
    "dashed": 2,
    "dotted": 2.5,
    "dash-dot": 2.5,
    "thick solid": 3,
    "wide dashed": 2.5,
}

colors = {
    "solid": "blue",
    "dashed": "darkblue",
    "dotted": "cyan",
    "dash-dot": "deepskyblue",
    "thick solid": "navy",
    "wide dashed": "steelblue",
}

alphas = {
    "solid": 0.7,
    "dashed": 0.75,
    "dotted": 0.8,
    "dash-dot": 0.75,
    "thick solid": 0.6,
    "wide dashed": 0.7,
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

styles_list = list(line_styles.keys())

for idx, (ax, style_name) in enumerate(zip(axes, styles_list)):
    # Create visualizer with default style
    car_viz = CarVisualizer(style="default")
    
    # Vehicle trajectory (arc path)
    t = np.linspace(0, np.pi, 5)
    positions = np.array([[2*np.cos(t_i), 2*np.sin(t_i)] for t_i in t])
    yaws = t + np.pi/2
    
    # Draw trajectory
    ax.plot(positions[:, 0], positions[:, 1], 'gray', alpha=0.3, linewidth=2, label='Trajectory')
    
    # Draw car at each position with wheelbase line
    for i, (pos, yaw) in enumerate(zip(positions, yaws)):
        x, y = pos
        
        # Get vehicle and wheelbase
        vehicle_corners = car_viz.get_vehicle_corners(x, y, yaw)
        vehicle_array = np.array(vehicle_corners)
        alpha_factor = 0.3 + (i / len(positions)) * 0.4
        ax.fill(vehicle_array[:, 0], vehicle_array[:, 1], 
               color='red', alpha=0.3 * alpha_factor)
        
        # Draw wheelbase with the current style
        half_l = car_viz.vehicle_length / 2
        rear_axle_x = x - (half_l * car_viz.wheelbase_fraction) * np.cos(yaw)
        rear_axle_y = y - (half_l * car_viz.wheelbase_fraction) * np.sin(yaw)
        front_axle_x = x + (half_l * car_viz.wheelbase_fraction) * np.cos(yaw)
        front_axle_y = y + (half_l * car_viz.wheelbase_fraction) * np.sin(yaw)
        
        ax.plot([rear_axle_x, front_axle_x], [rear_axle_y, front_axle_y],
               color=colors[style_name], linestyle=line_styles[style_name],
               linewidth=line_widths[style_name], alpha=alphas[style_name])
        
        # Draw wheels
        wheel_corners = car_viz.get_wheel_corners(x, y, yaw, 0.0)
        for corners in wheel_corners:
            corners_array = np.array(corners)
            ax.fill(corners_array[:, 0], corners_array[:, 1], 
                   color='black', alpha=0.3 * alpha_factor)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title(f"Style: {style_name.upper()}\n({line_styles[style_name]} line, width={line_widths[style_name]}, alpha={alphas[style_name]})",
                fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

plt.suptitle('Wheelbase Line Style Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Print recommendations
print("\n" + "="*60)
print("WHEELBASE LINE STYLE RECOMMENDATIONS")
print("="*60)
print("\n🏆 RECOMMENDED: 'dash-dot' (-.) with width=2.5")
print("   - Best visibility and clarity")
print("   - Distinguishes wheelbase from trajectory")
print("   - Professional appearance")
print("\n✅ ALTERNATIVES:")
print("   - 'dashed' (--): Classic, clear, traditional")
print("   - 'solid' (-): Simple, bold, always visible")
print("   - 'dotted' (:): Subtle, minimal interference")
print("\n❌ AVOID:")
print("   - 'solid' at normal thickness: Too heavy")
print("   - 'dotted' at normal width: Too subtle on small visualizations")
print("="*60 + "\n")

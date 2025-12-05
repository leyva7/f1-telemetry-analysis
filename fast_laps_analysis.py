"""
F1 Fast Laps Analysis
Analyze and compare fastest laps from Formula 1 sessions using FastF1 library.
Identifies key points where time is gained or lost between the two fastest drivers.
"""

import fastf1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Enable caching
fastf1.Cache.enable_cache("cache")

# Load session
session = fastf1.get_session(2025, 'Monaco', 'Q')
session.load()

# Find the two fastest drivers automatically
print("Finding fastest drivers...")
fastest_laps = session.laps.groupby('Driver')['LapTime'].min().sort_values()
top_two_drivers = fastest_laps.head(2).index.tolist()

driver_1 = top_two_drivers[0]  # Fastest
driver_2 = top_two_drivers[1]  # Second fastest

print(f"Fastest driver: {driver_1} ({fastest_laps[driver_1]})")
print(f"Second fastest: {driver_2} ({fastest_laps[driver_2]})")
print(f"Time difference: {fastest_laps[driver_2] - fastest_laps[driver_1]}\n")

# Get fastest laps for both drivers
lap_driver_1 = session.laps.pick_driver(driver_1).pick_fastest()
lap_driver_2 = session.laps.pick_driver(driver_2).pick_fastest()

# Get telemetry data
tel_1 = lap_driver_1.get_car_data().add_distance()
tel_2 = lap_driver_2.get_car_data().add_distance()

# Interpolate to same distance points for comparison
min_distance = max(tel_1['Distance'].min(), tel_2['Distance'].min())
max_distance = min(tel_1['Distance'].max(), tel_2['Distance'].max())
common_distance = np.linspace(min_distance, max_distance, len(tel_1))

tel_1_interp = np.interp(common_distance, tel_1['Distance'], tel_1['Speed'])
tel_2_interp = np.interp(common_distance, tel_2['Distance'], tel_2['Speed'])

# Calculate speed difference (positive = driver_1 faster, negative = driver_2 faster)
speed_diff = tel_1_interp - tel_2_interp

# Identify key sections where significant time is gained/lost
# Time gain/loss is approximated by speed difference over distance
# We'll look for sections where the difference is significant
threshold = np.std(speed_diff) * 0.5  # Threshold for significant differences

# Find sections where driver_1 is significantly faster
driver_1_advantage = speed_diff > threshold
# Find sections where driver_2 is significantly faster
driver_2_advantage = speed_diff < -threshold

# Create comprehensive visualization
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle(f'Fast Lap Analysis: {driver_1} vs {driver_2} - {session.event["EventName"]} {session.event.year} Q', 
             fontsize=14, fontweight='bold')

# Plot 1: Speed comparison
ax1 = axes[0]
ax1.plot(tel_1['Distance'], tel_1['Speed'], label=driver_1, linewidth=2, color='blue')
ax1.plot(tel_2['Distance'], tel_2['Speed'], label=driver_2, linewidth=2, color='red')
ax1.set_ylabel('Speed [km/h]', fontweight='bold')
ax1.set_title('Speed Comparison', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Speed difference (where time is gained/lost)
ax2 = axes[1]
ax2.plot(common_distance, speed_diff, linewidth=2, color='purple', alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.fill_between(common_distance, 0, speed_diff, where=(speed_diff > 0), 
                 alpha=0.3, color='green', label=f'{driver_1} advantage')
ax2.fill_between(common_distance, 0, speed_diff, where=(speed_diff < 0), 
                 alpha=0.3, color='orange', label=f'{driver_2} advantage')
ax2.set_ylabel('Speed Difference [km/h]', fontweight='bold')
ax2.set_title(f'Where Time is Gained/Lost (Positive = {driver_1} faster)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Brake comparison (braking points are critical)
ax3 = axes[2]
ax3.plot(tel_1['Distance'], tel_1['Brake'], label=driver_1, linewidth=2, alpha=0.7, color='blue')
ax3.plot(tel_2['Distance'], tel_2['Brake'], label=driver_2, linewidth=2, alpha=0.7, color='red')
ax3.set_ylabel('Brake [%]', fontweight='bold')
ax3.set_title('Braking Points Comparison', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Throttle comparison (acceleration zones)
ax4 = axes[3]
ax4.plot(tel_1['Distance'], tel_1['Throttle'], label=driver_1, linewidth=2, alpha=0.7, color='blue')
ax4.plot(tel_2['Distance'], tel_2['Throttle'], label=driver_2, linewidth=2, alpha=0.7, color='red')
ax4.set_xlabel('Distance [m]', fontweight='bold')
ax4.set_ylabel('Throttle [%]', fontweight='bold')
ax4.set_title('Throttle Usage Comparison', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print key insights
print("=" * 60)
print("KEY INSIGHTS:")
print("=" * 60)

# Find largest speed differences (key gain/loss points)
max_advantage_d1_idx = np.argmax(speed_diff)
max_advantage_d2_idx = np.argmin(speed_diff)

print(f"\n📍 Largest {driver_1} advantage:")
print(f"   Distance: {common_distance[max_advantage_d1_idx]:.0f} m")
print(f"   Speed advantage: {speed_diff[max_advantage_d1_idx]:.1f} km/h")

print(f"\n📍 Largest {driver_2} advantage:")
print(f"   Distance: {common_distance[max_advantage_d2_idx]:.0f} m")
print(f"   Speed advantage: {-speed_diff[max_advantage_d2_idx]:.1f} km/h")

# Calculate approximate time gained/lost in key sections
# Simple approximation: time difference ≈ distance / average_speed
avg_speed_1 = np.mean(tel_1['Speed'])
avg_speed_2 = np.mean(tel_2['Speed'])

total_time_diff = (fastest_laps[driver_2] - fastest_laps[driver_1]).total_seconds()
print(f"\n⏱️  Total lap time difference: {total_time_diff:.3f} seconds")
print(f"   Average speed - {driver_1}: {avg_speed_1:.1f} km/h")
print(f"   Average speed - {driver_2}: {avg_speed_2:.1f} km/h")
print("=" * 60)

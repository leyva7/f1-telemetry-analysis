"""
F1 Fast Laps Analysis
Analyze and compare fastest laps from Formula 1 sessions using FastF1 library.
Identifies key points where time is gained or lost between the two fastest drivers.
"""

import fastf1
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import pandas as pd
import numpy as np
from pathlib import Path

# Enable caching using pathlib
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist
fastf1.Cache.enable_cache(CACHE_DIR)


def analyze_session(year, grand_prix, session_type='Q'):
    """
    Analyze and compare fastest laps from a Formula 1 session.
    
    Parameters:
    -----------
    year : int
        Year of the race (e.g., 2024, 2025)
    grand_prix : str
        Name of the Grand Prix (e.g., 'Monaco', 'Spanish Grand Prix')
    session_type : str
        Type of session: 'FP1', 'FP2', 'FP3', 'Q' (Qualifying), 'R' (Race)
        Default is 'Q' (Qualifying)
    
    Returns:
    --------
    dict : Dictionary containing session, drivers, and analysis data
    """
    # Load session
    print(f"Loading {year} {grand_prix} - {session_type} session...")
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()
    print(f"Session loaded successfully!\n")

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

    # Get position data (X, Y coordinates) for circuit map
    # X and Y coordinates are in the telemetry data
    pos_1 = lap_driver_1.get_telemetry()
    pos_2 = lap_driver_2.get_telemetry()

    # Interpolate to same distance points for comparison
    min_distance = max(tel_1['Distance'].min(), tel_2['Distance'].min())
    max_distance = min(tel_1['Distance'].max(), tel_2['Distance'].max())
    common_distance = np.linspace(min_distance, max_distance, len(tel_1))

    # Interpolate all telemetry channels to common distance
    tel_1_speed = np.interp(common_distance, tel_1['Distance'], tel_1['Speed'])
    tel_2_speed = np.interp(common_distance, tel_2['Distance'], tel_2['Speed'])
    tel_1_throttle = np.interp(common_distance, tel_1['Distance'], tel_1['Throttle'])
    tel_2_throttle = np.interp(common_distance, tel_2['Distance'], tel_2['Throttle'])
    tel_1_brake = np.interp(common_distance, tel_1['Distance'], tel_1['Brake'])
    tel_2_brake = np.interp(common_distance, tel_2['Distance'], tel_2['Brake'])

    # Calculate speed difference (positive = driver_1 faster, negative = driver_2 faster)
    speed_diff = tel_1_speed - tel_2_speed

    # Calculate Time Delta Accumulated (more precise time gain/loss calculation)
    # Convert speed from km/h to m/s
    tel_1_speed_ms = tel_1_speed / 3.6
    tel_2_speed_ms = tel_2_speed / 3.6

    # Calculate distance increments
    delta_distance = np.diff(common_distance, prepend=common_distance[0])

    # Calculate time taken for each segment (avoid division by zero)
    time_1 = np.where(tel_1_speed_ms > 0.1, delta_distance / tel_1_speed_ms, 0)
    time_2 = np.where(tel_2_speed_ms > 0.1, delta_distance / tel_2_speed_ms, 0)

    # Calculate time difference for each segment (positive = driver_1 faster, negative = driver_2 faster)
    # Note: We invert the sign because if driver_1 is faster, time_1 < time_2, so time_2 - time_1 > 0
    time_delta_segment = time_2 - time_1

    # Calculate cumulative time delta (accumulated time difference)
    time_delta_accumulated = np.cumsum(time_delta_segment)

    # Identify key sections where significant time is gained/lost
    # Time gain/loss is approximated by speed difference over distance
    # We'll look for sections where the difference is significant
    threshold = np.std(speed_diff) * 0.5  # Threshold for significant differences

    # Find sections where driver_1 is significantly faster
    driver_1_advantage = speed_diff > threshold
    # Find sections where driver_2 is significantly faster
    driver_2_advantage = speed_diff < -threshold

    # Create comprehensive visualization
    session_name = session.event["EventName"]
    fig, axes = plt.subplots(5, 1, figsize=(14, 14))
    fig.suptitle(f'Fast Lap Analysis: {driver_1} vs {driver_2} - {session_name} {session.event.year} {session_type}', 
                 fontsize=14, fontweight='bold')

    # Plot 1: Speed comparison
    ax1 = axes[0]
    ax1.plot(tel_1['Distance'], tel_1['Speed'], label=driver_1, linewidth=2, color='blue')
    ax1.plot(tel_2['Distance'], tel_2['Speed'], label=driver_2, linewidth=2, color='red')
    ax1.set_ylabel('Speed [km/h]', fontweight='bold')
    ax1.set_title('Speed Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time Delta Accumulated (THE GOLD STANDARD for F1 analysis)
    ax2 = axes[1]
    ax2.plot(common_distance, time_delta_accumulated * 1000, linewidth=2.5, color='darkblue')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(common_distance, 0, time_delta_accumulated * 1000, 
                     where=(time_delta_accumulated > 0), alpha=0.3, color='green', 
                     label=f'{driver_1} gaining time')
    ax2.fill_between(common_distance, 0, time_delta_accumulated * 1000, 
                     where=(time_delta_accumulated < 0), alpha=0.3, color='orange', 
                     label=f'{driver_2} gaining time')
    ax2.set_ylabel('Time Delta [ms]', fontweight='bold')
    ax2.set_title(f'Cumulative Time Delta (Positive = {driver_1} ahead)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Speed difference (where time is gained/lost)
    ax3 = axes[2]
    ax3.plot(common_distance, speed_diff, linewidth=2, color='purple', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.fill_between(common_distance, 0, speed_diff, where=(speed_diff > 0), 
                     alpha=0.3, color='green', label=f'{driver_1} advantage')
    ax3.fill_between(common_distance, 0, speed_diff, where=(speed_diff < 0), 
                     alpha=0.3, color='orange', label=f'{driver_2} advantage')
    ax3.set_ylabel('Speed Difference [km/h]', fontweight='bold')
    ax3.set_title(f'Speed Difference (Positive = {driver_1} faster)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Brake comparison (braking points are critical)
    ax4 = axes[3]
    ax4.plot(tel_1['Distance'], tel_1['Brake'], label=driver_1, linewidth=2, alpha=0.7, color='blue')
    ax4.plot(tel_2['Distance'], tel_2['Brake'], label=driver_2, linewidth=2, alpha=0.7, color='red')
    ax4.set_ylabel('Brake [%]', fontweight='bold')
    ax4.set_title('Braking Points Comparison', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Throttle comparison (acceleration zones)
    ax5 = axes[4]
    ax5.plot(tel_1['Distance'], tel_1['Throttle'], label=driver_1, linewidth=2, alpha=0.7, color='blue')
    ax5.plot(tel_2['Distance'], tel_2['Throttle'], label=driver_2, linewidth=2, alpha=0.7, color='red')
    ax5.set_xlabel('Distance [m]', fontweight='bold')
    ax5.set_ylabel('Throttle [%]', fontweight='bold')
    ax5.set_title('Throttle Usage Comparison', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot circuit maps
    print("\nGenerating circuit maps...")
    plot_circuit_map(pos_1, pos_2, tel_1['Speed'], tel_2['Speed'], 
                     driver_1, driver_2, session.event, session_type,
                     time_delta_accumulated, common_distance, tel_1['Distance'], tel_2['Distance'])

    # Print key insights with contextual information
    print("=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)

    # Find largest time delta points (most critical for understanding where time is gained/lost)
    max_time_gain_d1_idx = np.argmax(time_delta_accumulated)
    max_time_gain_d2_idx = np.argmin(time_delta_accumulated)

    # Find largest speed differences
    max_speed_adv_d1_idx = np.argmax(speed_diff)
    max_speed_adv_d2_idx = np.argmin(speed_diff)

    # Get sector information if available
    try:
        sector_1_dist = lap_driver_1['Sector1SessionTime']
        sector_2_dist = lap_driver_1['Sector2SessionTime']
        # Approximate sector boundaries (this is a simplification)
        # In reality, you'd need to map distance to sectors more precisely
        has_sector_info = True
    except:
        has_sector_info = False

    def get_track_context(distance, tel_1_brake, tel_2_brake, tel_1_throttle, tel_2_throttle, idx):
        """Provide contextual information about track position"""
        brake_1 = tel_1_brake[idx]
        brake_2 = tel_2_brake[idx]
        throttle_1 = tel_1_throttle[idx]
        throttle_2 = tel_2_throttle[idx]
        
        context = []
        if brake_1 > 50 or brake_2 > 50:
            context.append("Heavy braking zone")
        elif brake_1 > 20 or brake_2 > 20:
            context.append("Braking zone")
        elif throttle_1 > 80 and throttle_2 > 80:
            context.append("Full throttle section")
        elif throttle_1 < 50 or throttle_2 < 50:
            context.append("Cornering section")
        else:
            context.append("Acceleration zone")
        
        return ", ".join(context)

    print(f"\n📍 Largest {driver_1} TIME GAIN (most critical point):")
    dist_max_d1 = common_distance[max_time_gain_d1_idx]
    print(f"   Distance: {dist_max_d1:.0f} m")
    print(f"   Time gained: {time_delta_accumulated[max_time_gain_d1_idx] * 1000:.1f} ms")
    print(f"   Speed: {driver_1} = {tel_1_speed[max_time_gain_d1_idx]:.1f} km/h, "
          f"{driver_2} = {tel_2_speed[max_time_gain_d1_idx]:.1f} km/h")
    context_d1 = get_track_context(dist_max_d1, tel_1_brake, tel_2_brake, 
                                   tel_1_throttle, tel_2_throttle, max_time_gain_d1_idx)
    print(f"   Context: {context_d1}")
    print(f"   Brake: {driver_1} = {tel_1_brake[max_time_gain_d1_idx]:.0f}%, "
          f"{driver_2} = {tel_2_brake[max_time_gain_d1_idx]:.0f}%")
    print(f"   Throttle: {driver_1} = {tel_1_throttle[max_time_gain_d1_idx]:.0f}%, "
          f"{driver_2} = {tel_2_throttle[max_time_gain_d1_idx]:.0f}%")

    print(f"\n📍 Largest {driver_2} TIME GAIN (most critical point):")
    dist_max_d2 = common_distance[max_time_gain_d2_idx]
    print(f"   Distance: {dist_max_d2:.0f} m")
    print(f"   Time gained: {-time_delta_accumulated[max_time_gain_d2_idx] * 1000:.1f} ms")
    print(f"   Speed: {driver_1} = {tel_1_speed[max_time_gain_d2_idx]:.1f} km/h, "
          f"{driver_2} = {tel_2_speed[max_time_gain_d2_idx]:.1f} km/h")
    context_d2 = get_track_context(dist_max_d2, tel_1_brake, tel_2_brake, 
                                   tel_1_throttle, tel_2_throttle, max_time_gain_d2_idx)
    print(f"   Context: {context_d2}")
    print(f"   Brake: {driver_1} = {tel_1_brake[max_time_gain_d2_idx]:.0f}%, "
          f"{driver_2} = {tel_2_brake[max_time_gain_d2_idx]:.0f}%")
    print(f"   Throttle: {driver_1} = {tel_1_throttle[max_time_gain_d2_idx]:.0f}%, "
          f"{driver_2} = {tel_2_throttle[max_time_gain_d2_idx]:.0f}%")

    print(f"\n📍 Largest {driver_1} SPEED ADVANTAGE:")
    print(f"   Distance: {common_distance[max_speed_adv_d1_idx]:.0f} m")
    print(f"   Speed advantage: {speed_diff[max_speed_adv_d1_idx]:.1f} km/h")
    context_speed_d1 = get_track_context(common_distance[max_speed_adv_d1_idx], tel_1_brake, tel_2_brake, 
                                         tel_1_throttle, tel_2_throttle, max_speed_adv_d1_idx)
    print(f"   Context: {context_speed_d1}")

    print(f"\n📍 Largest {driver_2} SPEED ADVANTAGE:")
    print(f"   Distance: {common_distance[max_speed_adv_d2_idx]:.0f} m")
    print(f"   Speed advantage: {-speed_diff[max_speed_adv_d2_idx]:.1f} km/h")
    context_speed_d2 = get_track_context(common_distance[max_speed_adv_d2_idx], tel_1_brake, tel_2_brake, 
                                         tel_1_throttle, tel_2_throttle, max_speed_adv_d2_idx)
    print(f"   Context: {context_speed_d2}")

    # Calculate final statistics
    avg_speed_1 = np.mean(tel_1['Speed'])
    avg_speed_2 = np.mean(tel_2['Speed'])
    total_time_diff = (fastest_laps[driver_2] - fastest_laps[driver_1]).total_seconds()
    final_time_delta = time_delta_accumulated[-1] * 1000  # Convert to ms

    print(f"\n⏱️  LAP TIME SUMMARY:")
    print(f"   Total lap time difference: {total_time_diff:.3f} seconds ({total_time_diff * 1000:.1f} ms)")
    print(f"   Calculated time delta (end of lap): {final_time_delta:.1f} ms")
    print(f"   Average speed - {driver_1}: {avg_speed_1:.1f} km/h")
    print(f"   Average speed - {driver_2}: {avg_speed_2:.1f} km/h")
    print("=" * 70)
    
    # Return analysis results
    return {
        'session': session,
        'driver_1': driver_1,
        'driver_2': driver_2,
        'lap_driver_1': lap_driver_1,
        'lap_driver_2': lap_driver_2,
        'tel_1': tel_1,
        'tel_2': tel_2,
        'time_delta_accumulated': time_delta_accumulated,
        'speed_diff': speed_diff
    }


# Create circuit map visualization
def plot_circuit_map(pos_data_1, pos_data_2, tel_speed_1, tel_speed_2, 
                     driver_1, driver_2, session_info, session_type='Q',
                     time_delta_accumulated=None, common_distance=None,
                     tel_distance_1=None, tel_distance_2=None):
    """Plot circuit map with speed coloring for both drivers and time delta overlay"""
    
    # Get coordinates
    x1 = pos_data_1['X'].values
    y1 = pos_data_1['Y'].values
    x2 = pos_data_2['X'].values
    y2 = pos_data_2['Y'].values
    
    # Get speed data aligned with position data
    # Speed data comes from telemetry, need to align with position data
    speed1 = tel_speed_1.values if hasattr(tel_speed_1, 'values') else np.array(tel_speed_1)
    speed2 = tel_speed_2.values if hasattr(tel_speed_2, 'values') else np.array(tel_speed_2)
    
    # Interpolate speeds to match position data length if needed
    # Position data and speed data might have different sampling rates
    if len(speed1) != len(x1):
        speed1 = np.interp(np.linspace(0, 1, len(x1)), 
                          np.linspace(0, 1, len(speed1)), speed1)
    if len(speed2) != len(x2):
        speed2 = np.interp(np.linspace(0, 1, len(x2)), 
                          np.linspace(0, 1, len(speed2)), speed2)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Circuit Map: {driver_1} vs {driver_2} - {session_info["EventName"]} {session_info.year} {session_type}', 
                 fontsize=16, fontweight='bold')
    
    # Common colormap and normalization
    vmin = min(speed1.min(), speed2.min())
    vmax = max(speed1.max(), speed2.max())
    norm = plt.Normalize(vmin, vmax)
    colormap = mpl.cm.plasma
    
    # Plot driver 1
    points1 = np.array([x1, y1]).T.reshape(-1, 1, 2)
    segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
    lc1 = LineCollection(segments1, cmap=colormap, norm=norm, linestyle='-', linewidth=4)
    lc1.set_array(speed1)
    ax1.add_collection(lc1)
    ax1.plot(x1, y1, color='black', linestyle='-', linewidth=6, zorder=0, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f'{driver_1} - Speed Map', fontsize=14, fontweight='bold')
    
    # Add colorbar for driver 1
    cbar1 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax1, 
                         orientation='horizontal', pad=0.05, aspect=30)
    cbar1.set_label('Speed [km/h]', fontsize=10, fontweight='bold')
    
    # Plot driver 2
    points2 = np.array([x2, y2]).T.reshape(-1, 1, 2)
    segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
    lc2 = LineCollection(segments2, cmap=colormap, norm=norm, linestyle='-', linewidth=4)
    lc2.set_array(speed2)
    ax2.add_collection(lc2)
    ax2.plot(x2, y2, color='black', linestyle='-', linewidth=6, zorder=0, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'{driver_2} - Speed Map', fontsize=14, fontweight='bold')
    
    # Add colorbar for driver 2
    cbar2 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax2, 
                         orientation='horizontal', pad=0.05, aspect=30)
    cbar2.set_label('Speed [km/h]', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Also create a combined overlay map colored by time delta
    if time_delta_accumulated is not None and common_distance is not None:
        fig2, ax3 = plt.subplots(figsize=(14, 10))
        fig2.suptitle(f'Circuit Map Overlay: {driver_1} vs {driver_2} - {session_info["EventName"]} {session_info.year} {session_type}', 
                      fontsize=16, fontweight='bold')
        
        # Get distance data for position coordinates
        # We need to map time_delta to position coordinates
        # First, get distance from position telemetry if available
        if 'Distance' in pos_data_1.columns and 'Distance' in pos_data_2.columns:
            pos_dist_1 = pos_data_1['Distance'].values
            pos_dist_2 = pos_data_2['Distance'].values
        elif tel_distance_1 is not None and tel_distance_2 is not None:
            # Interpolate distance to position data length
            tel_dist_1_vals = tel_distance_1.values if hasattr(tel_distance_1, 'values') else np.array(tel_distance_1)
            tel_dist_2_vals = tel_distance_2.values if hasattr(tel_distance_2, 'values') else np.array(tel_distance_2)
            
            # Create normalized indices for interpolation
            idx_norm_1 = np.linspace(0, 1, len(x1))
            idx_norm_2 = np.linspace(0, 1, len(x2))
            tel_norm_1 = np.linspace(0, 1, len(tel_dist_1_vals))
            tel_norm_2 = np.linspace(0, 1, len(tel_dist_2_vals))
            
            pos_dist_1 = np.interp(idx_norm_1, tel_norm_1, tel_dist_1_vals)
            pos_dist_2 = np.interp(idx_norm_2, tel_norm_2, tel_dist_2_vals)
        else:
            # Fallback: use index as proxy for distance (normalized)
            pos_dist_1 = np.linspace(common_distance.min(), common_distance.max(), len(x1))
            pos_dist_2 = np.linspace(common_distance.min(), common_distance.max(), len(x2))
        
        # Interpolate time_delta_accumulated to position coordinates
        # Ensure we're within the common_distance range
        pos_dist_1_clipped = np.clip(pos_dist_1, common_distance.min(), common_distance.max())
        pos_dist_2_clipped = np.clip(pos_dist_2, common_distance.min(), common_distance.max())
        
        # Interpolate time delta for driver positions
        time_delta_1 = np.interp(pos_dist_1_clipped, common_distance, time_delta_accumulated)
        time_delta_2 = np.interp(pos_dist_2_clipped, common_distance, time_delta_accumulated)
        
        # Create segments for LineCollection
        points1 = np.array([x1, y1]).T.reshape(-1, 1, 2)
        segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
        points2 = np.array([x2, y2]).T.reshape(-1, 1, 2)
        segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
        
        # Create colormap: green (positive = driver_1 ahead), red (negative = driver_2 ahead)
        # Normalize time delta (positive = driver_1 ahead, negative = driver_2 ahead)
        # Use symmetric normalization around zero
        max_abs_td = max(abs(time_delta_accumulated.max()), abs(time_delta_accumulated.min()))
        vmin_td = -max_abs_td
        vmax_td = max_abs_td
        norm_td = plt.Normalize(vmin_td, vmax_td)
        
        # Use a diverging colormap: green for positive (driver_1 ahead), red for negative (driver_2 ahead)
        # RdYlGn_r gives us green (positive) to red (negative)
        colormap_td = mpl.cm.RdYlGn_r
        
        # Plot driver 1 track colored by time delta
        lc1_td = LineCollection(segments1, cmap=colormap_td, norm=norm_td, 
                               linestyle='-', linewidth=3.5, alpha=0.9)
        lc1_td.set_array(time_delta_1)
        ax3.add_collection(lc1_td)
        
        # Plot driver 2 track colored by time delta
        lc2_td = LineCollection(segments2, cmap=colormap_td, norm=norm_td, 
                               linestyle='--', linewidth=2.5, alpha=0.9)
        lc2_td.set_array(time_delta_2)
        ax3.add_collection(lc2_td)
        
        # Add background track outline
        ax3.plot(x1, y1, color='black', linestyle='-', linewidth=8, zorder=0, alpha=0.2)
        
        # Add colorbar
        cbar3 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm_td, cmap=colormap_td), ax=ax3, 
                            orientation='horizontal', pad=0.05, aspect=30)
        cbar3.set_label(f'Time Delta (Green = {driver_1} ahead, Red = {driver_2} ahead)', 
                       fontsize=10, fontweight='bold')
        # Format colorbar to show milliseconds
        ticks = cbar3.get_ticks()
        cbar3.set_ticks(ticks)
        cbar3.set_ticklabels([f'{x*1000:.0f} ms' for x in ticks])
        
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.legend([f'{driver_1} (solid)', f'{driver_2} (dashed)'], 
                  loc='upper right', fontsize=12, framealpha=0.9)
        ax3.set_title('Track Overlay - Colored by Time Delta', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    else:
        # Fallback to simple overlay if time delta data not available
        fig2, ax3 = plt.subplots(figsize=(14, 10))
        fig2.suptitle(f'Circuit Map Overlay: {driver_1} vs {driver_2} - {session_info["EventName"]} {session_info.year} {session_type}', 
                      fontsize=16, fontweight='bold')
        
        # Plot both tracks with different colors
        # Use solid line for driver_1 and dashed line for driver_2 to ensure both are visible when overlapping
        ax3.plot(x1, y1, color='blue', linewidth=3, label=driver_1, alpha=0.8, linestyle='-')
        ax3.plot(x2, y2, color='red', linewidth=2.5, label=driver_2, alpha=0.8, linestyle='--')
        
        # Add background track outline
        ax3.plot(x1, y1, color='black', linestyle='-', linewidth=8, zorder=0, alpha=0.2)
        
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax3.set_title('Track Overlay Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# Main execution - call the function with default parameters
if __name__ == "__main__":
    # Default analysis: 2025 Monaco Qualifying
    results = analyze_session(2025, 'Monaco', 'Q')
    
    # Example: You can also analyze other sessions like this:
    # results = analyze_session(2024, 'Spanish Grand Prix', 'Q')
    # results = analyze_session(2024, 'Monaco', 'R')  # Race session

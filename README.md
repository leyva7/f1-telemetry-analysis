# F1 Fast Laps Analysis

A Python project for analyzing Formula 1 fastest laps using the FastF1 library. This project allows you to compare telemetry data, sector times, and performance metrics between drivers.

## Project Structure

```
f1-telemetry-analysis/
├── src/                          # Source code
│   ├── __init__.py
│   └── fast_laps_analysis.py    # Main analysis script
├── data/                         # Data directory
│   └── cache/                   # FastF1 cache (auto-generated)
├── outputs/                     # Generated plots and visualizations
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## Features

- Load Formula 1 sessions (Practice, Qualifying, Race)
- Extract fastest laps for drivers
- Compare telemetry data (speed, throttle, brake)
- Analyze sector times
- Generate visualization plots
- Circuit map visualization with time delta coloring
- Automatic saving of all plots as PNG files

## Installation

1. Install Python 3.8 or higher

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Script

```bash
python src/fast_laps_analysis.py
```

This will:
1. Load a Formula 1 session (default: 2025 Monaco Qualifying)
2. Identify the top 2 fastest drivers automatically
3. Compare their telemetry data
4. Generate and save comparison plots in the `outputs/` directory

### Using as a Module

```python
from src.fast_laps_analysis import analyze_session

# Analyze a specific session
results = analyze_session(2024, 'Spanish Grand Prix', 'Q')

# Or analyze a race
results = analyze_session(2024, 'Monaco', 'R')
```

### Function Parameters

```python
analyze_session(year, grand_prix, session_type='Q')
```

- `year`: Year of the race (e.g., 2024, 2025)
- `grand_prix`: Name of the Grand Prix (e.g., 'Monaco', 'Spanish Grand Prix')
- `session_type`: Type of session ('FP1', 'FP2', 'FP3', 'Q' for Qualifying, 'R' for Race)

## Output Files

All generated plots are saved in the `outputs/` directory:

1. `analysis_[driver1]_vs_[driver2]_[GP]_[year]_[session].png` - Main analysis plot (5 subplots)
2. `speed_maps_[driver1]_vs_[driver2]_[GP]_[year]_[session].png` - Speed maps side by side
3. `overlay_[driver1]_vs_[driver2]_[GP]_[year]_[session].png` - Circuit overlay with time delta coloring

## Color Scheme

- **Blue**: Driver 1 (fastest driver)
- **Red**: Driver 2 (second fastest driver)

In the overlay map:
- **Blue areas**: Driver 1 is ahead (positive time delta)
- **Red areas**: Driver 2 is ahead (negative time delta)

## Cache Directory

The project uses FastF1's caching system to store downloaded data locally in the `data/cache/` directory. This reduces API calls and speeds up subsequent runs.

## Data Available

The FastF1 library provides access to:
- Lap times and sector times
- Speed, RPM, gear data
- Throttle and brake usage
- DRS activation
- Position data (X, Y coordinates)
- Time delta calculations
- And much more!

## Resources

- [FastF1 Documentation](https://docs.fastf1.dev/)
- [FastF1 GitHub Repository](https://github.com/theOehrly/Fast-F1)

## License

This project is open source and available for educational purposes.

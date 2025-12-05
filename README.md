# F1 Fast Laps Analysis

A Python project for analyzing Formula 1 fastest laps using the FastF1 library. This project allows you to compare telemetry data, sector times, and performance metrics between drivers.

## Features

- Load Formula 1 sessions (Practice, Qualifying, Race)
- Extract fastest laps for drivers
- Compare telemetry data (speed, throttle, brake)
- Analyze sector times
- Generate visualization plots

## Installation

1. Install Python 3.8 or higher

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from fast_laps_analysis import load_session, compare_fastest_laps

# Load a qualifying session
session = load_session(2023, 'Spanish Grand Prix', 'Q')

# Compare fastest laps between two drivers
compare_fastest_laps(session, drivers=['VER', 'HAM'])
```

### Running the Main Script

```bash
python fast_laps_analysis.py
```

This will:
1. Load a Formula 1 session (default: 2023 Spanish Grand Prix Qualifying)
2. Identify the top 3 fastest drivers
3. Compare their telemetry data
4. Analyze sector times
5. Generate and save comparison plots

## Customization

You can modify the `main()` function in `fast_laps_analysis.py` to:
- Change the year, Grand Prix, or session type
- Select specific drivers to compare
- Adjust visualization parameters

## Cache Directory

The project uses FastF1's caching system to store downloaded data locally in the `./cache` directory. This reduces API calls and speeds up subsequent runs.

## Data Available

The FastF1 library provides access to:
- Lap times and sector times
- Speed, RPM, gear data
- Throttle and brake usage
- DRS activation
- Position data
- And much more!

## Resources

- [FastF1 Documentation](https://docs.fastf1.dev/)
- [FastF1 GitHub Repository](https://github.com/theOehrly/Fast-F1)

## License

This project is open source and available for educational purposes.


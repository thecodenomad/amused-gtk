# Muse Data Logger

A standalone script for logging heart rate and brain state data from Muse devices, designed as a precursor to meditation apps that can use this information to make environmental changes.

## Features

- **Automatic Device Discovery**: Discovers Muse devices with retry logic (up to 30 seconds)
- **Real-time Data Logging**: Logs heart rate and overall brain state continuously
- **Threading Support**: Background streaming with main thread available for data access
- **Data Retrieval API**: Methods to get last reading, last N readings
- **JSON Export**: Automatic export of session data when script terminates
- **Gen1/Gen3 Support**: Configurable for different Muse device generations
- **Signal Handling**: Graceful shutdown on Ctrl+C

## Usage

### Basic Usage (Auto-discovery)

```bash
python3 amused_example.py
```

### Specify Device Model

```bash
# For Gen1 devices (default preset: p1036)
python3 amused_example.py --model gen1

# For Gen3 devices (default preset: p1035)
python3 amused_example.py --model gen3
```

### Manual Device Address

```bash
python3 amused_example.py --device "XX:XX:XX:XX:XX:XX" --model gen1
```

### Custom Output File

```bash
python3 amused_example.py --output my_session.json
```

## API Usage

The script can also be used programmatically:

```python
from amused_example import MuseDataLogger

# Create logger instance
logger = MuseDataLogger(device_model='gen1')

# Start streaming
logger.start_streaming()

# Get real-time data
latest = logger.get_last_reading()
print(f"Heart Rate: {latest['heart_rate']}")
print(f"Brain State: {latest['overall_state']}")

# Get last 5 readings
recent = logger.get_last_n_readings(5)
print(f"Recent HR: {recent['heart_rate']}")

# Get status
status = logger.get_current_status()
print(f"Streaming: {status['is_streaming']}")

# Stop and export
logger.stop_streaming()
logger.export_to_json("my_data.json")
```

## Data Format

### Heart Rate Reading
```json
{
  "timestamp": 1640995200.123,
  "heart_rate": 75.5,
  "calibrated_hr": 56.625
}
```

### Overall State Reading
```json
{
  "timestamp": 1640995200.123,
  "frequency": 10.2,
  "brain_state": "Alpha"
}
```

### Session Export
```json
{
  "session_start": "2025-01-01T12:00:00.000000",
  "session_end": "2025-01-01T12:05:00.000000",
  "heart_rate_readings": [...],
  "overall_state_readings": [...],
  "eeg_channels": ["TP9", "AF7", "AF8", "TP10"],
  "metadata": {
    "device_address": "XX:XX:XX:XX:XX:XX",
    "device_model": "gen1",
    "preset": "p1036"
  }
}
```

## Brain States

The script identifies the following brain states based on dominant frequency:

- **Delta** (0.5-4 Hz): Deep sleep, healing
- **Theta** (4-8 Hz): Meditation, creativity
- **Alpha** (8-12 Hz): Relaxed wakefulness
- **Beta** (12-30 Hz): Active thinking
- **Gamma** (30-100 Hz): High cognitive activity

## Requirements

- Python 3.8+
- amused-py library (install with: `pip install git+https://github.com/thecodenomad/amused-py.git`)
- NumPy (optional, for frequency analysis)
- Muse S device (Gen1 or Gen3)

## Troubleshooting

### No Device Found
- Ensure Muse device is powered on and in pairing mode
- Check Bluetooth is enabled
- Try running with `--model gen1` for Gen1 devices

### Connection Issues
- For Gen1 devices, use preset `p1036` for better stability
- Ensure device is within Bluetooth range
- Try restarting the device

### Import Errors
- Install amused-py: `pip install git+https://github.com/thecodenomad/amused-py.git`
- Install NumPy: `pip install numpy`

## Meditation App Integration

This script serves as a foundation for meditation apps that can:

1. Monitor user's brain state in real-time
2. Adjust environmental factors (lighting, sound, temperature)
3. Provide feedback on meditation quality
4. Track progress over time
5. Trigger specific interventions based on brain state

The threading design allows the main application to continue running while data collection happens in the background.
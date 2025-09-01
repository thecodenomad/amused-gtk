#!/usr/bin/env python3
"""
Standalone Muse Data Logger Script

This script connects to a Muse device, logs heart rate and overall_state data,
and provides an interface for retrieving recent readings. Data is exported to JSON
when the script terminates.

Based on frequency_display.py and using the amused-py library.
"""

import os
import sys
import asyncio
import threading
import queue
import logging
import json
import signal
import time
from datetime import datetime
from collections import deque
from typing import List, Dict, Optional, Any

# Import amused-py SDK components
from amused.muse_client import MuseStreamClient

import numpy as np

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MuseTimeoutError(Exception):
    """Custom exception for Muse device discovery timeouts."""
    pass


class MuseDataLogger:
    """Standalone Muse data logger for heart rate and brain state monitoring."""

    def __init__(self, device_address: Optional[str] = None, device_model: str = 'gen3'):
        """
        Initialize the Muse data logger.

        Args:
            device_address: Bluetooth address of the Muse device (optional, will auto-discover)
            device_model: Device model ('gen1' or 'gen3', default: 'gen3')
        """
        self.device_address = device_address
        self.device_model = device_model
        self.preset = 'p1036' if device_model == 'gen1' else 'p1035'
        self.is_streaming = False
        self.client: Optional[MuseStreamClient] = None
        self.should_stop = False  # Flag to signal threads to stop

        # Data storage
        self.heart_rate_data: deque = deque(maxlen=1000)  # Store last 1000 readings
        self.overall_state_data: deque = deque(maxlen=1000)
        self.eeg_buffers = {
            'TP9': deque(maxlen=1000),
            'AF7': deque(maxlen=1000),
            'AF8': deque(maxlen=1000),
            'TP10': deque(maxlen=1000)
        }

        # Threading
        self.stream_thread: Optional[threading.Thread] = None
        self.data_lock = threading.Lock()

        # Session tracking
        self.session_start_time = None
        self.session_data = {
            'session_start': None,
            'session_end': None,
            'heart_rate_readings': [],
            'overall_state_readings': [],
            'eeg_channels': ['TP9', 'AF7', 'AF8', 'TP10'],
            'metadata': {
                'device_address': device_address,
                'device_model': device_model,
                'preset': self.preset
            }
        }

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.should_stop = True
        self.stop_streaming()
        # Don't call sys.exit() here - let the main thread handle cleanup
        # This prevents the core dump from daemon threads



    def start_streaming(self) -> bool:
        """
        Start streaming data from the Muse device.

        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        if self.is_streaming:
            logger.warning("Already streaming")
            return False

        # Start in a separate thread to avoid blocking (not daemon so it can finish cleanly)
        self.stream_thread = threading.Thread(target=self._run_streaming_async)
        self.stream_thread.start()

        # Wait a bit for the thread to start
        time.sleep(1)
        return self.is_streaming

    def _run_streaming_async(self):
        """Run the streaming process in an asyncio event loop."""
        try:
            asyncio.run(self._streaming_loop())
        except Exception as e:
            logger.error(f"Streaming error: {e}")

    async def _streaming_loop(self):
        """Main streaming loop."""
        try:
            # Create streaming client first
            try:
                self.client = MuseStreamClient(device_model='auto', verbose=True)
                logger.info("MuseStreamClient created successfully")
            except Exception as e:
                logger.error(f"Failed to create MuseStreamClient: {e}")
                logger.error("This may indicate an issue with the amused-py library")
                self.is_streaming = False
                return

            # Discover device if not provided
            if not self.device_address:
                logger.info("No device address provided, starting discovery...")
                try:
                    device = await self.client.find_device()
                    if device:
                        self.device_address = device.address
                        logger.info(f"Device discovery successful: {self.device_address}")
                    else:
                        logger.error("No Muse device found")
                        self.is_streaming = False
                        return
                except Exception as e:
                    logger.error(f"Unexpected error during device discovery: {e}")
                    logger.error("Cannot proceed without a device address")
                    self.is_streaming = False
                    return

            if not self.device_address:
                logger.error("No device address available after discovery attempt")
                self.is_streaming = False
                return

            # Set up callbacks
            try:
                self.client.on_eeg(self._process_eeg_data)
                self.client.on_ppg(self._process_ppg_data)
                self.client.on_heart_rate(self._process_heart_rate_data)
                logger.info("Callbacks registered successfully")
            except Exception as e:
                logger.error(f"Failed to register callbacks: {e}")
                logger.error("This may indicate an issue with the MuseStreamClient")
                self.is_streaming = False
                return

            # Mark session start
            self.session_start_time = datetime.now()
            self.session_data['session_start'] = self.session_start_time.isoformat()

            logger.info(f"Starting streaming from {self.device_address} with preset {self.preset}")
            self.is_streaming = True

            # Start streaming with reconnection logic
            reconnect_attempts = 0
            max_reconnect_attempts = 3

            while reconnect_attempts < max_reconnect_attempts and not self.should_stop:
                try:
                    logger.info(f"Starting streaming (attempt {reconnect_attempts + 1}/{max_reconnect_attempts})...")
                    # Create a task for streaming so we can cancel it
                    streaming_task = asyncio.create_task(
                        self.client.connect_and_stream(
                            self.device_address,
                            duration_seconds=0  # Continuous streaming
                        )
                    )

                    # Wait for either streaming to complete or should_stop to be set
                    while not streaming_task.done() and not self.should_stop:
                        await asyncio.sleep(0.1)  # Check every 100ms

                    if self.should_stop:
                        logger.info("Cancelling streaming task due to stop signal...")
                        streaming_task.cancel()
                        try:
                            await streaming_task
                        except asyncio.CancelledError:
                            logger.info("Streaming task cancelled successfully")
                        self.is_streaming = False
                        break

                    success = streaming_task.result()

                    if not success:
                        logger.error("Failed to start streaming")
                        logger.error("This may indicate:")
                        logger.error("  - Device is not in pairing mode")
                        logger.error("  - Device is already connected to another application")
                        logger.error("  - Bluetooth connection issues")
                        logger.error("  - Incorrect device address")
                        self.is_streaming = False
                        break
                    else:
                        logger.info("Streaming started successfully")
                        # Reset reconnect attempts on successful connection
                        reconnect_attempts = 0

                        # Keep streaming alive - this will block until streaming ends
                        # The streaming should continue indefinitely until interrupted
                        logger.info("Streaming is now active. Press Ctrl+C to stop.")
                        break

                except Exception as e:
                    reconnect_attempts += 1
                    logger.error(f"Error during streaming (attempt {reconnect_attempts}): {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Streaming traceback: {traceback.format_exc()}")

                    if reconnect_attempts < max_reconnect_attempts and not self.should_stop:
                        logger.info(f"Will attempt to reconnect in 5 seconds... ({reconnect_attempts}/{max_reconnect_attempts})")
                        try:
                            await asyncio.sleep(5)
                        except asyncio.CancelledError:
                            logger.info("Reconnection cancelled by user")
                            break
                    else:
                        logger.error("Maximum reconnection attempts reached")
                        self.is_streaming = False
                        break

            if not self.is_streaming:
                logger.error("Failed to establish stable streaming connection")

        except Exception as e:
            logger.error(f"Unexpected error in streaming loop: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Loop traceback: {traceback.format_exc()}")
            self.is_streaming = False

    def stop_streaming(self):
        """Stop streaming and clean up resources."""
        logger.info("Stopping streaming...")
        self.is_streaming = False

        if self.client:
            # Note: The client doesn't have a direct disconnect method in this version
            # The streaming will stop when is_streaming is set to False
            pass

        # Mark session end
        if self.session_start_time:
            self.session_data['session_end'] = datetime.now().isoformat()

        logger.info("Streaming stopped")

    def _process_eeg_data(self, data: Dict[str, Any]):
        """Process incoming EEG data."""
        try:
            if 'channels' in data:
                channels = data['channels']
                timestamp = time.time()

                with self.data_lock:
                    # Store EEG data for each channel
                    for channel_name, samples in channels.items():
                        if channel_name in self.eeg_buffers and isinstance(samples, list):
                            self.eeg_buffers[channel_name].extend(samples)

                    # Calculate overall brain state
                    overall_freq = self._calculate_overall_frequency()
                    if overall_freq > 0:
                        reading = {
                            'timestamp': timestamp,
                            'frequency': overall_freq,
                            'brain_state': self._get_brain_state_name(overall_freq)
                        }
                        self.overall_state_data.append(reading)
                        self.session_data['overall_state_readings'].append(reading)

        except Exception as e:
            logger.error(f"Error processing EEG data: {e}")

    def _process_ppg_data(self, data: Dict[str, Any]):
        """Process incoming PPG data for heart rate calculation."""
        try:
            if 'samples' in data:
                samples = data['samples']
                # Store PPG samples for heart rate calculation
                # This would be used by a more sophisticated heart rate algorithm
                pass
        except Exception as e:
            logger.error(f"Error processing PPG data: {e}")

    def _process_heart_rate_data(self, hr: float):
        """Process incoming heart rate data."""
        try:
            if hr and hr > 0:
                timestamp = time.time()
                reading = {
                    'timestamp': timestamp,
                    'heart_rate': hr,
                    'calibrated_hr': hr * 0.75  # Apply calibration as in frequency_display.py
                }

                with self.data_lock:
                    self.heart_rate_data.append(reading)
                    self.session_data['heart_rate_readings'].append(reading)

                logger.debug(f"Heart rate: {hr:.1f} BPM (calibrated: {reading['calibrated_hr']:.1f} BPM)")

        except Exception as e:
            logger.error(f"Error processing heart rate data: {e}")

    def _calculate_overall_frequency(self) -> float:
        """Calculate overall brain state frequency from EEG channels."""
        try:
            if np is None:
                logger.warning("NumPy not available, cannot calculate frequency")
                return 0.0

            # Simple dominant frequency calculation across all channels
            sample_rate = 250 if self.device_model == 'gen1' else 256
            dominant_freqs = []

            for channel_name in self.eeg_buffers:
                if len(self.eeg_buffers[channel_name]) >= 100:
                    # Take last 100 samples for FFT
                    data = list(self.eeg_buffers[channel_name])[-100:]
                    if len(data) >= 50:  # Need minimum samples
                        data_array = np.array(data)
                        data_array = data_array - np.mean(data_array)

                        fft = np.fft.rfft(data_array)
                        freqs = np.fft.rfftfreq(len(data_array), 1/sample_rate)
                        power = np.abs(fft) ** 2

                        # Find dominant frequency in 1-40 Hz range
                        valid_mask = (freqs >= 1) & (freqs <= 40)
                        if np.any(valid_mask):
                            valid_power = power[valid_mask]
                            valid_freqs = freqs[valid_mask]
                            peak_idx = np.argmax(valid_power)
                            dominant_freqs.append(float(valid_freqs[peak_idx]))

            if dominant_freqs:
                return float(np.mean(dominant_freqs))
            return 0.0

        except Exception as e:
            logger.error(f"Error calculating overall frequency: {e}")
            return 0.0

    def _get_brain_state_name(self, freq: float) -> str:
        """Get brain state name based on frequency."""
        if freq < 4:
            return "Delta"
        elif freq < 8:
            return "Theta"
        elif freq < 12:
            return "Alpha"
        elif freq < 30:
            return "Beta"
        else:
            return "Gamma"

    def get_last_reading(self) -> Dict[str, Any]:
        """
        Get the most recent heart rate and overall state reading.

        Returns:
            Dict containing latest heart_rate and overall_state data
        """
        with self.data_lock:
            latest_hr = self.heart_rate_data[-1] if self.heart_rate_data else None
            latest_state = self.overall_state_data[-1] if self.overall_state_data else None

        return {
            'heart_rate': latest_hr,
            'overall_state': latest_state,
            'timestamp': time.time()
        }

    def get_last_n_readings(self, n: int) -> Dict[str, Any]:
        """
        Get the last N readings for heart rate and overall state.

        Args:
            n: Number of recent readings to retrieve

        Returns:
            Dict containing lists of recent heart_rate and overall_state readings
        """
        with self.data_lock:
            hr_readings = list(self.heart_rate_data)[-n:] if self.heart_rate_data else []
            state_readings = list(self.overall_state_data)[-n:] if self.overall_state_data else []

        return {
            'heart_rate': hr_readings,
            'overall_state': state_readings,
            'count': max(len(hr_readings), len(state_readings))
        }

    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current streaming status and recent data summary.

        Returns:
            Dict with streaming status and data summary
        """
        with self.data_lock:
            hr_count = len(self.heart_rate_data)
            state_count = len(self.overall_state_data)

        return {
            'is_streaming': self.is_streaming,
            'device_address': self.device_address,
            'device_model': self.device_model,
            'total_hr_readings': hr_count,
            'total_state_readings': state_count,
            'session_duration': time.time() - (self.session_start_time.timestamp() if self.session_start_time else time.time()),
            'latest_reading': self.get_last_reading()
        }

    def export_to_json(self, filename: Optional[str] = None) -> str:
        """
        Export session data to JSON file.

        Args:
            filename: Optional filename (default: auto-generated with timestamp)

        Returns:
            str: Path to the exported JSON file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"muse_session_{timestamp}.json"

        # Update session end time
        self.session_data['session_end'] = datetime.now().isoformat()

        try:
            with open(filename, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)

            logger.info(f"Session data exported to {filename}")
            logger.info(f"Total heart rate readings: {len(self.session_data['heart_rate_readings'])}")
            logger.info(f"Total overall state readings: {len(self.session_data['overall_state_readings'])}")

            return filename

        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return ""


def main():
    """Main function to run the Muse data logger."""
    import argparse

    parser = argparse.ArgumentParser(description="Muse Data Logger")
    parser.add_argument('--device', '-d', help='Muse device Bluetooth address (optional, will auto-discover)')
    parser.add_argument('--model', '-m', choices=['gen1', 'gen3'], default='gen3',
                       help='Muse device model (default: gen3)')
    parser.add_argument('--output', '-o', help='Output JSON filename (optional)')

    args = parser.parse_args()

    # Create logger instance
    logger_instance = MuseDataLogger(device_address=args.device, device_model=args.model)

    print("Muse Data Logger")
    print("=================")
    print(f"Device: {args.device or 'Auto-discover'}")
    print(f"Model: {args.model}")
    print(f"Preset: {logger_instance.preset}")
    print("Press Ctrl+C to stop and export data")
    print()

    # Start streaming
    if logger_instance.start_streaming():
        print("Streaming started successfully!")
        print("Use the logger instance to get real-time data:")
        print("- logger.get_last_reading() - Get most recent data")
        print("- logger.get_last_n_readings(5) - Get last 5 readings")
        print("- logger.get_current_status() - Get streaming status")
        print()

        # Keep the main thread alive and wait for the streaming thread to finish
        try:
            while logger_instance.is_streaming and logger_instance.stream_thread and logger_instance.stream_thread.is_alive():
                # Use a shorter sleep to be more responsive to signals
                time.sleep(0.1)
                # Print status every 10 seconds
                if int(time.time()) % 10 == 0:
                    status = logger_instance.get_current_status()
                    print(f"Status: Streaming=True, HR readings={status['total_hr_readings']}, State readings={status['total_state_readings']}")

            # Wait for the streaming thread to finish cleanly
            if logger_instance.stream_thread and logger_instance.stream_thread.is_alive():
                logger_instance.stream_thread.join(timeout=5.0)
                if logger_instance.stream_thread.is_alive():
                    logger.warning("Streaming thread did not finish cleanly")

        except KeyboardInterrupt:
            print("\nStopping...")
            logger_instance.should_stop = True
            # Force stop streaming immediately
            logger_instance.stop_streaming()

            # Give threads a moment to cleanup
            time.sleep(0.5)

            # Export data and exit cleanly
            json_file = logger_instance.export_to_json(args.output)
            if json_file:
                print(f"Data exported to: {json_file}")
            sys.exit(0)

        # Stop streaming and export (normal exit)
        logger_instance.stop_streaming()

        # Wait a moment for cleanup
        time.sleep(0.5)

        json_file = logger_instance.export_to_json(args.output)
        if json_file:
            print(f"Data exported to: {json_file}")
    else:
        print("Failed to start streaming")
        sys.exit(1)


if __name__ == "__main__":
    main()

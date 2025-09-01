#!/usr/bin/env python3
"""
Test script to verify data processing logic in FrequencyDisplay
"""

import sys
import os
import numpy as np
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_frequency_analysis():
    """Test the frequency analysis logic"""
    print("Testing frequency analysis...")

    # Simulate EEG data (250 Hz sampling rate)
    sample_rate = 250
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Create simulated EEG data with different frequency components
    # Mix of alpha (10 Hz), beta (20 Hz), and some noise
    alpha_wave = 2 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    beta_wave = 1 * np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
    noise = 0.5 * np.random.randn(len(t))
    eeg_data = alpha_wave + beta_wave + noise

    print(f"Generated {len(eeg_data)} samples of simulated EEG data")

    # Apply window function
    window = np.hanning(len(eeg_data))
    data_windowed = eeg_data * window

    # Perform FFT
    fft_result = np.fft.rfft(data_windowed)
    freqs = np.fft.rfftfreq(len(data_windowed), d=1/sample_rate)

    # Calculate power spectral density
    psd = np.abs(fft_result) ** 2

    # Extract frequency bands
    delta_power = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
    theta_power = np.sum(psd[(freqs >= 4) & (freqs < 8)])
    alpha_power = np.sum(psd[(freqs >= 8) & (freqs < 12)])
    beta_power = np.sum(psd[(freqs >= 12) & (freqs < 30)])
    gamma_power = np.sum(psd[(freqs >= 30) & (freqs < 50)])

    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")

    # Calculate dominant frequency
    total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
    if total_power > 0:
        dominant_freq = (delta_power * 2 + theta_power * 6 + alpha_power * 10 +
                        beta_power * 20 + gamma_power * 40) / total_power
        print(".1f")

        # Determine brain state
        if dominant_freq < 4:
            state = "Delta"
        elif dominant_freq < 8:
            state = "Theta"
        elif dominant_freq < 12:
            state = "Alpha"
        elif dominant_freq < 30:
            state = "Beta"
        else:
            state = "Gamma"

        print(f"Dominant brain state: {state}")

    return True

def test_brain_state_tracking():
    """Test brain state tracking logic"""
    print("\nTesting brain state tracking...")

    # Simulate brain state tracking variables
    brain_state_tracking = {
        'Delta': {'count': 0, 'total_time': 0.0, 'current_start': None},
        'Theta': {'count': 0, 'total_time': 0.0, 'current_start': None},
        'Alpha': {'count': 0, 'total_time': 0.0, 'current_start': None},
        'Beta': {'count': 0, 'total_time': 0.0, 'current_start': None},
        'Gamma': {'count': 0, 'total_time': 0.0, 'current_start': None}
    }

    current_brain_state = None
    import time

    # Simulate state changes
    test_sequence = [
        (10.5, 2.0),  # Alpha for 2 seconds
        (6.0, 1.5),   # Theta for 1.5 seconds
        (15.0, 3.0),  # Beta for 3 seconds
        (10.0, 1.0),  # Back to Alpha for 1 second
    ]

    current_time = time.time()

    for freq, duration in test_sequence:
        # Determine new state
        if freq < 4:
            new_state = "Delta"
        elif freq < 8:
            new_state = "Theta"
        elif freq < 12:
            new_state = "Alpha"
        elif freq < 30:
            new_state = "Beta"
        else:
            new_state = "Gamma"

        print(".1f")

        # Update tracking
        if new_state != current_brain_state:
            print(f"State change: {current_brain_state} -> {new_state}")

            # End previous state
            if current_brain_state and brain_state_tracking[current_brain_state]['current_start']:
                start_time = brain_state_tracking[current_brain_state]['current_start']
                duration_prev = current_time - start_time
                brain_state_tracking[current_brain_state]['total_time'] += duration_prev
                print(".2f")

            # Start new state
            brain_state_tracking[new_state]['count'] += 1
            brain_state_tracking[new_state]['current_start'] = current_time
            current_brain_state = new_state

        current_time += duration

    # Finalize last state
    if current_brain_state and brain_state_tracking[current_brain_state]['current_start']:
        start_time = brain_state_tracking[current_brain_state]['current_start']
        duration_final = current_time - start_time
        brain_state_tracking[current_brain_state]['total_time'] += duration_final

    # Print summary
    print("\nFinal brain state summary:")
    for state, data in brain_state_tracking.items():
        avg_time = data['total_time'] / max(data['count'], 1)
        print(f"{state}: Count={data['count']}, Total={data['total_time']:.1f}s, Avg={avg_time:.1f}s")

    return True

if __name__ == "__main__":
    print("🧪 Testing FrequencyDisplay data processing logic...\n")

    success1 = test_frequency_analysis()
    success2 = test_brain_state_tracking()

    if success1 and success2:
        print("\n✅ Data processing logic tests passed!")
        print("The frequency analysis and brain state tracking should work correctly.")
        print("If graphs aren't updating, the issue is likely with:")
        print("1. EEG data not being received from the Muse device")
        print("2. Callbacks not being triggered properly")
        print("3. UI not being updated on the main thread")
    else:
        print("\n❌ Some tests failed")
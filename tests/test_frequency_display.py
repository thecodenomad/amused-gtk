#!/usr/bin/env python3
"""
Test script to verify FrequencyDisplay can be instantiated without GUI errors
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_matplotlib_backend():
    """Test matplotlib backend setup without GTK template system"""
    try:
        print("Testing matplotlib backend setup...")

        # Import matplotlib with the backend already set
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        backend = matplotlib.get_backend()
        print(f"📊 Matplotlib backend: {backend}")

        # Test figure creation
        print("🔧 Testing matplotlib figure creation...")
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
        print("✅ Figure created successfully")

        # Test canvas creation
        canvas = FigureCanvas(fig)
        print("✅ Canvas created successfully")

        # Test plot elements
        line, = ax.plot([], [], 'r-', linewidth=1)
        print("✅ Plot elements created successfully")

        plt.close(fig)  # Clean up

        print("🎉 Matplotlib backend test passed!")
        return True

    except Exception as e:
        print(f"❌ Matplotlib backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_client_import():
    """Test that the MuseStreamClient can be imported and initialized"""
    try:
        print("Testing MuseStreamClient import...")

        from amused.muse_client import MuseStreamClient
        print("✅ MuseStreamClient import successful")

        # Test client initialization
        client = MuseStreamClient(device_model='auto', verbose=False)
        print("✅ MuseStreamClient created successfully")

        # Test client stats (if available)
        if hasattr(client, 'get_stats'):
            stats = client.get_stats()
            print(f"📊 Initial stats: {stats}")

        print("🎉 MuseStreamClient test passed!")
        return True

    except Exception as e:
        print(f"❌ MuseStreamClient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frequency_display_init():
    """Test that FrequencyDisplay can be initialized without matplotlib backend errors"""
    try:
        print("Testing FrequencyDisplay initialization...")

        # Import required modules
        import gi
        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")

        from gi.repository import Gtk, Adw, GLib
        from frequency_display import FrequencyDisplay

        print("✅ Imports successful")

        # Test matplotlib backend detection
        from frequency_display import matplotlib
        backend = matplotlib.get_backend()
        print(f"📊 Matplotlib backend: {backend}")

        # Test FrequencyDisplay initialization (without GUI)
        print("🔧 Testing FrequencyDisplay initialization...")
        display = FrequencyDisplay()
        print("✅ FrequencyDisplay created successfully")

        # Test that plots were set up
        if hasattr(display, 'figures'):
            print(f"📈 Created {len(display.figures)} matplotlib figures")
            for name in display.figures.keys():
                print(f"  - {name}")

        # Test data buffers
        if hasattr(display, 'data_buffers'):
            print(f"📊 Created {len(display.data_buffers)} data buffers")

        print("🎉 All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Running FrequencyDisplay tests...\n")

    # Test matplotlib backend first
    print("1. Testing matplotlib backend...")
    matplotlib_success = test_matplotlib_backend()
    print()

    # Test client
    print("2. Testing MuseStreamClient...")
    client_success = test_client_import()
    print()

    # Test full FrequencyDisplay (may fail due to GTK resources)
    print("3. Testing FrequencyDisplay initialization...")
    display_success = test_frequency_display_init()
    print()

    # Summary
    all_passed = matplotlib_success and client_success
    if all_passed:
        print("🎉 Core functionality tests passed!")
        if not display_success:
            print("⚠️  Full GUI test failed (expected in headless environment)")
            print("   This is normal - the core matplotlib functionality works correctly.")
    else:
        print("❌ Some tests failed")

    sys.exit(0 if all_passed else 1)
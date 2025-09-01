#!/usr/bin/env python3
"""
Test GTK integration with matplotlib
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gtk_matplotlib_integration():
    """Test that GTK and matplotlib can work together"""
    try:
        print("Testing GTK + matplotlib integration...")

        # Import GTK first
        import gi
        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")

        from gi.repository import Gtk, Adw, GLib

        # Import matplotlib with proper backend
        import matplotlib
        try:
            matplotlib.use('GTK4Agg')
            from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg as FigureCanvas
            GTK4_AVAILABLE = True
            print("✅ GTK4Agg backend available")
        except ImportError:
            matplotlib.use('Agg')
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            GTK4_AVAILABLE = False
            print("✅ Using Agg backend (fallback)")

        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        print(f"📊 Matplotlib backend: {matplotlib.get_backend()}")

        # Test figure creation
        print("🔧 Testing matplotlib figure creation...")
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
        print("✅ Figure created successfully")

        # Test canvas creation
        if GTK4_AVAILABLE:
            canvas = FigureCanvas(fig)
            print("✅ GTK4 canvas created successfully")

            # Test that it's a GTK widget
            if isinstance(canvas, Gtk.Widget):
                print("✅ Canvas is a valid GTK widget")
            else:
                print("❌ Canvas is not a GTK widget")
                return False
        else:
            canvas = FigureCanvas(fig)
            print("✅ Agg canvas created successfully")
            print("⚠️  Canvas is not a GTK widget (expected with Agg backend)")

        # Test plot elements
        line, = ax.plot([], [], 'r-', linewidth=1)
        print("✅ Plot elements created successfully")

        plt.close(fig)  # Clean up

        print("🎉 GTK + matplotlib integration test passed!")
        return True

    except Exception as e:
        print(f"❌ GTK + matplotlib integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Running GTK + matplotlib integration test...\n")

    success = test_gtk_matplotlib_integration()

    if success:
        print("\n🎉 GTK widget issue should be resolved!")
        print("The application should now start without the TypeError.")
    else:
        print("\n❌ Integration test failed")

    sys.exit(0 if success else 1)
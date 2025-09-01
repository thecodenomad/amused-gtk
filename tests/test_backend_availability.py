#!/usr/bin/env python3
"""
Test which matplotlib backends are actually available and working
"""

import matplotlib
import sys

def test_backend(backend_name):
    """Test if a specific backend can be used"""
    try:
        print(f"\n🧪 Testing {backend_name} backend...")
        matplotlib.use(backend_name)

        # Try to create a figure
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 3))
        plt.close(fig)

        print(f"✅ {backend_name} backend works!")
        return True
    except Exception as e:
        print(f"❌ {backend_name} backend failed: {e}")
        return False

def main():
    print("🔍 Testing matplotlib backend availability...")
    print(f"📊 Current backend: {matplotlib.get_backend()}")

    # List of backends to test
    backends_to_test = [
        'GTK4Agg',
        'GTK3Agg',
        'TkAgg',
        'Qt5Agg',
        'Qt4Agg',
        'Agg',  # Non-interactive fallback
    ]

    working_backends = []

    for backend in backends_to_test:
        if test_backend(backend):
            working_backends.append(backend)

    print("\n📋 Summary:")
    print(f"✅ Working backends: {working_backends}")

    if 'GTK4Agg' in working_backends:
        print("🎉 GTK4Agg is available - should work with GTK4!")
        recommended = 'GTK4Agg'
    elif 'GTK3Agg' in working_backends:
        print("⚠️ GTK4Agg not available, but GTK3Agg is - consider using GTK3")
        recommended = 'GTK3Agg'
    elif working_backends:
        recommended = working_backends[0]
        print(f"⚠️ No GTK backends available, using {recommended}")
    else:
        recommended = 'Agg'
        print("❌ No backends work, using Agg")

    print(f"\n🎯 Recommended backend: {recommended}")

    # Test the recommended backend more thoroughly
    print(f"\n🔧 Testing {recommended} backend thoroughly...")
    matplotlib.use(recommended)
    import matplotlib.pyplot as plt

    try:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        line, = ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'r-')
        ax.set_title('Test Plot')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        plt.close(fig)
        print("✅ Thorough test passed!")
    except Exception as e:
        print(f"❌ Thorough test failed: {e}")

if __name__ == "__main__":
    main()
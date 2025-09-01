#!/usr/bin/env python3
"""
Test matplotlib figure to image rendering
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io

def test_matplotlib_to_image():
    """Test rendering matplotlib figure to image"""
    try:
        print("Testing matplotlib to image rendering...")

        # Create a simple plot
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_title('Test Plot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)

        # Render to canvas
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get buffer
        buf = canvas.buffer_rgba()
        print(f"Buffer type: {type(buf)}")
        print(f"Buffer shape: {buf.shape}")

        # Convert to numpy array
        arr = np.asarray(buf)
        print(f"Array shape: {arr.shape}")
        print(f"Array dtype: {arr.dtype}")

        # Convert to bytes
        image_bytes = arr.tobytes()
        print(f"Image bytes length: {len(image_bytes)}")

        # Test PIL conversion
        try:
            from PIL import Image
            image = Image.frombuffer('RGBA', canvas.get_width_height(), image_bytes, 'raw', 'RGBA', 0, 1)
            print(f"PIL image size: {image.size}")
            print(f"PIL image mode: {image.mode}")

            # Save to PNG
            png_buffer = io.BytesIO()
            image.save(png_buffer, format='PNG')
            png_bytes = png_buffer.getvalue()
            print(f"PNG bytes length: {len(png_bytes)}")

            print("✅ PIL conversion successful!")
        except ImportError:
            print("⚠️ PIL not available, but numpy conversion worked")

        plt.close(fig)
        print("🎉 Matplotlib to image rendering test passed!")
        return True

    except Exception as e:
        print(f"❌ Matplotlib to image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_matplotlib_to_image()
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")
    exit(0 if success else 1)
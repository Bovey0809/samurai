#!/usr/bin/env python3
"""
Test script for SAMURAI Web Demo
"""

import os
import sys
import tempfile
import shutil
import cv2
import numpy as np

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing dependencies...")
    
    try:
        import flask
        print(f"  ✅ Flask {flask.__version__}")
    except ImportError as e:
        print(f"  ❌ Flask: {e}")
        return False
    
    try:
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"  ❌ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ NumPy: {e}")
        return False
    
    try:
        from samurai_demo import SAMURAIDemo
        print("  ✅ SAMURAIDemo")
    except ImportError as e:
        print(f"  ❌ SAMURAIDemo: {e}")
        return False
    
    return True

def test_video_processing():
    """Test video processing functionality"""
    print("\n🎬 Testing video processing...")
    
    # Create a test video
    temp_dir = tempfile.mkdtemp()
    test_video_path = os.path.join(temp_dir, "test_video.mp4")
    
    try:
        # Create a simple test video (10 frames, 640x480)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video_path, fourcc, 10.0, (640, 480))
        
        for i in range(10):
            # Create a frame with a moving rectangle
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            x = 50 + i * 20
            cv2.rectangle(frame, (x, 200), (x + 100, 300), (0, 255, 0), -1)
            out.write(frame)
        
        out.release()
        print(f"  ✅ Test video created: {test_video_path}")
        
        # Test frame extraction
        cap = cv2.VideoCapture(test_video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"  ✅ Frame extracted: {frame.shape}")
            frame_path = os.path.join(temp_dir, "first_frame.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"  ✅ Frame saved: {frame_path}")
        else:
            print("  ❌ Failed to extract frame")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Video processing error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def test_web_app_imports():
    """Test if web app can be imported"""
    print("\n🌐 Testing web app imports...")
    
    try:
        # Test basic Flask app creation
        from flask import Flask
        app = Flask(__name__)
        print("  ✅ Flask app created")
        
        # Test web demo imports
        import web_demo
        print("  ✅ Web demo module imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Web app import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 SAMURAI Web Demo Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Test dependencies
    if not test_dependencies():
        all_passed = False
    
    # Test video processing
    if not test_video_processing():
        all_passed = False
    
    # Test web app imports
    if not test_web_app_imports():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Web demo is ready to run.")
        print("\nTo start the web demo:")
        print("  python3 web_demo.py")
        print("\nThen open your browser to: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

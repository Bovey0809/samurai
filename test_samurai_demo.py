#!/usr/bin/env python3
"""
Test script for SAMURAI Demo
This script demonstrates how to use the SAMURAI tracking demo
"""

import os
import sys
import subprocess
import argparse

def test_samurai_demo():
    """Test the SAMURAI demo with a sample video"""
    
    print("🧪 Testing SAMURAI Demo")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("samurai_demo.py"):
        print("❌ samurai_demo.py not found. Please run this from the project root.")
        return False
    
    # Check if SAM2 is available
    if not os.path.exists("sam2"):
        print("❌ sam2 directory not found. Please ensure SAM2 is installed.")
        return False
    
    # Check if checkpoints exist
    checkpoint_path = "/root/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run the setup script first to download checkpoints.")
        return False
    
    print("✅ Environment check passed")
    
    # Create a simple test video if none exists
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        print("📹 Creating test video...")
        create_test_video(test_video_path)
    
    # Run the demo
    print("🚀 Running SAMURAI demo...")
    output_dir = "test_results"
    
    cmd = [
        "python", "samurai_demo.py",
        "--input_video", test_video_path,
        "--output_dir", output_dir,
        "--model_path", checkpoint_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ SAMURAI demo test completed successfully!")
            print(f"📁 Results saved to: {output_dir}")
            
            # Check output files
            expected_files = [
                "samurai_tracking.mp4",
                "tracking_results.csv", 
                "tracking_results.json",
                "summary.txt"
            ]
            
            for file in expected_files:
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path):
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} (missing)")
            
            return True
        else:
            print("❌ SAMURAI demo test failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def create_test_video(output_path, duration=5, fps=30):
    """Create a simple test video with moving objects"""
    try:
        import cv2
        import numpy as np
        
        # Video parameters
        width, height = 640, 480
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create moving circle
        center_x, center_y = width // 2, height // 2
        radius = 30
        
        for frame_idx in range(duration * fps):
            # Create background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)  # Dark gray background
            
            # Move circle in a circular pattern
            angle = 2 * np.pi * frame_idx / (fps * 2)  # Complete circle every 2 seconds
            x = int(center_x + 100 * np.cos(angle))
            y = int(center_y + 100 * np.sin(angle))
            
            # Draw moving circle
            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
            
            # Add frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Created test video: {output_path}")
        
    except ImportError:
        print("❌ OpenCV not available. Please install opencv-python")
        return False
    except Exception as e:
        print(f"❌ Error creating test video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test SAMURAI Demo")
    parser.add_argument("--video", help="Path to test video (optional)")
    parser.add_argument("--create_test_video", action="store_true", 
                       help="Create a test video")
    
    args = parser.parse_args()
    
    if args.create_test_video:
        create_test_video("test_video.mp4")
        return
    
    if args.video:
        # Test with provided video
        if not os.path.exists(args.video):
            print(f"❌ Video not found: {args.video}")
            return
        
        output_dir = "test_results"
        checkpoint_path = "/root/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        
        cmd = [
            "python", "samurai_demo.py",
            "--input_video", args.video,
            "--output_dir", output_dir,
            "--model_path", checkpoint_path
        ]
        
        print(f"🚀 Running SAMURAI demo with: {args.video}")
        subprocess.run(cmd)
        
    else:
        # Run standard test
        success = test_samurai_demo()
        if success:
            print("\n🎉 All tests passed! SAMURAI demo is working correctly.")
        else:
            print("\n❌ Tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

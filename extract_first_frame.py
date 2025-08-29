#!/usr/bin/env python3
"""
Simple script to extract the first frame from a video file
"""

import cv2
import os
import sys

def extract_first_frame(video_path, output_path=None):
    """
    Extract the first frame from a video file
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path for the output image (optional)
    
    Returns:
        str: Path to the saved frame image
    """
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if not ret:
        print("Error: Could not read frame from video")
        return None
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_path = f"{base_name}_first_frame.jpg"
    
    # Save the frame
    success = cv2.imwrite(output_path, frame)
    
    if success:
        print(f"✅ First frame extracted successfully!")
        print(f"📁 Saved to: {output_path}")
        print(f"📏 Frame dimensions: {frame.shape[1]}x{frame.shape[0]} pixels")
        return output_path
    else:
        print(f"Error: Failed to save frame to {output_path}")
        return None

if __name__ == "__main__":
    # Video file path
    video_path = "/root/samurai/Feishu20250829-174910.mov"
    
    print(f"🎬 Extracting first frame from: {video_path}")
    print("-" * 50)
    
    # Extract the frame
    result = extract_first_frame(video_path)
    
    if result:
        print("-" * 50)
        print("🎉 Frame extraction completed successfully!")
    else:
        print("-" * 50)
        print("❌ Frame extraction failed!")
        sys.exit(1)

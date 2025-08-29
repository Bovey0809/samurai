#!/usr/bin/env python3
"""
Script to convert MOV files to MP4 format using FFmpeg
"""

import subprocess
import os
import sys

def convert_mov_to_mp4(input_path, output_path=None):
    """
    Convert MOV file to MP4 format using FFmpeg
    
    Args:
        input_path (str): Path to the input MOV file
        output_path (str): Path for the output MP4 file (optional)
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.mp4"
    
    # Check if output file already exists
    if os.path.exists(output_path):
        print(f"Warning: Output file already exists: {output_path}")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            return False
    
    print(f"🎬 Converting: {input_path}")
    print(f"📁 Output: {output_path}")
    print("-" * 50)
    
    # FFmpeg command for conversion
    # -c:v libx264: Use H.264 codec for video
    # -c:a aac: Use AAC codec for audio
    # -preset medium: Balance between speed and compression
    # -crf 23: Constant Rate Factor for quality (lower = better quality)
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'medium',
        '-crf', '23',
        '-y',  # Overwrite output file without asking
        output_path
    ]
    
    try:
        # Run FFmpeg command
        print("🔄 Starting conversion...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ Conversion completed successfully!")
        
        # Get file sizes
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"📊 File sizes:")
        print(f"   Input (MOV):  {input_size:.1f} MB")
        print(f"   Output (MP4): {output_size:.1f} MB")
        print(f"   Compression:  {((input_size - output_size) / input_size * 100):.1f}%")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed!")
        print(f"Error: {e}")
        if e.stderr:
            print(f"FFmpeg error: {e.stderr}")
        return False
        
    except FileNotFoundError:
        print("❌ Error: FFmpeg not found!")
        print("Please install FFmpeg first:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  CentOS/RHEL: sudo yum install ffmpeg")
        print("  macOS: brew install ffmpeg")
        return False

if __name__ == "__main__":
    # Input MOV file path
    input_path = "/root/samurai/Feishu20250829-174910.mov"
    
    print("🎥 MOV to MP4 Converter")
    print("=" * 50)
    
    # Convert the file
    success = convert_mov_to_mp4(input_path)
    
    if success:
        print("=" * 50)
        print("🎉 Conversion completed successfully!")
    else:
        print("=" * 50)
        print("❌ Conversion failed!")
        sys.exit(1)

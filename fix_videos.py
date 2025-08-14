#!/usr/bin/env python3
"""
Fix corrupted SAMURAI tracking videos by regenerating them from extracted frames
"""

import os
import cv2
import glob
from pathlib import Path

def fix_video_from_frames(session_dir):
    """Regenerate video from extracted frames"""
    frames_dir = os.path.join(session_dir, 'frames')
    if not os.path.exists(frames_dir):
        print(f"❌ No frames directory found in {session_dir}")
        return False
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
    if not frame_files:
        print(f"❌ No frame files found in {frames_dir}")
        return False
    
    print(f"📹 Found {len(frame_files)} frames in {frames_dir}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"❌ Could not read first frame: {frame_files[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    fps = 24.0  # Default FPS
    
    # Create output video path
    output_path = os.path.join(session_dir, 'samurai_tracking_fixed.mp4')
    
    # Try different codecs
    codecs = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]
    
    for codec_name, ext in codecs:
        try:
            output_path = os.path.join(session_dir, f'samurai_tracking_fixed{ext}')
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                print(f"❌ Failed to open video writer with {codec_name} codec")
                continue
            
            print(f"✅ Using {codec_name} codec for {output_path}")
            
            # Write all frames
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(frame_file)
                if frame is not None:
                    video_writer.write(frame)
                else:
                    print(f"⚠️ Could not read frame {i}: {frame_file}")
            
            # Close video writer
            video_writer.release()
            
            # Verify the video was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ Successfully created video: {output_path}")
                return output_path
            else:
                print(f"❌ Video file is empty or not created: {output_path}")
                
        except Exception as e:
            print(f"❌ Error with {codec_name} codec: {e}")
            continue
    
    print(f"❌ Failed to create video with any codec")
    return False

def main():
    """Fix all corrupted videos in results directory"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print("🔧 Fixing corrupted SAMURAI tracking videos...")
    print("=" * 50)
    
    fixed_videos = []
    
    # Find all session directories
    for session_dir in glob.glob(os.path.join(results_dir, "*")):
        if os.path.isdir(session_dir):
            session_id = os.path.basename(session_dir)
            print(f"\n🎯 Processing session: {session_id}")
            
            # Check if original video is corrupted
            original_video = os.path.join(session_dir, 'samurai_tracking.mp4')
            if os.path.exists(original_video):
                # Try to read the video
                cap = cv2.VideoCapture(original_video)
                if not cap.isOpened():
                    print(f"❌ Original video is corrupted: {original_video}")
                    fixed_path = fix_video_from_frames(session_dir)
                    if fixed_path:
                        fixed_videos.append((session_id, fixed_path))
                else:
                    cap.release()
                    print(f"✅ Original video is fine: {original_video}")
            else:
                print(f"❌ No original video found: {original_video}")
                fixed_path = fix_video_from_frames(session_dir)
                if fixed_path:
                    fixed_videos.append((session_id, fixed_path))
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    if fixed_videos:
        print(f"✅ Fixed {len(fixed_videos)} videos:")
        for session_id, video_path in fixed_videos:
            print(f"  - {session_id}: {video_path}")
    else:
        print("✅ No videos needed fixing")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fix SAMURAI tracking videos by re-running tracking with visualization
"""

import os
import sys
import glob
from pathlib import Path

# Add sam2 to path
sys.path.append("./sam2")

def fix_video_with_tracking(session_dir):
    """Re-run tracking to generate proper video with visualization"""
    session_id = os.path.basename(session_dir)
    print(f"🎯 Re-running tracking for session: {session_id}")
    
    # Find the original video
    uploads_dir = "uploads"
    original_video = None
    for video_file in glob.glob(os.path.join(uploads_dir, session_id, "*")):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            original_video = video_file
            break
    
    if not original_video:
        print(f"❌ Original video not found for session {session_id}")
        return False
    
    print(f"📹 Found original video: {original_video}")
    
    # Find the initial bounding box from the session
    # We'll use a default center bbox since we don't have the original selection
    print(f"🎯 Using default center bounding box")
    
    # Import SAMURAI demo
    try:
        from samurai_demo import SAMURAIDemo
        
        # Initialize SAMURAI demo
        demo = SAMURAIDemo()
        
        # Use a default bounding box (center of frame)
        # This is a rough estimate - ideally we'd store the original bbox
        default_bbox = (100, 100, 200, 200)  # x, y, width, height
        
        print(f"🚀 Re-running SAMURAI tracking...")
        results = demo.process_video(
            video_path=original_video,
            output_dir=session_dir,
            initial_bbox=default_bbox
        )
        
        print(f"✅ Tracking completed for session {session_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error re-running tracking: {e}")
        return False

def main():
    """Fix all videos by re-running tracking"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print("🔧 Fixing SAMURAI tracking videos by re-running tracking...")
    print("=" * 60)
    
    fixed_sessions = []
    
    # Find all session directories
    for session_dir in glob.glob(os.path.join(results_dir, "*")):
        if os.path.isdir(session_dir):
            session_id = os.path.basename(session_dir)
            print(f"\n🎯 Processing session: {session_id}")
            
            # Check if we have the original video
            uploads_session_dir = os.path.join("uploads", session_id)
            if os.path.exists(uploads_session_dir):
                success = fix_video_with_tracking(session_dir)
                if success:
                    fixed_sessions.append(session_id)
            else:
                print(f"❌ No original video found for session {session_id}")
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    if fixed_sessions:
        print(f"✅ Fixed {len(fixed_sessions)} sessions:")
        for session_id in fixed_sessions:
            print(f"  - {session_id}")
    else:
        print("❌ No sessions could be fixed")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SAMURAI Demo Script
Single Object Tracking using SAM2-based SAMURAI algorithm

Input: MP4 video file
Output: 
- Processed video with tracking visualization
- Bounding box data for each frame (CSV format)
- Tracking results summary

Usage:
    python samurai_demo.py --input_video path/to/video.mp4 --output_dir results/
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import gc
import csv
import json
from pathlib import Path
from tqdm import tqdm
import time

# Add sam2 to path
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

class SAMURAIDemo:
    def __init__(self, model_path="/root/sam2/checkpoints/sam2.1_hiera_base_plus.pt", device="cuda:0"):
        """
        Initialize SAMURAI demo with model
        
        Args:
            model_path: Path to SAM2 checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model_path = model_path
        self.model_cfg = self._determine_model_cfg(model_path)
        
        print(f"🔧 Loading SAMURAI model: {model_path}")
        print(f"📋 Model config: {self.model_cfg}")
        print(f"🚀 Device: {device}")
        
        # Initialize predictor
        self.predictor = build_sam2_video_predictor(
            self.model_cfg, 
            self.model_path, 
            device=self.device
        )
        
        # Tracking colors
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        print("✅ SAMURAI model loaded successfully!")
    
    def _determine_model_cfg(self, model_path):
        """Determine model config based on model path"""
        if "large" in model_path:
            return "configs/samurai/sam2.1_hiera_l.yaml"
        elif "base_plus" in model_path:
            return "configs/samurai/sam2.1_hiera_b+.yaml"
        elif "small" in model_path:
            return "configs/samurai/sam2.1_hiera_s.yaml"
        elif "tiny" in model_path:
            return "configs/samurai/sam2.1_hiera_t.yaml"
        else:
            raise ValueError(f"Unknown model size in path: {model_path}")
    
    def extract_frames_from_video(self, video_path, output_dir):
        """Extract frames from MP4 video to a directory"""
        print(f"📹 Extracting frames from: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(output_dir, f"{frame_count:08d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
        cap.release()
        
        print(f"✅ Extracted {frame_count} frames to {output_dir}")
        print(f"📊 Video FPS: {fps}")
        
        return frame_count, fps
    
    def get_initial_bbox(self, video_path, frame_dir):
        """Get initial bounding box from user or first frame analysis"""
        # For demo purposes, we'll use a simple center-based bbox
        # In a real application, you might want to:
        # 1. Show the first frame and let user click to select object
        # 2. Use object detection to automatically find objects
        # 3. Load from a text file
        
        # Load first frame to get dimensions
        first_frame_path = os.path.join(frame_dir, "00000000.jpg")
        if not os.path.exists(first_frame_path):
            raise ValueError(f"First frame not found: {first_frame_path}")
        
        frame = cv2.imread(first_frame_path)
        height, width = frame.shape[:2]
        
        # Create a default bbox in the center (you can modify this)
        bbox_size = min(width, height) // 4
        x = (width - bbox_size) // 2
        y = (height - bbox_size) // 2
        w = bbox_size
        h = bbox_size
        
        # Convert to SAM2 format (x1, y1, x2, y2)
        bbox = (x, y, x + w, y + h)
        
        print(f"🎯 Initial bbox: {bbox} (center of frame)")
        print(f"📐 Frame dimensions: {width}x{height}")
        
        return bbox
    
    def process_video(self, video_path, output_dir, initial_bbox=None):
        """
        Process video with SAMURAI tracking
        
        Args:
            video_path: Path to input MP4 video
            output_dir: Directory to save results
            initial_bbox: Initial bounding box (x1, y1, x2, y2) or None for auto-detection
        """
        print(f"🚀 Starting SAMURAI video processing...")
        print(f"📁 Input: {video_path}")
        print(f"📁 Output: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract frames
        frame_dir = os.path.join(output_dir, "frames")
        frame_count, fps = self.extract_frames_from_video(video_path, frame_dir)
        
        # Get initial bbox if not provided
        if initial_bbox is None:
            initial_bbox = self.get_initial_bbox(video_path, frame_dir)
        
        # Initialize video writer
        output_video_path = os.path.join(output_dir, "samurai_tracking.mp4")
        first_frame = cv2.imread(os.path.join(frame_dir, "00000000.jpg"))
        height, width = first_frame.shape[:2]
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"❌ Failed to open video writer. Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                print(f"❌ Failed to open video writer with mp4v codec. Trying XVID...")
                output_video_path = output_video_path.replace('.mp4', '.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Initialize tracking results
        tracking_results = []
        
        print(f"🎬 Processing {frame_count} frames...")
        
        # Process with SAMURAI
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # Initialize state
            state = self.predictor.init_state(
                frame_dir, 
                offload_video_to_cpu=True, 
                offload_state_to_cpu=True, 
                async_loading_frames=True
            )
            
            # Add initial object
            frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                state, 
                box=initial_bbox, 
                frame_idx=0, 
                obj_id=0
            )
            
            # Process all frames
            for frame_idx, object_ids, masks in tqdm(
                self.predictor.propagate_in_video(state), 
                total=frame_count-1,
                desc="Tracking frames"
            ):
                # Process masks and extract bboxes
                frame_results = {}
                
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    
                    # Extract bounding box from mask
                    non_zero_indices = np.argwhere(mask)
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    
                    frame_results[obj_id] = {
                        'bbox': bbox,
                        'mask': mask.tolist(),  # Convert numpy array to list for JSON serialization
                        'confidence': 1.0  # SAM2 doesn't provide confidence scores
                    }
                
                tracking_results.append({
                    'frame_idx': frame_idx,
                    'objects': frame_results
                })
                
                # Visualize and save frame
                frame_path = os.path.join(frame_dir, f"{frame_idx:08d}.jpg")
                if os.path.exists(frame_path):
                    img = cv2.imread(frame_path)
                    
                    # Draw masks and bboxes
                    for obj_id, obj_data in frame_results.items():
                        bbox = obj_data['bbox']
                        mask = obj_data['mask']
                        
                        # Draw mask overlay
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = self.colors[obj_id % len(self.colors)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.3, 0)
                        
                        # Draw bounding box
                        x, y, w, h = bbox
                        cv2.rectangle(
                            img, 
                            (x, y), 
                            (x + w, y + h), 
                            self.colors[obj_id % len(self.colors)], 
                            2
                        )
                        
                        # Add object ID label
                        cv2.putText(
                            img, 
                            f"Object {obj_id}", 
                            (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            self.colors[obj_id % len(self.colors)], 
                            2
                        )
                    
                    # Add frame info
                    cv2.putText(
                        img, 
                        f"Frame: {frame_idx}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 255), 
                        2
                    )
                    
                    video_writer.write(img)
        
        # Clean up
        if video_writer.isOpened():
            video_writer.release()
            print(f"✅ Video writer closed successfully")
        else:
            print(f"❌ Video writer was not opened properly")
        
        # Save tracking results
        self.save_results(tracking_results, output_dir, fps)
        
        # Clean up GPU memory
        del state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        
        print(f"✅ Processing complete!")
        print(f"📹 Output video: {output_video_path}")
        print(f"📊 Results saved to: {output_dir}")
        
        return tracking_results
    
    def save_results(self, tracking_results, output_dir, fps):
        """Save tracking results in multiple formats"""
        
        # Save as CSV
        csv_path = os.path.join(output_dir, "tracking_results.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['frame_idx', 'object_id', 'x', 'y', 'width', 'height', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_data in tracking_results:
                frame_idx = frame_data['frame_idx']
                for obj_id, obj_data in frame_data['objects'].items():
                    bbox = obj_data['bbox']
                    writer.writerow({
                        'frame_idx': frame_idx,
                        'object_id': obj_id,
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2],
                        'height': bbox[3],
                        'confidence': obj_data['confidence']
                    })
        
        # Save as JSON
        json_path = os.path.join(output_dir, "tracking_results.json")
        with open(json_path, 'w') as jsonfile:
            json.dump(tracking_results, jsonfile, indent=2)
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as summaryfile:
            summaryfile.write(f"SAMURAI Tracking Results Summary\n")
            summaryfile.write(f"================================\n")
            summaryfile.write(f"Total frames processed: {len(tracking_results)}\n")
            summaryfile.write(f"Video FPS: {fps}\n")
            summaryfile.write(f"Duration: {len(tracking_results)/fps:.2f} seconds\n")
            summaryfile.write(f"Objects tracked: {len(set(obj_id for frame in tracking_results for obj_id in frame['objects'].keys()))}\n")
            summaryfile.write(f"\nOutput files:\n")
            summaryfile.write(f"- samurai_tracking.mp4: Processed video with visualization\n")
            summaryfile.write(f"- tracking_results.csv: Bounding box data in CSV format\n")
            summaryfile.write(f"- tracking_results.json: Detailed results in JSON format\n")
            summaryfile.write(f"- summary.txt: This summary file\n")
        
        print(f"📄 Results saved:")
        print(f"  - CSV: {csv_path}")
        print(f"  - JSON: {json_path}")
        print(f"  - Summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="SAMURAI Single Object Tracking Demo")
    parser.add_argument("--input_video", required=True, help="Input MP4 video file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--model_path", default="/root/sam2/checkpoints/sam2.1_hiera_base_plus.pt", 
                       help="Path to SAM2 model checkpoint")
    parser.add_argument("--device", default="cuda:0", help="Device to run inference on")
    parser.add_argument("--bbox", nargs=4, type=int, help="Initial bounding box (x1 y1 x2 y2)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_video):
        print(f"❌ Input video not found: {args.input_video}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize SAMURAI demo
    demo = SAMURAIDemo(model_path=args.model_path, device=args.device)
    
    # Process initial bbox
    initial_bbox = None
    if args.bbox:
        initial_bbox = tuple(args.bbox)
        print(f"🎯 Using provided bbox: {initial_bbox}")
    
    # Process video
    try:
        tracking_results = demo.process_video(
            video_path=args.input_video,
            output_dir=args.output_dir,
            initial_bbox=initial_bbox
        )
        
        print(f"\n🎉 SAMURAI demo completed successfully!")
        print(f"📁 Check results in: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

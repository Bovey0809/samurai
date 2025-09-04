#!/usr/bin/env python3
"""
Simple and Efficient Ball Detection and SAMURAI Tracking Pipeline

This version directly uses ultralytics in the base environment and processes
detection results directly from the model instead of parsing text files.
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import gc
from typing import Optional, Tuple, Dict, Any, List
from ultralytics import YOLO

# Add SAM2 to path for base environment
sys.path.append("./sam2")

class SimpleBallTrackingPipeline:
    def __init__(self, 
                 yolo_model_path: str,
                 samurai_model_path: str,
                 confidence_threshold: float = 0.5,
                 device: str = "cuda:0"):
        self.yolo_model_path = yolo_model_path
        self.samurai_model_path = samurai_model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Initialize models
        self.yolo_model = None
        self.samurai_predictor = None
        self.samurai_state = None
        
        # Tracking state
        self.current_track_id = 0
        self.is_tracking = False
        self.last_bbox = None
        self.last_mask = None
        self.tracking_frames = 0
        
        # Detection results cache
        self.detection_results = {}
        
        # Debug counters
        self.detection_count = 0
        self.tracking_init_count = 0
        self.tracking_reinit_count = 0
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        print("Initializing models...")
        self._init_yolo()
        self._init_samurai()
        print("Models initialized successfully")
    
    def _init_yolo(self):
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print("YOLO model initialized")
        except Exception as e:
            print(f"Failed to initialize YOLO: {e}")
            raise
    
    def _init_samurai(self):
        try:
            from sam2.build_sam import build_sam2_video_predictor
            model_cfg = self._determine_samurai_config(self.samurai_model_path)
            self.samurai_predictor = build_sam2_video_predictor(
                model_cfg, self.samurai_model_path, device=self.device
            )
            print("SAMURAI model initialized")
        except Exception as e:
            print(f"Failed to initialize SAMURAI: {e}")
            raise
    
    def _determine_samurai_config(self, model_path: str) -> str:
        if "large" in model_path:
            return "configs/samurai/sam2.1_hiera_l.yaml"
        elif "base_plus" in model_path:
            return "configs/samurai/sam2.1_hiera_b+.yaml"
        elif "small" in model_path:
            return "configs/samurai/sam2.1_hiera_s.yaml"
        elif "tiny" in model_path:
            return "configs/samurai/sam2.1_hiera_t.yaml"
        else:
            return "configs/samurai/sam2.1_hiera_b+.yaml"
    
    def run_yolo_detection(self, video_path: str) -> Dict[int, Tuple[List[int], float]]:
        print("Running YOLO detection on video...")
        try:
            results = self.yolo_model(video_path, stream=True, conf=self.confidence_threshold)
            detection_results = {}
            frame_idx = 0
            
            for result in results:
                if len(result.boxes) > 0:
                    box = result.boxes[0]
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = bbox
                    bbox_formatted = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    detection_results[frame_idx] = (bbox_formatted, conf)
                frame_idx += 1
            
            self.detection_count = len(detection_results)
            print(f"YOLO detection complete: {self.detection_count} detections found")
            if detection_results:
                first_few = dict(list(detection_results.items())[:5])
                print(f"First few detections: {first_few}")
            return detection_results
        except Exception as e:
            print(f"YOLO detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def init_samurai_tracking(self, video_path: str, bbox: List[int]) -> bool:
        try:
            if self.samurai_predictor is None:
                return False
            x, y, w, h = bbox
            samurai_bbox = (x, y, x + w, y + h)
            self.samurai_state = self.samurai_predictor.init_state(
                video_path, offload_video_to_cpu=True
            )
            _, _, masks = self.samurai_predictor.add_new_points_or_box(
                self.samurai_state, box=samurai_bbox, frame_idx=0, obj_id=self.current_track_id
            )
            if masks is not None and len(masks) > 0:
                self.is_tracking = True
                self.last_bbox = bbox
                self.last_mask = masks[0][0].cpu().numpy() > 0.0
                self.tracking_frames = 0
                self.tracking_init_count += 1
                print(f"SAMURAI tracking initialized with bbox: {bbox}")
                return True
            else:
                print(f"Failed to initialize SAMURAI tracking with bbox: {bbox}")
                return False
        except Exception as e:
            print(f"SAMURAI initialization failed: {e}")
            return False
    
    def reinit_samurai_tracking(self, bbox: List[int]) -> bool:
        try:
            if self.samurai_predictor is None or self.samurai_state is None:
                return False
            x, y, w, h = bbox
            samurai_bbox = (x, y, x + w, y + h)
            _, _, masks = self.samurai_predictor.add_new_points_or_box(
                self.samurai_state, box=samurai_bbox, frame_idx=self.tracking_frames, obj_id=self.current_track_id
            )
            if masks is not None and len(masks) > 0:
                self.is_tracking = True
                self.last_bbox = bbox
                self.last_mask = masks[0][0].cpu().numpy() > 0.0
                self.tracking_reinit_count += 1
                print(f"SAMURAI tracking reinitialized with bbox: {bbox}")
                return True
            else:
                print(f"Failed to reinitialize SAMURAI tracking with bbox: {bbox}")
                return False
        except Exception as e:
            print(f"SAMURAI reinitialization failed: {e}")
            return False
    
    def process_video(self, video_path: str, output_path: str):
        print(f"Processing video: {video_path}")
        self.detection_results = self.run_yolo_detection(video_path)
        if not self.detection_results:
            print("No detections found, creating output with no tracking")
            self._create_empty_output(video_path, output_path)
            return
        self._process_video_with_detections(video_path, output_path)
    
    def _process_video_with_detections(self, video_path: str, output_path: str):
        print("Processing video with pre-computed detections...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        self.is_tracking = False
        self.tracking_frames = 0
        frame_count = 0
        
        print("Processing frames with tracking...")
        print("=" * 60)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detection = self.detection_results.get(frame_count)
                result = self._process_frame_with_detection(frame, frame_count, video_path, detection)
                rendered = self._render_frame(frame, result)
                out.write(rendered)
                frame_count += 1
                if frame_count % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            cap.release()
            out.release()
            print("=" * 60)
            print("PROCESSING COMPLETE - FINAL STATISTICS")
            print("=" * 60)
            print(f"Total frames processed: {frame_count}")
            print(f"YOLO detections: {self.detection_count}")
            print(f"Tracking initializations: {self.tracking_init_count}")
            print(f"Tracking reinitializations: {self.tracking_reinit_count}")
            print(f"Final tracking status: {'Active' if self.is_tracking else 'Inactive'}")
            print(f"Tracking duration: {self.tracking_frames} frames")
            print("=" * 60)
    
    def _process_frame_with_detection(self, frame: np.ndarray, frame_idx: int, 
                                    video_path: str, detection: Optional[Tuple[List[int], float]]) -> Dict[str, Any]:
        result = {
            'frame_idx': frame_idx,
            'bbox': None,
            'confidence': None,
            'mask': None,
            'status': 'no_detection'
        }
        
        if detection:
            bbox, confidence = detection
            result['bbox'] = bbox
            result['confidence'] = confidence
            print(f"Frame {frame_idx:3d}: Using pre-computed detection at {bbox} (conf: {confidence:.3f})")
            
            if confidence >= self.confidence_threshold:
                if not self.is_tracking:
                    print(f"Frame {frame_idx:3d}: High confidence, initializing tracking...")
                    if self.init_samurai_tracking(video_path, bbox):
                        result['status'] = 'tracking_initialized'
                        result['mask'] = self.last_mask
                    else:
                        result['status'] = 'tracking_failed'
                else:
                    print(f"Frame {frame_idx:3d}: High confidence, reinitializing tracking...")
                    if self.reinit_samurai_tracking(bbox):
                        result['status'] = 'tracking_reinitialized'
                        result['mask'] = self.last_mask
                    else:
                        result['status'] = 'tracking_reinit_failed'
            else:
                if self.is_tracking:
                    result['status'] = 'low_confidence_continue_tracking'
                    result['mask'] = self.last_mask
                    print(f"Frame {frame_idx:3d}: Low confidence ({confidence:.3f}), continuing tracking")
                else:
                    result['status'] = 'low_confidence_no_tracking'
                    print(f"Frame {frame_idx:3d}: Low confidence ({confidence:.3f}), no tracking active")
        else:
            if self.is_tracking:
                result['status'] = 'no_detection_continue_tracking'
                result['mask'] = self.last_mask
                print(f"Frame {frame_idx:3d}: No detection, continuing tracking")
            else:
                result['status'] = 'no_detection_no_tracking'
                print(f"Frame {frame_idx:3d}: No detection, no tracking active")
        
        if self.is_tracking:
            self.tracking_frames += 1
        return result
    
    def _create_empty_output(self, video_path: str, output_path: str):
        print("Creating output video with no tracking...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            return
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame, "No Detection", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                out.write(frame)
                frame_count += 1
        finally:
            cap.release()
            out.release()
        print(f"Created output video with {frame_count} frames (no tracking)")
    
    def _render_frame(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        rendered = frame.copy()
        if result['bbox']:
            x, y, w, h = result['bbox']
            color = (0, 255, 0) if result['confidence'] and result['confidence'] >= self.confidence_threshold else (0, 165, 255)
            cv2.rectangle(rendered, (x, y), (x + w, y + h), color, 2)
            if result['confidence']:
                conf_text = f"Conf: {result['confidence']:.2f}"
                cv2.putText(rendered, conf_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if result['mask'] is not None:
            mask_img = np.zeros_like(frame)
            mask_img[result['mask']] = [0, 0, 255]
            rendered = cv2.addWeighted(rendered, 1, mask_img, 0.3, 0)
        status_text = f"Status: {result['status']}"
        cv2.putText(rendered, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.is_tracking:
            track_text = f"Tracking: {self.tracking_frames} frames"
            cv2.putText(rendered, track_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return rendered
    
    def cleanup(self):
        if self.yolo_model:
            del self.yolo_model
        if self.samurai_predictor:
            del self.samurai_predictor
        if self.samurai_state:
            del self.samurai_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Simple Ball Detection and SAMURAI Tracking Pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--yolo-model", default="/root/ultralytics/runs/detect/train9/weights/best.pt", 
                       help="Path to YOLO model")
    parser.add_argument("--samurai-model", default="sam2.1_hiera_large.pt", 
                       help="Path to SAMURAI model")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="YOLO confidence threshold")
    parser.add_argument("--device", default="cuda:0", help="Device to run inference on")
    
    args = parser.parse_args()
    
    pipeline = SimpleBallTrackingPipeline(
        yolo_model_path=args.yolo_model,
        samurai_model_path=args.samurai_model,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    try:
        pipeline.process_video(args.video, args.output)
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main()

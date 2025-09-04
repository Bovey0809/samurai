import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def get_video_info(video_path):
    """Get video dimensions and frame rate without loading all frames"""
    if osp.isdir(video_path):
        # For directory of frames, get info from first frame
        frames = sorted([osp.join(video_path, f) for f in os.listdir(video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
        if not frames:
            raise ValueError("No image files found in directory")
        first_frame = cv2.imread(frames[0])
        height, width = first_frame.shape[:2]
        frame_rate = 30  # Default frame rate for image sequences
        total_frames = len(frames)
    else:
        # For video file, get info from video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    return width, height, frame_rate, total_frames

def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)

    # Get video info without loading all frames
    width, height, frame_rate, total_frames = get_video_info(args.video_path)
    print(f"Video info: {width}x{height}, {frame_rate} FPS, {total_frames} frames")

    # Initialize video writer if needed
    out = None
    if args.save_to_video:
        # Try different codecs for video output
        codecs_to_try = ['mp4v', 'XVID', 'MJPG', 'X264']
        
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))
                if out.isOpened():
                    print(f"Successfully opened video writer with codec: {codec}")
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                print(f"Failed to use codec {codec}: {e}")
                if out:
                    out.release()
                    out = None
        
        if out is None:
            print("Warning: Could not initialize video writer. Video output will be disabled.")
            args.save_to_video = False

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # Initialize state with streaming support
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        # Check if first frame tracking was successful
        if masks is None or len(masks) == 0:
            print(f"ERROR: Failed to track object in first frame. Bbox: {bbox}, Track label: {track_label}")
            print("Tracking failed - no masks generated for initial object")
            return
        else:
            print(f"SUCCESS: First frame tracking successful. Generated {len(masks)} mask(s) for bbox: {bbox}")

        # Process frames in streaming mode
        frame_count = 0
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if args.save_to_video and out is not None:
                # Load only the current frame for visualization
                if osp.isdir(args.video_path):
                    frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
                    if frame_idx < len(frames):
                        img = cv2.imread(frames[frame_idx])
                    else:
                        print(f"Warning: Frame {frame_idx} not found, skipping visualization")
                        continue
                else:
                    # For video file, read the specific frame
                    cap = cv2.VideoCapture(args.video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, img = cap.read()
                    cap.release()
                    if not ret:
                        print(f"Warning: Could not read frame {frame_idx}, skipping visualization")
                        continue

                # Apply masks and bounding boxes
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                out.write(img)
                
                # Clear frame from memory
                del img
                if 'mask_img' in locals():
                    del mask_img

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
                # Force garbage collection periodically
                gc.collect()
                torch.cuda.empty_cache()

        print(f"Total frames processed: {frame_count}")

        if args.save_to_video and out is not None:
            out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    args = parser.parse_args()
    main(args)

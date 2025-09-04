# Ball Detection and SAMURAI Tracking Pipeline

This pipeline integrates YOLO ball detection with SAMURAI tracking to create a robust ball tracking system that can handle detection failures gracefully.

## Overview

The pipeline implements the logic described in `pipeline.md`:

1. **YOLO Detection**: Detects balls in each frame using a trained YOLO model
2. **SAMURAI Tracking**: Uses SAMURAI for precise object tracking and segmentation
3. **Smart Integration**: Combines both systems to handle detection failures and maintain tracking continuity

## Architecture

```
Video Input → YOLO Detection → SAMURAI Tracking → Rendered Output
     ↓              ↓              ↓              ↓
   Frame      Bounding Box    Tracking Mask   Final Video
   Stream      + Confidence    + Continuity    + Overlays
```

## Key Features

- **Environment Isolation**: YOLO runs in `ultralytics` environment, SAMURAI in `base` environment
- **Confidence-Based Tracking**: High-confidence detections reinitialize tracking, low-confidence ones continue existing tracking
- **Graceful Degradation**: Continues tracking even when YOLO detection fails
- **Real-time Visualization**: Shows bounding boxes, confidence scores, tracking masks, and status information
- **Memory Efficient**: Processes frames sequentially with periodic cleanup

## Requirements

### System Requirements
- CUDA-capable GPU
- Python 3.10+
- Conda environments: `base` and `ultralytics`

### Model Files
- **YOLO Model**: `/root/ultralytics/runs/detect/train9/weights/best.pt`
- **SAMURAI Model**: `sam2.1_hiera_large.pt` (or other available models)

## Installation

1. **Ensure both conda environments are available**:
   ```bash
   conda env list
   # Should show: base and ultralytics
   ```

2. **Verify model files exist**:
   ```bash
   ls -la /root/ultralytics/runs/detect/train9/weights/best.pt
   ls -la sam2.1_hiera_large.pt
   ```

3. **Test the setup**:
   ```bash
   python test_pipeline.py
   ```

## Usage

### Basic Usage

```bash
python ball_tracking_pipeline_v2.py \
    --video input_video.mp4 \
    --output output_video.mp4 \
    --confidence 0.5
```

### Advanced Usage

```bash
python ball_tracking_pipeline_v2.py \
    --video input_video.mp4 \
    --output output_video.mp4 \
    --yolo-model /path/to/custom/yolo.pt \
    --samurai-model sam2.1_hiera_base_plus.pt \
    --confidence 0.7 \
    --device cuda:0
```

### Command Line Arguments

- `--video`: Input video file path (required)
- `--output`: Output video file path (required)
- `--yolo-model`: Path to YOLO model (default: `/root/ultralytics/runs/detect/train9/weights/best.pt`)
- `--samurai-model`: Path to SAMURAI model (default: `sam2.1_hiera_large.pt`)
- `--confidence`: YOLO confidence threshold (default: 0.5)
- `--device`: Device for inference (default: `cuda:0`)

## Pipeline Logic

The pipeline implements the following decision logic:

```python
if bbox & confidence >= threshold:
    if not tracking:
        initialize_samurai_tracking(bbox)
    else:
        reinitialize_samurai_tracking(bbox)
elif bbox & confidence < threshold:
    if tracking:
        continue_tracking()
    else:
        wait_for_better_detection()
else:  # no bbox
    if tracking:
        continue_tracking()
    else:
        wait_for_detection()
```

## Output Visualization

The rendered video includes:

- **Bounding Boxes**: 
  - Green: High-confidence detections (≥ threshold)
  - Orange: Low-confidence detections (< threshold)
- **Tracking Masks**: Red overlay showing SAMURAI segmentation
- **Status Information**: Current pipeline state and tracking duration
- **Confidence Scores**: YOLO detection confidence for each bbox

## Status Messages

The pipeline provides detailed status information:

- `tracking_initialized`: First successful detection and tracking start
- `tracking_reinitialized`: High-confidence detection reinitialized tracking
- `low_confidence_continue_tracking`: Low-confidence detection, continuing existing tracking
- `no_detection_continue_tracking`: No detection, continuing existing tracking
- `tracking_failed`: Failed to initialize tracking
- `tracking_lost`: Tracking was lost and needs reinitialization

## Performance Considerations

- **Memory Usage**: SAMURAI models can be memory-intensive; ensure sufficient GPU memory
- **Processing Speed**: YOLO detection + SAMURAI tracking per frame; expect 1-5 FPS depending on hardware
- **Batch Processing**: Consider processing videos in chunks for very long videos

## Troubleshooting

### Common Issues

1. **YOLO Environment Not Available**:
   ```bash
   conda activate ultralytics
   python -c "from ultralytics import YOLO; print('OK')"
   ```

2. **SAMURAI Model Not Found**:
   ```bash
   ls -la sam2.1_hiera_*.pt
   # Download appropriate model if missing
   ```

3. **CUDA Out of Memory**:
   - Reduce video resolution
   - Use smaller SAMURAI model (e.g., `base_plus` instead of `large`)
   - Process shorter video segments

4. **Video Codec Issues**:
   - Try different output codecs: `mp4v`, `XVID`, `MJPG`
   - Ensure input video is in a supported format

### Debug Mode

Enable verbose logging by modifying the pipeline script:

```python
# Add debug prints in _process_frame method
print(f"Frame {frame_idx}: Detection={detection}, Tracking={self.is_tracking}")
```

## Testing

Run the test suite to verify all components work:

```bash
python test_pipeline.py
```

This will test:
- YOLO environment availability
- SAMURAI model initialization
- Video I/O functionality

## Future Improvements

- **Multi-object Tracking**: Support for tracking multiple balls simultaneously
- **Kalman Filtering**: Add motion prediction for better tracking stability
- **Real-time Processing**: Optimize for live video streams
- **Custom Visualization**: User-configurable overlay styles and information display

## Contributing

When modifying the pipeline:

1. Test with the provided test suite
2. Maintain the existing API structure
3. Add appropriate error handling and logging
4. Update this documentation

## License

This pipeline is part of the SAMURAI project. See the main project license for details.

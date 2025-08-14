# SAMURAI Demo - Single Object Tracking

SAMURAI is a single object tracking algorithm based on SAM2 (Segment Anything 2). This demo allows you to track objects in MP4 videos and output both processed videos with visualization and detailed bounding box data.

## 🎯 Features

- **Single Object Tracking**: Track one object throughout a video sequence
- **MP4 Input Support**: Process standard MP4 video files
- **Multiple Output Formats**: 
  - Processed video with tracking visualization
  - CSV file with bounding box data for each frame
  - JSON file with detailed tracking results
  - Summary report
- **Flexible Initialization**: Automatic center-based initialization or manual bbox specification
- **GPU Acceleration**: CUDA-enabled for fast processing

## 📋 Requirements

- Python 3.10+
- PyTorch with CUDA support
- OpenCV
- SAM2 environment (see setup instructions below)

## 🚀 Quick Start

### 1. Setup Environment

First, ensure you have the SAM2 environment set up:

```bash
# Source environment variables
source setup_env_vars.sh

# Test SAM2 installation
python sam2/test_demo.py
```

### 2. Run the Demo

#### Basic Usage
```bash
python samurai_demo.py --input_video your_video.mp4 --output_dir results/
```

#### With Custom Initial Bounding Box
```bash
python samurai_demo.py \
    --input_video your_video.mp4 \
    --output_dir results/ \
    --bbox 100 100 200 200  # x1 y1 x2 y2
```

#### With Different Model
```bash
python samurai_demo.py \
    --input_video your_video.mp4 \
    --output_dir results/ \
    --model_path sam2/checkpoints/sam2.1_hiera_large.pt
```

### 3. Test the Demo

Run the test script to verify everything works:

```bash
# Create a test video and run demo
python test_samurai_demo.py

# Or test with your own video
python test_samurai_demo.py --video your_video.mp4

# Just create a test video
python test_samurai_demo.py --create_test_video
```

## 📁 Output Structure

After running the demo, you'll get the following files in your output directory:

```
results/
├── frames/                    # Extracted video frames
├── samurai_tracking.mp4      # Processed video with tracking visualization
├── tracking_results.csv      # Bounding box data in CSV format
├── tracking_results.json     # Detailed results in JSON format
└── summary.txt              # Processing summary
```

### CSV Output Format

The `tracking_results.csv` file contains:
- `frame_idx`: Frame number (0-based)
- `object_id`: Object identifier (usually 0 for single object tracking)
- `x, y`: Top-left corner coordinates
- `width, height`: Bounding box dimensions
- `confidence`: Confidence score (1.0 for SAM2)

### JSON Output Format

The `tracking_results.json` file contains detailed information:
```json
[
  {
    "frame_idx": 0,
    "objects": {
      "0": {
        "bbox": [x, y, width, height],
        "mask": [[...]],  // Binary mask array
        "confidence": 1.0
      }
    }
  }
]
```

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_video` | str | Required | Input MP4 video file path |
| `--output_dir` | str | Required | Output directory for results |
| `--model_path` | str | `sam2/checkpoints/sam2.1_hiera_base_plus.pt` | SAM2 model checkpoint path |
| `--device` | str | `cuda:0` | Device for inference (`cuda:0`, `cpu`) |
| `--bbox` | int[4] | None | Initial bounding box (x1 y1 x2 y2) |

### Available Models

- `sam2.1_hiera_tiny.pt` - Fastest, lowest accuracy
- `sam2.1_hiera_small.pt` - Fast, good accuracy
- `sam2.1_hiera_base_plus.pt` - Balanced (default)
- `sam2.1_hiera_large.pt` - Slowest, highest accuracy

## 🎬 Demo Examples

### Example 1: Basic Tracking
```bash
# Track object in center of frame
python samurai_demo.py \
    --input_video sample_video.mp4 \
    --output_dir tracking_results/
```

### Example 2: Custom Object Selection
```bash
# Track specific object with manual bbox
python samurai_demo.py \
    --input_video sample_video.mp4 \
    --output_dir tracking_results/ \
    --bbox 150 200 300 400
```

### Example 3: High-Quality Tracking
```bash
# Use large model for best accuracy
python samurai_demo.py \
    --input_video sample_video.mp4 \
    --output_dir tracking_results/ \
    --model_path sam2/checkpoints/sam2.1_hiera_large.pt
```

## 🔧 Advanced Usage

### Batch Processing

For processing multiple videos:

```bash
#!/bin/bash
for video in videos/*.mp4; do
    basename=$(basename "$video" .mp4)
    python samurai_demo.py \
        --input_video "$video" \
        --output_dir "results/$basename/"
done
```

### Custom Initialization

You can modify the `get_initial_bbox()` method in `samurai_demo.py` to implement custom object selection:

```python
def get_initial_bbox(self, video_path, frame_dir):
    # Load first frame
    first_frame = cv2.imread(os.path.join(frame_dir, "00000000.jpg"))
    
    # Show frame and let user click to select object
    # Or use object detection to find objects
    # Or implement other selection methods
    
    return bbox
```

### Performance Optimization

For better performance:

1. **Use smaller models** for faster processing
2. **Reduce video resolution** if needed
3. **Use SSD storage** for faster frame extraction
4. **Ensure sufficient GPU memory** for large videos

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use smaller model (tiny/small)
   - Reduce video resolution
   - Use CPU mode: `--device cpu`

2. **Video Codec Issues**
   - Ensure video is in MP4 format
   - Try re-encoding with H.264 codec

3. **Import Errors**
   - Ensure SAM2 is properly installed
   - Run from the correct directory
   - Check Python path

4. **Poor Tracking Results**
   - Use larger model for better accuracy
   - Provide better initial bounding box
   - Ensure object is clearly visible in first frame

### Debug Mode

For debugging, you can modify the demo script to add more verbose output:

```python
# Add to samurai_demo.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Metrics

Typical performance on RTX 4090:

| Model | FPS | Memory Usage | Accuracy |
|-------|-----|--------------|----------|
| Tiny | ~15 | 4GB | Good |
| Small | ~10 | 6GB | Better |
| Base+ | ~7 | 8GB | Best |
| Large | ~4 | 12GB | Excellent |

## 🤝 Contributing

To contribute to the SAMURAI demo:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This demo is based on SAM2 and follows the same license terms.

## 🙏 Acknowledgments

- SAM2 team for the base segmentation model
- SAMURAI authors for the tracking algorithm
- OpenCV and PyTorch communities

---

For more information about SAMURAI, see the original paper and repository.

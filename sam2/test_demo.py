#!/usr/bin/env python3
"""
Simple SAM2 Demo Test
This script tests SAM2 functionality from within the sam2 directory.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

print("🚀 Starting SAM2 Demo Test")
print("=" * 50)

# Test basic imports
print("\n🔍 Testing basic setup...")
print(f"  - PyTorch version: {torch.__version__}")
print(f"  - CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - GPU count: {torch.cuda.device_count()}")
    print(f"  - Device name: {torch.cuda.get_device_name()}")

# Test SAM2 imports
print("\n📦 Testing SAM2 imports...")
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("  ✅ SAM2 imports successful")
except ImportError as e:
    print(f"  ❌ Import error: {e}")
    sys.exit(1)

# Test model creation
print("\n🏗️  Testing model creation...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Using device: {device}")
    
    # Check checkpoint
    checkpoint_path = "/root/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    if os.path.exists(checkpoint_path):
        print(f"  - Found checkpoint: {checkpoint_path}")
        
        # Create model
        model = build_sam2(
            config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
            ckpt_path=checkpoint_path,
            device=device,
            mode="eval"
        )
        print("  ✅ Model created successfully")
        print(f"  - Model type: {type(model)}")
        print(f"  - Model device: {next(model.parameters()).device}")
    else:
        print(f"  ❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
        
except Exception as e:
    print(f"  ❌ Model creation failed: {e}")
    sys.exit(1)

# Test predictor creation
print("\n🎯 Testing predictor creation...")
try:
    # Create predictor using the model we already created
    predictor = SAM2ImagePredictor(sam_model=model)
    print("  ✅ Predictor created successfully")
except Exception as e:
    print(f"  ❌ Predictor creation failed: {e}")
    sys.exit(1)

# Test simple inference
print("\n🖼️  Testing simple inference...")
try:
    # Create a simple test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_image)
    print(f"  - Created test image: {test_image.size}")
    
    # Set image
    predictor.set_image(test_image)
    print("  ✅ Image set successfully")
    
    # Test point prompt
    input_point = np.array([[256, 256]])
    input_label = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    print(f"  ✅ Inference successful")
    print(f"  - Generated {len(masks)} masks")
    print(f"  - Mask shapes: {[mask.shape for mask in masks]}")
    print(f"  - Scores: {scores}")
    
except Exception as e:
    print(f"  ❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("🎉 All tests passed! SAM2 is working correctly.")
print("\n📋 Summary:")
print("  ✅ PyTorch with CUDA support")
print("  ✅ SAM2 model creation")
print("  ✅ SAM2 predictor creation")
print("  ✅ Basic inference with dummy image")
print("\n🚀 You can now use SAM2 for image segmentation tasks!")

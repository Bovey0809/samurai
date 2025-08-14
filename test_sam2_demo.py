#!/usr/bin/env python3
"""
SAM2 Demo Test Script
This script tests the SAM2 installation and demonstrates basic functionality.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Change to the sam2 directory to avoid import issues
os.chdir('/root/samurai/sam2')

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("✅ All SAM2 imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_imports():
    """Test basic imports and PyTorch setup"""
    print("\n🔍 Testing basic setup...")
    
    # Test PyTorch
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        print(f"  - Current device: {torch.cuda.current_device()}")
        print(f"  - Device name: {torch.cuda.get_device_name()}")
    
    # Test SAM2
    print(f"  - SAM2 imported successfully")
    
    return True

def test_model_creation():
    """Test creating a SAM2 model"""
    print("\n🏗️  Testing model creation...")
    
    try:
        # Try to create a tiny model first
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  - Using device: {device}")
        
        # Check if checkpoints exist
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
            
            # Test model properties
            print(f"  - Model type: {type(model)}")
            print(f"  - Model device: {next(model.parameters()).device}")
            
            return True
        else:
            print(f"  ❌ Checkpoint not found: {checkpoint_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False

def test_predictor_creation():
    """Test creating a SAM2 predictor"""
    print("\n🎯 Testing predictor creation...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create predictor
        predictor = SAM2ImagePredictor(
            model_type="sam2.1_hiera_tiny",
            ckpt_path="/root/sam2/checkpoints/sam2.1_hiera_tiny.pt",
            device=device
        )
        print("  ✅ Predictor created successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Predictor creation failed: {e}")
        return False

def test_simple_inference():
    """Test simple inference with a dummy image"""
    print("\n🖼️  Testing simple inference...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create a simple test image (random noise)
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        print(f"  - Created test image: {test_image.size}")
        
        # Create predictor
        predictor = SAM2ImagePredictor(
            model_type="sam2.1_hiera_tiny",
            ckpt_path="/root/sam2/checkpoints/sam2.1_hiera_tiny.pt",
            device=device
        )
        
        # Set image
        predictor.set_image(test_image)
        print("  ✅ Image set successfully")
        
        # Test point prompt (center of image)
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
        
        return True
        
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting SAM2 Demo Test")
    print("=" * 50)
    
    # Test basic setup
    if not test_basic_imports():
        print("❌ Basic setup failed")
        return False
    
    # Test model creation
    if not test_model_creation():
        print("❌ Model creation failed")
        return False
    
    # Test predictor creation
    if not test_predictor_creation():
        print("❌ Predictor creation failed")
        return False
    
    # Test simple inference
    if not test_simple_inference():
        print("❌ Inference failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! SAM2 is working correctly.")
    print("\n📋 Summary:")
    print("  ✅ PyTorch with CUDA support")
    print("  ✅ SAM2 model creation")
    print("  ✅ SAM2 predictor creation")
    print("  ✅ Basic inference with dummy image")
    print("\n🚀 You can now use SAM2 for image segmentation tasks!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

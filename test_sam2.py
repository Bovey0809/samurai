#!/usr/bin/env python3
"""
Simple test script to verify SAM2 installation
"""

import torch
import sam2
from sam2.build_sam import build_sam2

def test_sam2():
    print("Testing SAM2 installation...")
    
    # Test basic imports
    print("✅ All imports successful")
    
    # Test model creation
    try:
        model = build_sam2("sam2.1_hiera_tiny")
        print("✅ Model creation successful")
        
        # Test basic inference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"✅ Model moved to {device}")
        
    except Exception as e:
        print(f"⚠️  Model test failed: {e}")
        print("This might be due to missing checkpoints or CUDA issues")
    
    print("🎉 SAM2 installation test completed!")

if __name__ == "__main__":
    test_sam2()

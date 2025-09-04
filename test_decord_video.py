import torch

# Try to allocate a large tensor
try:
    # 96 GB tensor (same as your video)
    large_tensor = torch.zeros(8000, 3, 1024, 1024, dtype=torch.float32)
    print("Success! Large tensor allocation works")
    del large_tensor
except RuntimeError as e:
    print(f"Failed to allocate large tensor: {e}")
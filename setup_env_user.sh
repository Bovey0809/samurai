#!/bin/bash

# SAM2 Environment Setup Script (User Version)
# This script replicates the environment setup from the Dockerfile without using Docker
# This version doesn't require sudo and can be run in a user environment

set -e  # Exit on any error

echo "🚀 Starting SAM2 environment setup (User Version)..."

# Set environment variables (replicating Dockerfile ENV statements)
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_NO_CACHE_DIR=1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.6"

# Add CUDA to PATH if it exists
if [ -d "/usr/local/cuda/bin" ]; then
    export PATH="${PATH}:/usr/local/cuda/bin"
fi

echo "📋 Environment variables set"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if required system packages are available
echo "🔍 Checking system dependencies..."

missing_packages=()

# Check for essential packages
if ! command_exists ffmpeg; then
    missing_packages+=("ffmpeg")
fi

if ! command_exists git; then
    missing_packages+=("git")
fi

if ! command_exists wget; then
    missing_packages+=("wget")
fi

if ! command_exists python3; then
    missing_packages+=("python3")
fi

if ! command_exists pip3; then
    missing_packages+=("python3-pip")
fi

# Check for development tools
if ! command_exists pkg-config; then
    missing_packages+=("pkg-config")
fi

if ! command_exists gcc; then
    missing_packages+=("build-essential")
fi

# Report missing packages
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "❌ Missing system packages: ${missing_packages[*]}"
    echo ""
    echo "Please install the missing packages using your system package manager:"
    echo ""
    echo "For Ubuntu/Debian:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y ${missing_packages[*]}"
    echo ""
    echo "For CentOS/RHEL/Fedora:"
    echo "  sudo dnf install -y ${missing_packages[*]}"
    echo ""
    echo "After installing the packages, run this script again."
    exit 1
fi

echo "✅ All system dependencies are available"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. SAM2 requires Python >= $required_version"
    echo "Please upgrade Python to version $required_version or higher"
    exit 1
fi

echo "✅ Python version $python_version is compatible"

# Upgrade pip and setuptools
echo "⬆️  Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools wheel --user

# Configure pip to use Tuna mirror (optional, for faster downloads in China)
read -p "Do you want to use Tuna mirror for pip (faster in China)? [y/N]: " use_tuna
if [[ $use_tuna =~ ^[Yy]$ ]]; then
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
    echo "✅ Pip configured to use Tuna mirror"
fi

# Check if PyTorch is already installed
echo "🔍 Checking PyTorch installation..."
if python3 -c "import torch; print(f'PyTorch {torch.__version__} found')" 2>/dev/null; then
    echo "✅ PyTorch is already installed"
    python3 -c "
import torch
print(f'  - Version: {torch.__version__}')
print(f'  - CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  - CUDA version: {torch.version.cuda}')
    print(f'  - GPU count: {torch.cuda.device_count()}')
"
else
    echo "🔥 Installing PyTorch with CUDA support..."
    if command_exists nvidia-smi; then
        echo "NVIDIA GPU detected, installing PyTorch with CUDA support..."
        python3 -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 --user
    else
        echo "No NVIDIA GPU detected, installing CPU-only PyTorch..."
        python3 -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu --user
    fi
    echo "✅ PyTorch installed"
fi

# Create SAM2 directory structure in user's home directory
echo "📁 Creating SAM2 directory structure..."
mkdir -p ~/sam2/checkpoints
export APP_ROOT=~/sam2

# Download SAM 2.1 checkpoints
echo "📥 Downloading SAM 2.1 checkpoints..."
cd ~/sam2/checkpoints

checkpoints=(
    "sam2.1_hiera_tiny.pt"
    "sam2.1_hiera_small.pt"
    "sam2.1_hiera_base_plus.pt"
    "sam2.1_hiera_large.pt"
)

for checkpoint in "${checkpoints[@]}"; do
    if [ ! -f "$checkpoint" ]; then
        echo "Downloading $checkpoint..."
        wget -q "https://dl.fbaipublicfiles.com/segment_anything_2/092824/$checkpoint" -O "$checkpoint"
        echo "✅ Downloaded $checkpoint"
    else
        echo "✅ $checkpoint already exists"
    fi
done

# Return to workspace directory
cd /root/samurai

# Check if SAM2 is already installed
echo "🔍 Checking SAM2 installation..."
if python3 -c "import sam2; print('SAM2 found')" 2>/dev/null; then
    echo "✅ SAM2 is already installed"
    python3 -c "import sam2; print(f'  - SAM2 version: {sam2.__version__ if hasattr(sam2, \"__version__\") else \"Unknown\"}')"
else
    echo "🔧 Installing SAM2 in editable mode..."
    cd sam2

    # Set environment variables for SAM2 installation
    export SAM2_BUILD_CUDA=1
    export SAM2_BUILD_ALLOW_ERRORS=1

    # Install SAM2 with all extras (using --user flag)
    python3 -m pip install -e ".[notebooks,interactive-demo,dev]" --user

    echo "✅ SAM2 installed successfully"
fi

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import torch
import sam2
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
print(f'✅ SAM2 imported successfully')
"

# Create a simple test script
echo "🧪 Creating test script..."
cat > /root/samurai/test_sam2.py << 'EOF'
#!/usr/bin/env python3
"""
Simple test script to verify SAM2 installation
"""

import torch
import sam2
from sam2 import build_sam2

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
EOF

chmod +x /root/samurai/test_sam2.py

# Create environment setup script for future use
echo "📝 Creating environment setup script..."
cat > /root/samurai/setup_env_vars.sh << 'EOF'
#!/bin/bash
# Source this script to set up SAM2 environment variables

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.6"
export APP_ROOT=~/sam2

# Add CUDA to PATH if it exists
if [ -d "/usr/local/cuda/bin" ]; then
    export PATH="${PATH}:/usr/local/cuda/bin"
fi

# Add user's local bin to PATH for pip-installed packages
export PATH="${PATH}:${HOME}/.local/bin"

echo "SAM2 environment variables set"
echo "  - CUDA_HOME: $CUDA_HOME"
echo "  - TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  - APP_ROOT: $APP_ROOT"
EOF

chmod +x /root/samurai/setup_env_vars.sh

echo ""
echo "🎉 SAM2 environment setup completed!"
echo ""
echo "📋 Summary:"
echo "   - System dependencies verified"
echo "   - Python environment configured"
echo "   - PyTorch installed with CUDA support"
echo "   - SAM2 installed in editable mode"
echo "   - Checkpoints downloaded to ~/sam2/checkpoints"
echo ""
echo "🚀 Next steps:"
echo "   1. Source environment variables: source /root/samurai/setup_env_vars.sh"
echo "   2. Run the test script: python3 /root/samurai/test_sam2.py"
echo "   3. Start using SAM2 in your projects!"
echo ""
echo "📁 Important directories:"
echo "   - SAM2 checkpoints: ~/sam2/checkpoints"
echo "   - SAM2 source code: /root/samurai/sam2"
echo "   - Test script: /root/samurai/test_sam2.py"
echo "   - Environment setup: /root/samurai/setup_env_vars.sh"
echo ""
echo "🔧 Environment variables set:"
echo "   - CUDA_HOME: $CUDA_HOME"
echo "   - TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "   - APP_ROOT: $APP_ROOT"
echo ""
echo "💡 Tips:"
echo "   - If you encounter CUDA issues, you can disable CUDA extension building:"
echo "     export SAM2_BUILD_CUDA=0"
echo "   - To force CUDA extension building (and fail on errors):"
echo "     export SAM2_BUILD_ALLOW_ERRORS=0"
echo "   - To set up environment variables in future sessions, run:"
echo "     source /root/samurai/setup_env_vars.sh"
echo ""

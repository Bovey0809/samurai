#!/bin/bash

echo "🗡️  Starting SAMURAI Web Demo..."
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python3 -c "import flask, cv2, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    python3 -m pip install -r requirements_web.txt --user
fi

# Check if SAM2 is available
if [ ! -f "/root/sam2/checkpoints/sam2.1_hiera_base_plus.pt" ]; then
    echo "⚠️  SAM2 checkpoint not found. Please ensure SAM2 is properly installed."
    echo "   Expected path: /root/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
fi

# Start the web server
echo "🚀 Starting Flask server..."
echo "📱 Open your browser to: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python3 web_demo.py

# 🗡️ SAMURAI Web Demo

A beautiful, interactive web interface for single object tracking using SAM2 (Segment Anything 2).

## 🌟 Features

- **Drag & Drop Upload**: Easy video file upload with drag-and-drop support
- **Interactive Object Selection**: Click and drag to select objects in the first frame
- **Real-time Processing**: Live progress tracking during video processing
- **Beautiful UI**: Modern, responsive design with smooth animations
- **Multiple Video Formats**: Support for MP4, AVI, MOV, MKV, and WEBM
- **Session Management**: Unique session IDs for each tracking job
- **Result Visualization**: Display tracking results with bounding boxes

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install web demo requirements
python3 -m pip install -r requirements_web.txt --user

# Or install individually
python3 -m pip install flask werkzeug opencv-python numpy tqdm --user
```

### 2. Test the Setup

```bash
# Run the test suite
python3 test_web_demo.py
```

### 3. Start the Web Server

```bash
# Start the Flask application
python3 web_demo.py
```

### 4. Open in Browser

Navigate to: **http://localhost:5000**

## 📖 Usage Guide

### Step 1: Upload Video
1. Click the upload area or drag and drop your video file
2. Supported formats: MP4, AVI, MOV, MKV, WEBM (max 100MB)
3. Click "Upload Video" to proceed

### Step 2: Select Object
1. The first frame of your video will be displayed
2. Click and drag to draw a rectangle around the object you want to track
3. The selected bounding box coordinates will be shown
4. Click "Start Tracking" to begin processing

### Step 3: View Results
1. Wait for processing to complete (progress bar shows status)
2. The tracking result video will be displayed with bounding boxes
3. Tracking summary information is shown below the video

## 🏗️ Architecture

### File Structure
```
samurai/
├── web_demo.py              # Main Flask application
├── templates/
│   └── index.html          # Web interface template
├── static/                 # Static files (images, videos)
├── uploads/                # Uploaded video files
├── results/                # Processing results
├── requirements_web.txt    # Web dependencies
├── test_web_demo.py       # Test suite
└── WEB_DEMO_README.md     # This file
```

### Key Components

#### Flask Application (`web_demo.py`)
- **Upload Handler**: Processes video uploads and extracts first frames
- **Tracking API**: Handles object tracking requests
- **Session Management**: Creates unique sessions for each user
- **File Management**: Organizes uploads and results

#### Web Interface (`templates/index.html`)
- **Modern UI**: Beautiful, responsive design
- **Interactive Canvas**: Object selection with mouse events
- **Progress Tracking**: Real-time processing status
- **Error Handling**: User-friendly error messages

#### Backend Integration
- **SAMURAI Tracker**: Integrates with the core tracking algorithm
- **Video Processing**: Frame extraction and result generation
- **File I/O**: Efficient file handling and cleanup

## 🔧 Configuration

### Environment Variables
```bash
# Flask configuration
export FLASK_ENV=development
export FLASK_DEBUG=1

# File size limits
export MAX_CONTENT_LENGTH=100MB
```

### Customization Options

#### Change Port
Edit `web_demo.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # Change port here
```

#### Modify File Size Limit
Edit `web_demo.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
```

#### Add Video Formats
Edit `web_demo.py`:
```python
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}  # Add formats
```

## 🧪 Testing

### Run Test Suite
```bash
python3 test_web_demo.py
```

### Manual Testing
1. Start the web server: `python3 web_demo.py`
2. Open browser to `http://localhost:5000`
3. Upload a test video
4. Select an object and start tracking
5. Verify results are displayed correctly

### Test Coverage
- ✅ Dependency verification
- ✅ Video processing functionality
- ✅ Web app imports
- ✅ Flask application creation
- ✅ File upload handling
- ✅ Frame extraction
- ✅ Object selection interface

## 🐛 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Install missing dependencies
python3 -m pip install flask werkzeug opencv-python --user
```

#### Port already in use
```bash
# Find and kill process using port 5000
lsof -ti:5000 | xargs kill -9

# Or use a different port
python3 web_demo.py --port 8080
```

#### File upload fails
- Check file size (max 100MB)
- Verify file format is supported
- Ensure sufficient disk space

#### Tracking fails
- Verify SAM2 installation
- Check checkpoint files exist
- Review error logs in browser console

### Debug Mode
```bash
# Enable debug mode for detailed error messages
export FLASK_DEBUG=1
python3 web_demo.py
```

## 🔒 Security Considerations

### File Upload Security
- File type validation
- File size limits
- Secure filename handling
- Temporary file cleanup

### Session Security
- Unique session IDs
- Session isolation
- Automatic cleanup of old sessions

### Production Deployment
For production use, consider:
- HTTPS encryption
- User authentication
- Rate limiting
- Load balancing
- Database integration

## 📊 Performance

### Optimization Tips
- Use GPU acceleration when available
- Implement video compression
- Add caching for processed results
- Optimize image processing pipeline

### Monitoring
- Track processing times
- Monitor memory usage
- Log user interactions
- Monitor error rates

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python3 test_web_demo.py`
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Write unit tests

## 📄 License

This web demo is part of the SAMURAI project and follows the same license terms.

## 🙏 Acknowledgments

- **Flask**: Web framework
- **OpenCV**: Video processing
- **SAM2**: Core tracking algorithm
- **Modern Web Technologies**: HTML5, CSS3, JavaScript

---

**Ready to track objects? Start the web demo and let SAMURAI do the work! 🗡️**

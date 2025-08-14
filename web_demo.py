#!/usr/bin/env python3
"""
SAMURAI Web Demo - Single Object Tracking
A Flask web application for interactive video object tracking using SAM2
"""

import os
import cv2
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import tempfile
import shutil
from datetime import datetime
import uuid
from samurai_demo import SAMURAIDemo

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_first_frame(video_path):
    """Extract the first frame from a video for object selection"""
    print(f"Extracting frame from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, None
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Save first frame for display - use proper path construction
        base_path = os.path.splitext(video_path)[0]  # Remove any extension
        frame_path = f"{base_path}_first_frame.jpg"
        print(f"Saving frame to: {frame_path}")
        
        success = cv2.imwrite(frame_path, frame)
        if not success:
            print(f"Failed to save frame to: {frame_path}")
            return None, None
            
        print(f"Frame saved successfully: {frame_path}, shape: {frame.shape[:2]}")
        return frame_path, frame.shape[:2]  # Return path and dimensions
    
    print("Failed to read frame from video")
    return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and extract first frame"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, or WEBM'}), 400
    
    # Create unique session ID
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    # Save uploaded video
    filename = secure_filename(file.filename)
    video_path = os.path.join(session_folder, filename)
    file.save(video_path)
    
    # Extract first frame
    try:
        frame_path, dimensions = extract_first_frame(video_path)
        if frame_path is None:
            return jsonify({'error': 'Could not extract frame from video'}), 400
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return jsonify({'error': f'Frame extraction failed: {str(e)}'}), 500
    
    # Copy frame to static folder for web display
    static_frame_path = os.path.join('static', f'{session_id}_first_frame.jpg')
    shutil.copy2(frame_path, static_frame_path)
    
    return jsonify({
        'session_id': session_id,
        'frame_url': url_for('static', filename=f'{session_id}_first_frame.jpg'),
        'dimensions': dimensions,
        'video_path': video_path
    })

@app.route('/track', methods=['POST'])
def track_object():
    """Handle object tracking request"""
    data = request.get_json()
    session_id = data.get('session_id')
    bbox = data.get('bbox')  # [x, y, width, height]
    
    if not session_id or not bbox:
        return jsonify({'error': 'Missing session_id or bbox'}), 400
    
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    video_path = None
    
    # Find the video file in session folder
    for file in os.listdir(session_folder):
        if file.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            video_path = os.path.join(session_folder, file)
            break
    
    if not video_path:
        return jsonify({'error': 'Video file not found'}), 400
    
    try:
        print(f"🎯 Starting tracking for session: {session_id}")
        print(f"📹 Video path: {video_path}")
        print(f"🎯 Bounding box: {bbox}")
        
        # Initialize SAMURAI demo
        demo = SAMURAIDemo()
        
        # Run tracking
        print("🚀 Running SAMURAI tracking...")
        results = demo.process_video(
            video_path=video_path,
            output_dir=os.path.join(app.config['RESULTS_FOLDER'], session_id),
            initial_bbox=tuple(bbox)
        )
        
        # Generate result URLs - check for different video formats
        result_video_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'samurai_tracking.mp4')
        if not os.path.exists(result_video_path):
            result_video_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'samurai_tracking.avi')
        
        static_video_path = os.path.join('static', f'{session_id}_result{os.path.splitext(result_video_path)[1]}')
        
        print(f"📁 Copying result from {result_video_path} to {static_video_path}")
        shutil.copy2(result_video_path, static_video_path)
        
        print(f"✅ Tracking completed successfully for session: {session_id}")
        
        return jsonify({
            'success': True,
            'result_video_url': url_for('static', filename=f'{session_id}_result.mp4'),
            'tracking_data': results
        })
        
    except Exception as e:
        print(f"❌ Tracking failed for session {session_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Tracking failed: {str(e)}'}), 500

@app.route('/check_result/<session_id>')
def check_result(session_id):
    """Check if tracking result exists for a session"""
    result_video_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'samurai_tracking.mp4')
    if not os.path.exists(result_video_path):
        result_video_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'samurai_tracking.avi')
    
    if os.path.exists(result_video_path):
        # Copy to static if not already there
        video_ext = os.path.splitext(result_video_path)[1]
        static_video_path = os.path.join('static', f'{session_id}_result{video_ext}')
        
        if not os.path.exists(static_video_path):
            shutil.copy2(result_video_path, static_video_path)
        
        return jsonify({
            'success': True,
            'result_video_url': url_for('static', filename=f'{session_id}_result{video_ext}'),
            'message': 'Tracking result found'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Tracking result not found'
        })

@app.route('/static/<filename>')
def static_files(filename):
    """Serve static files"""
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

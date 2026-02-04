"""
Mockup Displacement API

Flask API for generating photorealistic t-shirt mockups using
OpenCV displacement mapping (Photoshop-style quality).
"""

import os
import time
import uuid
import logging
from io import BytesIO

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from displacement import generate_mockup, create_displacement_map

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Lovable frontend

# Configuration
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/tmp/outputs')
MAX_IMAGE_SIZE = 4000  # Max dimension in pixels
REQUEST_TIMEOUT = 30  # Seconds
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_image(url, timeout=15):
    """
    Download image from URL and decode to OpenCV format.

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        OpenCV image (BGR or BGRA)

    Raises:
        ValueError: If download or decode fails
    """
    try:
        logger.info(f"Downloading: {url[:80]}...")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # Decode image from bytes
        img_array = np.frombuffer(response.content, np.uint8)

        # Try to decode with alpha channel first (for PNG transparency)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to decode image from URL: {url}")

        # Convert RGBA to BGRA if needed (OpenCV uses BGR order)
        if len(img.shape) == 2:
            # Grayscale, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        logger.info(f"Downloaded image: {img.shape}")
        return img

    except requests.RequestException as e:
        raise ValueError(f"Failed to download image: {str(e)}")


def validate_request(data):
    """
    Validate incoming request data.

    Args:
        data: Request JSON data

    Returns:
        Tuple (is_valid, error_message)
    """
    required_fields = ['baseImageUrl', 'designImageUrl', 'position']

    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    position = data['position']
    if not isinstance(position, dict) or 'x' not in position or 'y' not in position:
        return False, "Position must be an object with 'x' and 'y' properties"

    if not isinstance(position['x'], (int, float)) or not isinstance(position['y'], (int, float)):
        return False, "Position x and y must be numbers"

    if 'scale' in data and (not isinstance(data['scale'], (int, float)) or data['scale'] <= 0):
        return False, "Scale must be a positive number"

    if 'displacementStrength' in data:
        strength = data['displacementStrength']
        if not isinstance(strength, (int, float)) or strength < 0 or strength > 100:
            return False, "displacementStrength must be between 0 and 100"

    return True, None


@app.route('/', methods=['GET'])
def index():
    """API info endpoint."""
    return jsonify({
        'name': 'Mockup Displacement API',
        'version': '1.0.0',
        'endpoints': {
            'POST /generate-mockup': 'Generate a displacement mockup',
            'GET /output/<filename>': 'Retrieve generated mockup',
            'GET /health': 'Health check'
        },
        'documentation': {
            'baseImageUrl': 'URL to base t-shirt image',
            'designImageUrl': 'URL to design/logo image (PNG with transparency recommended)',
            'position': {'x': 'number', 'y': 'number'},
            'scale': 'number (default: 1.0)',
            'rotation': 'number in degrees (default: 0)',
            'displacementStrength': 'number 0-100 (default: 15)',
            'blendMode': '"normal" or "multiply" (default: "normal")'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })


@app.route('/generate-mockup', methods=['POST'])
def generate_mockup_endpoint():
    """
    Generate a mockup with displacement mapping.

    Request body (JSON):
    {
        "baseImageUrl": "https://...",
        "designImageUrl": "https://...",
        "position": {"x": 150, "y": 200},
        "scale": 0.8,
        "rotation": 0,
        "displacementStrength": 15,
        "blendMode": "normal"
    }

    Response:
    {
        "success": true,
        "mockupUrl": "https://..../output/abc123.jpg",
        "processingTime": 0.8
    }
    """
    start_time = time.time()

    try:
        # Parse JSON body
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body must be JSON'
            }), 400

        # Validate request
        is_valid, error_message = validate_request(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_message
            }), 400

        logger.info(f"Processing mockup request: position={data['position']}, "
                    f"scale={data.get('scale', 1.0)}, strength={data.get('displacementStrength', 15)}")

        # Download images
        base_img = download_image(data['baseImageUrl'])
        design_img = download_image(data['designImageUrl'])

        # Extract parameters with defaults
        position = data['position']
        scale = data.get('scale', 1.0)
        rotation = data.get('rotation', 0)
        displacement_strength = data.get('displacementStrength', 15)
        blend_mode = data.get('blendMode', 'normal')

        # Generate mockup
        logger.info("Generating mockup with displacement...")
        result = generate_mockup(
            base_img=base_img,
            design_img=design_img,
            position=position,
            scale=scale,
            rotation=rotation,
            displacement_strength=displacement_strength,
            blend_mode=blend_mode
        )

        # Generate unique output filename
        output_id = str(uuid.uuid4())[:12]
        output_filename = f'{output_id}.jpg'
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Save result with good quality
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logger.info(f"Saved output to: {output_path}")

        # Calculate processing time
        processing_time = round(time.time() - start_time, 2)

        # Build output URL
        # Use request.url_root for the base URL
        base_url = request.url_root.rstrip('/')
        mockup_url = f"{base_url}/output/{output_filename}"

        return jsonify({
            'success': True,
            'mockupUrl': mockup_url,
            'processingTime': processing_time,
            'outputId': output_id
        })

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Processing error: {str(e)}"
        }), 500


@app.route('/output/<filename>', methods=['GET'])
def serve_output(filename):
    """Serve generated mockup images."""
    filepath = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'error': 'File not found'
        }), 404

    return send_file(filepath, mimetype='image/jpeg')


@app.route('/generate-dispmap', methods=['POST'])
def generate_dispmap_endpoint():
    """
    Generate only the displacement map (for debugging/preview).

    Request body (JSON):
    {
        "imageUrl": "https://..."
    }

    Response: JPEG image of displacement map
    """
    try:
        data = request.get_json()
        if not data or 'imageUrl' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing imageUrl'
            }), 400

        # Download and process
        img = download_image(data['imageUrl'])
        dispmap = create_displacement_map(img)

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', dispmap)
        return send_file(
            BytesIO(buffer.tobytes()),
            mimetype='image/jpeg'
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Cleanup old files (optional, for production)
def cleanup_old_outputs(max_age_seconds=3600):
    """Remove output files older than max_age_seconds."""
    try:
        now = time.time()
        for filename in os.listdir(OUTPUT_DIR):
            filepath = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(filepath):
                age = now - os.path.getmtime(filepath)
                if age > max_age_seconds:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting Mockup API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

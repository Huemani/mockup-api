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

app = Flask(__name__)
CORS(app)

# Create output directory
os.makedirs('/tmp/outputs', exist_ok=True)


def download_image(url):
    """
    Download image from URL and convert to OpenCV format.
    
    Args:
        url: Image URL (Cloudinary or other)
        
    Returns:
        OpenCV image (BGR numpy array)
    """
    logger.info(f"Downloading image from: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Decode image
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Failed to decode image from {url}")
    
    logger.info(f"Image downloaded: {img.shape}")
    return img


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'mockup-displacement-api',
        'timestamp': time.time()
    })


@app.route('/create-displacement-map', methods=['POST'])
def create_displacement_map_endpoint():
    """
    Generate a displacement map from a base image.
    
    Request JSON:
        {
            "imageUrl": "https://..."
        }
    
    Returns:
        Base64-encoded grayscale displacement map
    """
    try:
        data = request.get_json()
        image_url = data.get('imageUrl')
        
        if not image_url:
            return jsonify({'success': False, 'error': 'imageUrl required'}), 400
        
        logger.info(f"Creating displacement map for: {image_url}")
        
        # Download image
        img = download_image(image_url)
        
        # Generate displacement map
        dispmap = create_displacement_map(img)
        
        # Save as PNG
        output_path = f"/tmp/outputs/dispmap_{uuid.uuid4().hex[:8]}.png"
        cv2.imwrite(output_path, dispmap)
        
        logger.info(f"Displacement map saved: {output_path}")
        
        return send_file(output_path, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error creating displacement map: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/generate-mockup', methods=['POST'])
def generate_mockup_endpoint():
    """
    Generate a mockup with displacement mapping.
    
    Request JSON:
        {
            "baseImageUrl": "https://...",
            "designImageUrl": "https://...",
            "position": {"x": 100, "y": 200},
            "scale": 0.22  // Frontend sends as decimal (0.22 = 22%)
        }
    
    Returns:
        Base64-encoded JPEG mockup image
    """
    try:
        data = request.get_json()
        
        # Get parameters
        base_url = data.get('baseImageUrl')
        design_url = data.get('designImageUrl')
        position = data.get('position', {'x': 0, 'y': 0})
        
        # CRITICAL FIX: Hardcoded displacement scale
        # Frontend sends incorrect scale values, so we use a fixed optimal value
        # This gives natural fabric displacement without distortion
        scale_decimal = data.get('scale', 0.22)  # Still log what frontend sends
        displacement_scale = 15  # HARDCODED: Optimal value for fabric texture (10-20% range is good)
        
        logger.info(f"Processing mockup request: position={position}, scale={scale_decimal}, displacement_scale={displacement_scale}")
        
        # Validate inputs
        if not base_url or not design_url:
            return jsonify({
                'success': False, 
                'error': 'baseImageUrl and designImageUrl required'
            }), 400
        
        # Download images
        logger.info(f"Downloading base image: {base_url}")
        base_img = download_image(base_url)
        logger.info(f"Base image shape: {base_img.shape}")
        
        logger.info(f"Downloading design image: {design_url}")
        design_img = download_image(design_url)
        logger.info(f"Design image shape: {design_img.shape}")
        
        # Scale base image if too large (performance optimization)
        base_h, base_w = base_img.shape[:2]
        max_dimension = 2000
        
        if max(base_h, base_w) > max_dimension:
            scale_factor = max_dimension / max(base_h, base_w)
            new_w = int(base_w * scale_factor)
            new_h = int(base_h * scale_factor)
            
            base_img = cv2.resize(
                base_img, 
                (new_w, new_h), 
                interpolation=cv2.INTER_AREA
            )
            logger.info(f"Scaled base image: {base_w}x{base_h} -> {new_w}x{new_h}")
            
            # Scale position coordinates proportionally
            position['x'] = int(position['x'] * scale_factor)
            position['y'] = int(position['y'] * scale_factor)
            
            # Scale design image proportionally
            design_h, design_w = design_img.shape[:2]
            new_design_w = int(design_w * scale_factor)
            new_design_h = int(design_h * scale_factor)
            
            design_img = cv2.resize(
                design_img,
                (new_design_w, new_design_h),
                interpolation=cv2.INTER_AREA
            )
            logger.info(f"Scaled design image: {design_w}x{design_h} -> {new_design_w}x{new_design_h}")
        
        # Get final dimensions
        base_h, base_w = base_img.shape[:2]
        design_h, design_w = design_img.shape[:2]
        
        # Log viewport positioning
        logger.info(f"Viewport offset on base: ({position['x']}, {position['y']})")
        logger.info(f"Design position on base: ({position['x']}, {position['y']}), scale: {scale_decimal}")
        
        # Generate mockup with displacement
        logger.info("Generating mockup with displacement...")
        result = generate_mockup(
            base_img,
            design_img,
            (position['x'], position['y']),
            scale=displacement_scale  # ✅ Photoshop percent (22, not 0.22)
        )
        
        # Save output as JPEG
        output_path = f"/tmp/outputs/{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Saved output to: {output_path}")
        
        # Return as base64-encoded data URL
        import base64
        with open(output_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_base64}"
        })
        
    except Exception as e:
        logger.error(f"Error generating mockup: {str(e)}", exc_info=True)
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500


@app.route('/test-displacement', methods=['POST'])
def test_displacement_endpoint():
    """
    Test endpoint for debugging displacement parameters.
    
    Returns displacement info without generating full mockup.
    """
    try:
        data = request.get_json()
        
        scale_decimal = data.get('scale', 0.22)
        displacement_scale = scale_decimal * 100
        displacement_scale = max(1, min(50, displacement_scale))
        
        return jsonify({
            'success': True,
            'input_scale': scale_decimal,
            'displacement_scale': displacement_scale,
            'formula': f"(gray_value - 128) * ({displacement_scale} / 100) * 2",
            'max_offset_pixels': displacement_scale * 2,
            'info': 'Photoshop-compatible displacement scale'
        })
        
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Run Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

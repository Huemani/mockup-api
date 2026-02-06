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
        
        # CRITICAL FIX: Scale conversion
        # Frontend sends scale as decimal: 0.22 = 22%
        # Backend needs Photoshop percent: 22 (not 0.22)
        scale_decimal = data.get('scale', 0.22)  # Default 22%
        displacement_scale = scale_decimal * 100  # Convert: 0.22 → 22
        
        # Clamp to reasonable Photoshop range
        # Typical: 5-15%, but allow 1-50% for flexibility
        displacement_scale = max(1, min(50, displacement_scale))
        
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
        app.run(host='0.0.0.0', port=port, debug=False)"""
Photoshop-style Displacement Mapping using OpenCV

Key insight: Displacement map is generated from BASE image and stays at BASE image size.
The design is displaced based on where it sits on the base image.
"""

import cv2
import numpy as np


def create_displacement_map(base_img):
    """
    Generate displacement map from base image.
    
    Process (mimics Photoshop):
    1. Convert to grayscale
    2. Apply Gaussian blur for smooth transitions
    3. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        base_img: BGR image (numpy array)
        
    Returns:
        Grayscale displacement map with enhanced contrast
    """
    # Convert to grayscale
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur (mimics Photoshop blur for smooth displacement)
    # Kernel size should be odd
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    return enhanced


def apply_displacement(design, dispmap_region, scale=10):
    """
    Apply Photoshop-style displacement using OpenCV remap.
    
    This uses bicubic interpolation for smooth, high-quality results –
    unlike PixiJS which uses 8-bit pixel shifting.
    
    IMPORTANT: Adds padding around design so edges can also warp!
    
    Args:
        design: Design image (BGR or BGRA with alpha)
        dispmap_region: Grayscale displacement map cropped to design region
                       (must be same size as design)
        scale: Displacement intensity in Photoshop percent (typically 5-15)
               Default 10 = 10% displacement
    
    Returns:
        Displaced design image (same size as input, edges will warp)
    """
    height, width = design.shape[:2]
    
    # Ensure dispmap matches design size
    if dispmap_region.shape[:2] != (height, width):
        dispmap_region = cv2.resize(dispmap_region, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Add padding around design so edges can also warp
    # Padding size should be at least as large as max displacement
    pad = max(int(scale * 2), 20)
    
    # ALWAYS convert to BGRA (with alpha) so padding can be transparent
    if design.shape[2] == 3:
        # BGR -> BGRA: Add full opacity alpha channel
        design = cv2.cvtColor(design, cv2.COLOR_BGR2BGRA)
    
    # Pad design with transparent pixels
    design_padded = cv2.copyMakeBorder(
        design,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0)  # Transparent
    )
    
    # Pad displacement map with edge replication (mirrors Photoshop "Repeat Edge Pixels")
    dispmap_padded = cv2.copyMakeBorder(
        dispmap_region,
        pad, pad, pad, pad,
        cv2.BORDER_REPLICATE
    )
    
    padded_height, padded_width = design_padded.shape[:2]
    
    # Create coordinate grids for padded size
    y_coords, x_coords = np.mgrid[0:padded_height, 0:padded_width].astype(np.float32)
    
    # Calculate displacement offsets using PHOTOSHOP FORMULA
    # Photoshop formula: offset = (gray_value - 128) * (scale / 100) * 2
    # 128 (50% gray) = no displacement
    # 0 (black) = -scale displacement
    # 255 (white) = +scale displacement
    offsets = (dispmap_padded.astype(np.float32) - 128.0) * (scale / 100.0) * 2
    
    # Apply offsets to both X and Y coordinates
    # This creates the "fabric fold" effect
    map_x = x_coords + offsets
    map_y = y_coords + offsets
    
    # Remap with BICUBIC interpolation (high quality, smooth results)
    # BORDER_CONSTANT with transparent pixels for clean edges
    displaced_padded = cv2.remap(
        design_padded,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)  # Transparent
    )
    
    # Return the padded result (edges now warp into the padding area)
    # We keep the padding so warped edges are visible
    return displaced_padded, pad


def apply_displacement_directional(design, dispmap_region, scale_x=10, scale_y=10):
    """
    Apply displacement with separate horizontal and vertical scale.
    
    This allows for anisotropic displacement (different strength in X vs Y).
    Useful for fabrics that stretch more in one direction.
    
    Args:
        design: Design image (BGR or BGRA)
        dispmap_region: Grayscale displacement map
        scale_x: Horizontal displacement scale (Photoshop percent)
        scale_y: Vertical displacement scale (Photoshop percent)
    
    Returns:
        Displaced design image with padding, padding amount
    """
    height, width = design.shape[:2]
    
    if dispmap_region.shape[:2] != (height, width):
        dispmap_region = cv2.resize(dispmap_region, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding based on max of both scales
    pad = max(int(max(scale_x, scale_y) * 2), 20)
    
    if design.shape[2] == 3:
        design = cv2.cvtColor(design, cv2.COLOR_BGR2BGRA)
    
    design_padded = cv2.copyMakeBorder(
        design,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0)
    )
    
    dispmap_padded = cv2.copyMakeBorder(
        dispmap_region,
        pad, pad, pad, pad,
        cv2.BORDER_REPLICATE
    )
    
    padded_height, padded_width = design_padded.shape[:2]
    y_coords, x_coords = np.mgrid[0:padded_height, 0:padded_width].astype(np.float32)
    
    dispmap_gray = dispmap_padded.astype(np.float32)
    
    # Apply Photoshop formula separately for X and Y
    offset_x = (dispmap_gray - 128.0) * (scale_x / 100.0) * 2
    offset_y = (dispmap_gray - 128.0) * (scale_y / 100.0) * 2
    
    map_x = x_coords + offset_x
    map_y = y_coords + offset_y
    
    displaced_padded = cv2.remap(
        design_padded,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    
    return displaced_padded, pad


def generate_mockup(base_img, design_img, position, scale=10):
    """
    Generate a complete mockup with displacement.
    
    Workflow:
    1. Create displacement map from base image (stays at base size)
    2. Calculate design region on base image
    3. Extract displacement map region for design area
    4. Apply displacement to design
    5. Composite displaced design back onto base
    
    Args:
        base_img: Base product image (BGR)
        design_img: Design to place (BGR or BGRA)
        position: (x, y) tuple for design placement on base
        scale: Displacement scale in Photoshop percent (5-15 typical)
    
    Returns:
        Final mockup image with displaced design
    """
    # Generate displacement map from base (Photoshop workflow)
    dispmap = create_displacement_map(base_img)
    
    x, y = position
    design_h, design_w = design_img.shape[:2]
    base_h, base_w = base_img.shape[:2]
    
    # Extract displacement map region for design area
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(base_w, x + design_w), min(base_h, y + design_h)
    
    dispmap_region = dispmap[y1:y2, x1:x2]
    
    # Crop design if it extends beyond base
    design_crop = design_img[
        max(0, -y):design_h - max(0, (y + design_h) - base_h),
        max(0, -x):design_w - max(0, (x + design_w) - base_w)
    ]
    
    # Apply displacement (returns padded result)
    displaced_design, pad = apply_displacement(design_crop, dispmap_region, scale)
    
    # Create result image
    result = base_img.copy()
    
    # Composite displaced design (with padding) onto base
    # Account for padding offset
    paste_x = max(0, x - pad)
    paste_y = max(0, y - pad)
    
    displaced_h, displaced_w = displaced_design.shape[:2]
    
    # Calculate paste region on base
    paste_x2 = min(base_w, paste_x + displaced_w)
    paste_y2 = min(base_h, paste_y + displaced_h)
    
    # Calculate corresponding region in displaced design
    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    src_x2 = src_x1 + (paste_x2 - paste_x)
    src_y2 = src_y1 + (paste_y2 - paste_y)
    
    # Extract region to composite
    to_composite = displaced_design[src_y1:src_y2, src_x1:src_x2]
    
    if to_composite.shape[2] == 4:  # Has alpha
        # Alpha composite
        alpha = to_composite[:, :, 3:4] / 255.0
        result[paste_y:paste_y2, paste_x:paste_x2] = (
            to_composite[:, :, :3] * alpha +
            result[paste_y:paste_y2, paste_x:paste_x2] * (1 - alpha)
        ).astype(np.uint8)
    else:
        # Direct paste (no alpha)
        result[paste_y:paste_y2, paste_x:paste_x2] = to_composite
    
    return result

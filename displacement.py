"""
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
    
    # Apply LARGE Gaussian blur for smooth fabric-like displacement
    # Photoshop typically uses blur radius 20-50 pixels for fabric texture
    # Kernel size must be odd: 51x51 with sigma=20 gives smooth, natural results
    blurred = cv2.GaussianBlur(gray, (51, 51), 20)
    
    # Simple contrast enhancement (no CLAHE - it creates patchy artifacts)
    # Alpha=1.5 increases contrast, beta=-50 darkens slightly for better range
    enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=-50)
    
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
    
    # Apply offsets ONLY HORIZONTALLY (standard for fabric texture)
    # Vertical displacement creates unrealistic "melting" effect
    # Horizontal displacement follows natural fabric wrinkles
    map_x = x_coords + offsets  # Horizontal displacement
    map_y = y_coords            # NO vertical displacement
    
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

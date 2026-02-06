"""
Photoshop-style Displacement Mapping using OpenCV

CORRECT WORKFLOW:
1. Displacement map = 1:1 representation of base image texture
2. Design placed at position (x, y) extracts dispmap region [y:y+h, x:x+w]
3. Design warps based on that exact region of base texture
4. Displaced design pastes back at same position
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_displacement_map(base_img):
    """
    Generate displacement map from base image.
    
    OUTPUT: Same size as base image (1:1 texture mapping)
    
    Args:
        base_img: BGR image
        
    Returns:
        Grayscale displacement map (same size as base)
    """
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    
    # Large Gaussian blur for smooth fabric texture
    blurred = cv2.GaussianBlur(gray, (51, 51), 20)
    
    # Simple contrast enhancement
    enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=-50)
    
    logger.info(f"Created displacement map: {enhanced.shape[:2]} (matches base)")
    return enhanced


def apply_displacement(design, dispmap_region, scale=10):
    """
    Warp design using displacement map region.
    
    CRITICAL: design and dispmap_region MUST be same size (no resize!)
    
    Args:
        design: Design image (BGR or BGRA)
        dispmap_region: Grayscale displacement map (SAME SIZE as design)
        scale: Displacement intensity (Photoshop percent, 5-15 typical)
    
    Returns:
        Displaced design (same size as input)
    """
    height, width = design.shape[:2]
    
    # Verify sizes match (critical!)
    if dispmap_region.shape[:2] != (height, width):
        raise ValueError(
            f"SIZE MISMATCH! Design: {(height, width)}, "
            f"Dispmap region: {dispmap_region.shape[:2]}. "
            f"This means region extraction is broken!"
        )
    
    # Convert to BGRA for proper alpha compositing
    if len(design.shape) == 2 or design.shape[2] == 3:
        if len(design.shape) == 2:
            design = cv2.cvtColor(design, cv2.COLOR_GRAY2BGRA)
        else:
            design = cv2.cvtColor(design, cv2.COLOR_BGR2BGRA)
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
    
    # Photoshop displacement formula
    offsets = (dispmap_region.astype(np.float32) - 128.0) * (scale / 100.0) * 2
    
    # Horizontal displacement only (natural for fabric)
    map_x = x_coords + offsets
    map_y = y_coords
    
    # Warp design
    displaced = cv2.remap(
        design,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101  # Better edge handling
    )
    
    logger.info(f"Displaced design: {displaced.shape[:2]}, scale={scale}%")
    return displaced


def generate_mockup(base_img, design_img, position, scale=10):
    """
    Generate mockup with correct displacement workflow.
    
    WORKFLOW:
    1. Create displacement map (1:1 with base texture)
    2. Extract region from dispmap based on design position
    3. Warp design using that region
    4. Paste displaced design back at same position
    
    Args:
        base_img: Base product image (BGR)
        design_img: Design to place (BGR or BGRA)
        position: (x, y) placement on base
        scale: Displacement intensity (5-15 typical)
    
    Returns:
        Final mockup image
    """
    x, y = position
    base_h, base_w = base_img.shape[:2]
    design_h, design_w = design_img.shape[:2]
    
    logger.info(f"=== MOCKUP GENERATION ===")
    logger.info(f"Base: {base_w}x{base_h}")
    logger.info(f"Design: {design_w}x{design_h}")
    logger.info(f"Position: ({x}, {y})")
    
    # Step 1: Create displacement map (1:1 with base)
    dispmap = create_displacement_map(base_img)
    
    # Step 2: Calculate visible region (design might extend beyond base)
    # Design region on base
    x1_base = max(0, x)
    y1_base = max(0, y)
    x2_base = min(base_w, x + design_w)
    y2_base = min(base_h, y + design_h)
    
    # Corresponding region in design
    x1_design = max(0, -x)
    y1_design = max(0, -y)
    x2_design = x1_design + (x2_base - x1_base)
    y2_design = y1_design + (y2_base - y1_base)
    
    # Extract regions
    dispmap_region = dispmap[y1_base:y2_base, x1_base:x2_base]
    design_crop = design_img[y1_design:y2_design, x1_design:x2_design]
    
    logger.info(f"Visible region: dispmap={dispmap_region.shape[:2]}, design={design_crop.shape[:2]}")
    
    # Verify sizes match
    if dispmap_region.shape[:2] != design_crop.shape[:2]:
        raise ValueError(
            f"REGION EXTRACTION FAILED! "
            f"Dispmap region: {dispmap_region.shape[:2]}, "
            f"Design crop: {design_crop.shape[:2]}"
        )
    
    # Step 3: Warp design using displacement map region
    displaced_design = apply_displacement(design_crop, dispmap_region, scale)
    
    # Step 4: Paste displaced design onto base
    result = base_img.copy()
    
    # Alpha composite
    if displaced_design.shape[2] == 4:
        alpha = displaced_design[:, :, 3:4] / 255.0
        rgb = displaced_design[:, :, :3]
        base_region = result[y1_base:y2_base, x1_base:x2_base]
        
        blended = (rgb * alpha + base_region * (1 - alpha)).astype(np.uint8)
        result[y1_base:y2_base, x1_base:x2_base] = blended
        logger.info(f"Alpha composited at ({x1_base}, {y1_base})")
    else:
        result[y1_base:y2_base, x1_base:x2_base] = displaced_design[:, :, :3]
        logger.info(f"Direct paste at ({x1_base}, {y1_base})")
    
    logger.info("=== MOCKUP COMPLETE ===")
    return result

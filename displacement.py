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
        Grayscale displacement map (same size as base_img)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur - critical for smooth displacement without jagged edges
    # Kernel size 5x5 gives good balance between detail and smoothness
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # CLAHE for contrast enhancement
    # This brings out fabric wrinkles and folds
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    return enhanced


def apply_displacement(design, dispmap_region, strength=15):
    """
    Apply Photoshop-style displacement using OpenCV remap.

    This uses bicubic interpolation for smooth, high-quality results -
    unlike PixiJS which uses 8-bit pixel shifting.

    IMPORTANT: Adds padding around design so edges can also warp!

    Args:
        design: Design image (BGR or BGRA with alpha)
        dispmap_region: Grayscale displacement map cropped to design region
                       (must be same size as design)
        strength: Displacement intensity in pixels (typically 10-20)

    Returns:
        Displaced design image (same size as input, edges will warp)
    """
    height, width = design.shape[:2]

    # Ensure dispmap matches design size
    if dispmap_region.shape[:2] != (height, width):
        dispmap_region = cv2.resize(dispmap_region, (width, height), interpolation=cv2.INTER_LINEAR)

    # Add padding around design so edges can warp
    # Padding size should be at least as large as max displacement
    pad = int(strength * 1.5)

    # Pad the design with transparent pixels (or black if no alpha)
    if design.shape[2] == 4:
        # BGRA - pad with transparent
        design_padded = cv2.copyMakeBorder(
            design, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0, 0)
        )
    else:
        # BGR - pad with black
        design_padded = cv2.copyMakeBorder(
            design, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    # Pad the displacement map with neutral gray (128 = no displacement)
    dispmap_padded = cv2.copyMakeBorder(
        dispmap_region, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=128
    )

    # Get padded dimensions
    padded_height, padded_width = design_padded.shape[:2]

    # Create coordinate grids for padded size
    y_coords, x_coords = np.mgrid[0:padded_height, 0:padded_width].astype(np.float32)

    # Calculate displacement offsets
    # 128 (50% gray) = no displacement
    # 0 (black) = -strength displacement
    # 255 (white) = +strength displacement
    offsets = (dispmap_padded.astype(np.float32) - 128.0) / 128.0 * strength

    # Apply offsets to both X and Y coordinates
    # This creates the "fabric fold" effect
    map_x = x_coords + offsets
    map_y = y_coords + offsets

    # Remap with BICUBIC interpolation (high quality, smooth results)
    # BORDER_CONSTANT with transparent/black for clean edges
    if design.shape[2] == 4:
        border_value = (0, 0, 0, 0)
    else:
        border_value = (0, 0, 0)

    displaced_padded = cv2.remap(
        design_padded,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    # Return the padded result (edges now warp into the padding area)
    # We keep the padding so warped edges are visible
    return displaced_padded, pad


def apply_displacement_directional(design, dispmap_region, strength_x=15, strength_y=15):
    """
    Apply displacement with separate X and Y control.

    Useful if you want more horizontal or vertical displacement.

    Args:
        design: Design image
        dispmap_region: Grayscale displacement map (same size as design)
        strength_x: Horizontal displacement strength
        strength_y: Vertical displacement strength

    Returns:
        Displaced design image
    """
    height, width = design.shape[:2]

    if dispmap_region.shape[:2] != (height, width):
        dispmap_region = cv2.resize(dispmap_region, (width, height), interpolation=cv2.INTER_LINEAR)

    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)

    # Normalize dispmap to -1 to +1 range
    normalized = (dispmap_region.astype(np.float32) - 128.0) / 128.0

    # Apply separate X and Y strengths
    map_x = x_coords + normalized * strength_x
    map_y = y_coords + normalized * strength_y

    displaced = cv2.remap(
        design,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )

    return displaced


def process_design(design, scale=1.0, rotation=0):
    """
    Process design image: resize and rotate.

    Args:
        design: Input design image (BGR or BGRA)
        scale: Scale factor (1.0 = original size)
        rotation: Rotation in degrees (clockwise)

    Returns:
        Processed design image
    """
    result = design.copy()

    # Resize if scale != 1.0
    if scale != 1.0:
        new_width = int(design.shape[1] * scale)
        new_height = int(design.shape[0] * scale)
        result = cv2.resize(result, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Rotate if needed
    if rotation != 0:
        height, width = result.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, -rotation, 1.0)  # Negative for clockwise

        # Calculate new bounding box size
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)

        # Adjust rotation matrix for new center
        matrix[0, 2] += (new_width - width) / 2
        matrix[1, 2] += (new_height - height) / 2

        # Apply rotation with transparent background for BGRA images
        if result.shape[2] == 4:
            result = cv2.warpAffine(result, matrix, (new_width, new_height),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))
        else:
            result = cv2.warpAffine(result, matrix, (new_width, new_height))

    return result


def composite_on_base(base, design, position, blend_mode='normal'):
    """
    Composite design onto base image at specified position.

    Handles alpha channel for transparency.

    Args:
        base: Base image (BGR)
        design: Design image (BGR or BGRA)
        position: Tuple (x, y) for top-left corner
        blend_mode: 'normal' or 'multiply' (for fabric texture effect)

    Returns:
        Composited result image
    """
    x, y = int(position[0]), int(position[1])
    dh, dw = design.shape[:2]
    bh, bw = base.shape[:2]

    # Create output (copy of base)
    result = base.copy()

    # Calculate the region where design overlaps with base
    # Handle cases where design extends beyond base edges
    src_x_start = max(0, -x)
    src_y_start = max(0, -y)
    src_x_end = min(dw, bw - x)
    src_y_end = min(dh, bh - y)

    dst_x_start = max(0, x)
    dst_y_start = max(0, y)
    dst_x_end = min(bw, x + dw)
    dst_y_end = min(bh, y + dh)

    # Check if there's any overlap
    if src_x_end <= src_x_start or src_y_end <= src_y_start:
        return result  # No overlap, return base unchanged

    # Extract regions
    design_region = design[src_y_start:src_y_end, src_x_start:src_x_end]
    base_region = result[dst_y_start:dst_y_end, dst_x_start:dst_x_end]

    if design.shape[2] == 4:
        # Has alpha channel
        alpha = design_region[:, :, 3:4].astype(np.float32) / 255.0
        design_rgb = design_region[:, :, :3].astype(np.float32)
        base_rgb = base_region.astype(np.float32)

        if blend_mode == 'multiply':
            # Multiply blend for fabric texture
            blended = (design_rgb / 255.0) * base_rgb
            composited = alpha * blended + (1 - alpha) * base_rgb
        else:
            # Normal blend
            composited = alpha * design_rgb + (1 - alpha) * base_rgb

        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = composited.astype(np.uint8)
    else:
        # No alpha channel
        if blend_mode == 'multiply':
            blended = (design_region.astype(np.float32) / 255.0) * base_region.astype(np.float32)
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = blended.astype(np.uint8)
        else:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = design_region

    return result


def generate_mockup(base_img, design_img, position, scale=1.0, rotation=0,
                    displacement_strength=15, blend_mode='normal'):
    """
    Complete mockup generation pipeline.

    CRITICAL: Displacement map is generated from BASE image and stays at base size.
    Only the region where the design sits is used for displacement.

    Args:
        base_img: Base t-shirt image (BGR)
        design_img: Design/logo image (BGR or BGRA)
        position: Dict with 'x' and 'y' keys
        scale: Design scale factor
        rotation: Design rotation in degrees
        displacement_strength: How much to displace (pixels)
        blend_mode: 'normal' or 'multiply'

    Returns:
        Final composited mockup image
    """
    # Step 1: Generate displacement map from FULL base image
    dispmap_full = create_displacement_map(base_img)

    # Step 2: Process design (scale, rotate)
    design_processed = process_design(design_img, scale=scale, rotation=rotation)

    # Step 3: Get position
    x, y = int(position.get('x', 0)), int(position.get('y', 0))
    dh, dw = design_processed.shape[:2]
    bh, bw = base_img.shape[:2]

    # Step 4: Extract the displacement map region where design will be placed
    # This is the KEY fix - dispmap region matches design position on base
    disp_x_start = max(0, x)
    disp_y_start = max(0, y)
    disp_x_end = min(bw, x + dw)
    disp_y_end = min(bh, y + dh)

    # Crop displacement map to design region
    dispmap_region = dispmap_full[disp_y_start:disp_y_end, disp_x_start:disp_x_end]

    # Handle edge cases where design extends beyond base
    design_crop_x = max(0, -x)
    design_crop_y = max(0, -y)
    design_cropped = design_processed[
        design_crop_y:design_crop_y + dispmap_region.shape[0],
        design_crop_x:design_crop_x + dispmap_region.shape[1]
    ]

    # Step 5: Apply displacement to design using the correct region of dispmap
    if dispmap_region.size > 0 and design_cropped.size > 0:
        design_displaced, pad = apply_displacement(
            design_cropped,
            dispmap_region,
            strength=displacement_strength
        )

        # Adjust position to account for padding
        # The displaced image is larger due to padding, so we offset the position
        adjusted_x = x - pad + design_crop_x
        adjusted_y = y - pad + design_crop_y

        # Use the displaced (padded) design directly
        design_final = design_displaced
    else:
        design_final = design_processed
        adjusted_x = x
        adjusted_y = y
        pad = 0

    # Step 6: Composite displaced design onto base
    result = composite_on_base(base_img, design_final, (adjusted_x, adjusted_y), blend_mode=blend_mode)

    return result

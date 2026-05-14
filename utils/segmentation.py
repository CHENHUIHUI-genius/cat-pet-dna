"""
Cat Pet DNA — Stage 1: Foreground Segmentation
Low-compute: Canny edge detection + GrabCut (OpenCV built-in, no model required)
"""

import cv2
import numpy as np


def segment_foreground(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Extract cat foreground mask using edge detection + GrabCut.

    Args:
        image: RGB image (H, W, 3), any size

    Returns:
        mask: binary uint8 mask (H, W), 255=foreground, 0=background
        quality: float 0~1 segmentation quality estimate
    """
    h, w = image.shape[:2]
    # Resize to 512 for consistent processing
    scale = 512.0 / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    img_small = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # --- Step 1: Canny edge detection ---
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # --- Step 2: Morphological close to fill gaps ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Step 3: Find largest connected component as initial foreground ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: use whole image center as initial
        fg_mask = np.zeros(img_small.shape[:2], dtype=np.uint8)
        cx, cy = new_size[0] // 2, new_size[1] // 2
        cv2.rectangle(fg_mask, (cx - new_size[0]//4, cy - new_size[1]//4),
                      (cx + new_size[0]//4, cy + new_size[1]//4), 255, -1)
        init_mask = fg_mask
    else:
        largest = max(contours, key=cv2.contourArea)
        init_mask = np.zeros(img_small.shape[:2], dtype=np.uint8)
        cv2.drawContours(init_mask, [largest], -1, 255, -1)

    # --- Step 4: Prepare GrabCut mask ---
    # GrabCut uses: 0=BG, 1=FG, 2=pr_BG, 3=pr_FG
    gc_mask = np.zeros(img_small.shape[:2], dtype=np.uint8)
    gc_mask[init_mask == 0] = cv2.GC_BGD
    gc_mask[init_mask == 255] = cv2.GC_PR_FGD

    # --- Step 5: Run GrabCut (1 iteration for speed) ---
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_small, gc_mask, None, bgd_model, fgd_model,
                    iterCount=1, mode=cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # Fallback: use init_mask directly
        final_mask_small = init_mask
        quality = 0.3
    else:
        final_mask_small = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                                    255, 0).astype(np.uint8)

    # --- Step 6: Resize back to original size ---
    if new_size != (w, h):
        final_mask = cv2.resize(final_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        final_mask = final_mask_small

    # --- Quality assessment ---
    fg_pixels = np.count_nonzero(final_mask)
    total_pixels = h * w
    fg_ratio = fg_pixels / total_pixels

    # Reasonable cat: should occupy 5%~70% of frame
    ratio_score = 1.0 if 0.05 < fg_ratio < 0.7 else max(0, 1.0 - abs(fg_ratio - 0.3) * 2)

    # Edge continuity: check mask boundary smoothness
    mask_edges = cv2.Canny(final_mask, 0, 1)
    edge_length = np.count_nonzero(mask_edges)
    # For a smooth shape, boundary / sqrt(area) should be moderate
    smoothness = min(1.0, (2 * np.pi * np.sqrt(fg_pixels / np.pi)) / max(edge_length, 1))
    smoothness = max(0, min(1, smoothness))

    quality = 0.6 * ratio_score + 0.4 * smoothness
    quality = max(0, min(1, quality))

    return final_mask, quality


def extract_head_roi(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract head region from mask. Head = top 30%~40% of the contour bounding rect.

    Args:
        image: RGB image
        mask: binary mask

    Returns:
        head_mask: binary mask of head region
        head_img: cropped RGB head region (or None if too small)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask), np.zeros((10, 10, 3), dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Head = top 35% of bounding box (approximate)
    head_y_end = y + int(h * 0.35)
    head_region = mask[y:head_y_end, x:x + w]

    # Find head contour within this region
    head_mask = np.zeros_like(mask)
    head_mask[y:head_y_end, x:x + w] = head_region

    # Crop head image
    head_img = image[max(0, y):min(image.shape[0], head_y_end),
                     max(0, x):min(image.shape[1], x + w)]

    if head_img.size == 0:
        head_img = np.zeros((10, 10, 3), dtype=np.uint8)

    return head_mask, head_img
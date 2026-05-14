"""
Cat Pet DNA — Stage 4: Head & Face Feature Analysis
- Head shape from contour (visual algorithm + rule)
- Ear detection via convexity defects (visual algorithm)
- Eye ROI via adaptive thresholding (visual algorithm)
- Face symmetry (visual algorithm)
"""

import cv2
import numpy as np
import math


def analyze_head_shape(head_mask: np.ndarray) -> dict:
    """
    Analyze head shape from head region mask.

    Returns dict with: shape, circularity, aspect_ratio, confidence
    """
    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"shape": "unknown", "circularity": 0, "aspect_ratio": 0, "confidence": 0}

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, True))
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    circularity = min(1.0, circularity)

    _, (w, h), _ = cv2.minAreaRect(largest)
    aspect_ratio = (h / max(w, 1)) if w > 0 else 0

    # Rule-based shape classification
    if circularity > 0.75 and 0.8 <= aspect_ratio <= 1.2:
        shape = "round"
        confidence = min(1.0, circularity)
    elif aspect_ratio > 1.3:
        shape = "wedge"
        confidence = min(1.0, aspect_ratio / 2.0)
    elif circularity < 0.6 and aspect_ratio < 0.9:
        shape = "square"
        confidence = 0.6
    else:
        shape = "triangular"
        confidence = 0.5

    return {
        "shape": shape,
        "circularity": round(circularity, 3),
        "aspect_ratio": round(aspect_ratio, 3),
        "confidence": round(confidence, 3),
    }


def detect_ears(head_mask: np.ndarray) -> dict:
    """
    Detect ear tips from head contour convexity defects.

    Returns dict with: tips_detected, ear_type, ear_spread_ratio
    """
    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"tips_detected": 0, "ear_type": "unknown", "ear_spread_ratio": 0}

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest, returnPoints=False)

    if hull.size < 3:
        return {"tips_detected": 0, "ear_type": "unknown", "ear_spread_ratio": 0}

    defects = cv2.convexityDefects(largest, hull)
    if defects is None:
        return {"tips_detected": 0, "ear_type": "unknown", "ear_spread_ratio": 0}

    # Find topmost convex hull points (potential ear tips)
    hull_points = cv2.convexHull(largest)
    # hull_points shape: (N, 1, 2) → reshape to (N, 2)
    hull_points = hull_points.reshape(-1, 2)
    # Sort by y-coordinate (top = smallest y)
    sorted_pts = hull_points[hull_points[:, 1].argsort()]
    top_pts = sorted_pts[:min(4, len(sorted_pts))]

    if len(top_pts) < 2:
        return {"tips_detected": 0, "ear_type": "unknown", "ear_spread_ratio": 0}

    # Take two topmost distinct points as ear tips
    ear_tips = []
    for pt in top_pts:
        x, y = int(pt[0]), int(pt[1])
        # Check if this is a distinct point (not too close to another)
        is_distinct = True
        for ex, ey in ear_tips:
            if abs(x - ex) < 10 and abs(y - ey) < 10:
                is_distinct = False
                break
        if is_distinct:
            ear_tips.append((x, y))
        if len(ear_tips) == 2:
            break

    if len(ear_tips) < 2:
        return {"tips_detected": len(ear_tips), "ear_type": "unknown", "ear_spread_ratio": 0}

    # Calculate ear spread ratio
    ear_dist = math.sqrt((ear_tips[0][0] - ear_tips[1][0]) ** 2 +
                         (ear_tips[0][1] - ear_tips[1][1]) ** 2)
    head_width = float(cv2.boundingRect(largest)[2])
    spread_ratio = ear_dist / max(head_width, 1)

    # Ear type rule
    if spread_ratio > 0.6:
        ear_type = "pointed"
    elif spread_ratio > 0.4:
        ear_type = "rounded"
    else:
        ear_type = "folded"

    return {
        "tips_detected": 2,
        "ear_type": ear_type,
        "ear_spread_ratio": round(spread_ratio, 3),
        "ear_tips": ear_tips,
    }


def detect_eyes(head_img: np.ndarray) -> dict:
    """
    Detect potential eye regions using adaptive thresholding + morphology.

    Returns dict with: eyes_detected, eye_regions
    """
    if head_img.size < 100:
        return {"eyes_detected": False, "eye_count": 0}

    gray = cv2.cvtColor(head_img, cv2.COLOR_RGB2GRAY)
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive threshold to find dark regions (eyes are typically dark)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find potential eye contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eye_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        # Eyes: small, roughly oval, in upper half of head
        if 20 < area < 500 and 0.5 < w / max(h, 1) < 2.0 and y < head_img.shape[0] * 0.7:
            eye_candidates.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "area": float(area),
            })

    # Sort by area, take top 2 as most likely eyes
    eye_candidates.sort(key=lambda e: -e["area"])
    eyes = eye_candidates[:2]

    return {
        "eyes_detected": len(eyes) >= 1,
        "eye_count": len(eyes),
        "eye_regions": eyes,
    }


def compute_face_symmetry(head_mask: np.ndarray) -> float:
    """
    Compute face symmetry by comparing left/right halves of head mask.
    Returns 0~1 score.
    """
    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 10:
        return 0.0

    # Extract head region
    head_region = head_mask[y:y + h, x:x + w]
    if head_region.shape[1] < 4:
        return 0.0

    # Split into left and right halves
    mid = head_region.shape[1] // 2
    left = head_region[:, :mid]
    right = head_region[:, mid:2 * mid]  # same width as left

    if left.shape[1] != right.shape[1]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))

    # Flip right half
    right_flipped = cv2.flip(right, 1)

    # Compute overlap ratio
    overlap = cv2.bitwise_and(left, right_flipped)
    union = cv2.bitwise_or(left, right_flipped)
    overlap_px = np.count_nonzero(overlap)
    union_px = np.count_nonzero(union)
    symmetry = overlap_px / max(union_px, 1)

    return round(symmetry, 3)
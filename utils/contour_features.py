"""
Cat Pet DNA — Stage 3: Contour Geometry Features
All computed via OpenCV contour functions — pure visual algorithm, no learning.
"""

import cv2
import numpy as np
import math


def extract_body_geometry(mask: np.ndarray) -> dict:
    """
    Extract geometric features from the largest contour in the mask.

    Args:
        mask: binary uint8 mask (H, W)

    Returns:
        dict with: area_px, aspect_ratio, circularity, solidity, extent,
                   body_angle, perimeter, bounding_rect
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area_px": 0, "aspect_ratio": 0, "circularity": 0,
            "solidity": 0, "extent": 0, "body_angle": 0,
            "perimeter": 0, "bounding_rect": [0, 0, 0, 0],
        }

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, True))

    # Minimum area rectangle
    rect = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect
    if w < h:
        aspect_ratio = h / max(w, 1)
        body_angle = angle
    else:
        aspect_ratio = w / max(h, 1)
        body_angle = angle + 90

    # Circularity: 4πA / P² (1.0 = perfect circle)
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    circularity = min(1.0, circularity)

    # Solidity: area / convex hull area
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0
    solidity = min(1.0, solidity)

    # Extent: area / bounding rect area
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    extent = min(1.0, extent)

    # Bounding rect
    bx, by, bw, bh = cv2.boundingRect(largest)

    return {
        "area_px": round(area, 1),
        "aspect_ratio": round(aspect_ratio, 3),
        "circularity": round(circularity, 3),
        "solidity": round(solidity, 3),
        "extent": round(extent, 3),
        "body_angle": round(body_angle, 1),
        "perimeter": round(perimeter, 1),
        "bounding_rect": [int(bx), int(by), int(bw), int(bh)],
    }


def estimate_body_size(area_px: float, image_area: float) -> str:
    """Rule-based body size estimate from normalized area."""
    ratio = area_px / max(image_area, 1)
    if ratio > 0.4:
        return "large"
    elif ratio > 0.15:
        return "medium"
    elif ratio > 0.02:
        return "small"
    return "unknown"
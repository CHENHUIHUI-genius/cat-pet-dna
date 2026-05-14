"""
Cat Pet DNA — Visualization Overlay Tool
Draws segmentation contour, bounding rect, ear tips, color swatches, and DNA text on image.
"""

import cv2
import numpy as np


def draw_dna_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    dna: dict,
    ear_tips: list = None,
) -> np.ndarray:
    """
    Draw visual DNA overlay on image.

    Args:
        image: original RGB image
        mask: binary foreground mask
        dna: Pet DNA dict
        ear_tips: list of (x, y) ear tip coordinates

    Returns:
        annotated RGB image
    """
    overlay = image.copy()
    h, w = image.shape[:2]

    # --- 1. Green contour ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [largest], -1, (0, 255, 0), 2)

    # --- 2. Blue bounding rect ---
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
        # Aspect ratio text
        ar = dna.get("appearance", {}).get("body_geometry", {}).get("aspect_ratio", 0)
        cv2.putText(overlay, f"AR={ar:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # --- 3. Red ear tips ---
    if ear_tips:
        for pt in ear_tips:
            cv2.circle(overlay, pt, 5, (0, 0, 255), -1)
            cv2.circle(overlay, pt, 7, (0, 0, 255), 2)

    # --- 4. Color swatch bar (top-right) ---
    palette = dna.get("appearance", {}).get("color_palette", [])
    if palette:
        swatch_h = 20
        swatch_w = max(40, w // (len(palette) + 1))
        total_swatch_w = len(palette) * swatch_w
        swatch_x = w - total_swatch_w - 10
        swatch_y = 10
        for i, color in enumerate(palette):
            rgb = color.get("rgb", [128, 128, 128])
            color_bgr = (rgb[2], rgb[1], rgb[0])  # RGB → BGR for OpenCV
            cv2.rectangle(overlay,
                          (swatch_x + i * swatch_w, swatch_y),
                          (swatch_x + (i + 1) * swatch_w, swatch_y + swatch_h),
                          color_bgr, -1)
            cv2.rectangle(overlay,
                          (swatch_x + i * swatch_w, swatch_y),
                          (swatch_x + (i + 1) * swatch_w, swatch_y + swatch_h),
                          (255, 255, 255), 1)
            # Ratio text
            ratio = color.get("ratio", 0)
            cv2.putText(overlay, f"{ratio:.0%}",
                        (swatch_x + i * swatch_w + 2, swatch_y + swatch_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # --- 5. DNA info text panel (bottom-left) ---
    panel_x, panel_y = 10, h - 120
    line_h = 18

    # Semi-transparent background
    panel_w = 320
    panel_h = 115
    sub_img = overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    alpha = 0.6
    overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = \
        cv2.addWeighted(sub_img, 1 - alpha, black_rect, alpha, 0)

    # Text lines
    pattern = dna.get("appearance", {}).get("color_pattern", {}).get("type", "?")
    colors = ", ".join(dna.get("appearance", {}).get("primary_colors", ["?"]))
    pose = dna.get("pose", {}).get("pose_type", "?")
    action = dna.get("action_tendency", {}).get("state", "?")
    conf = dna.get("confidence", {}).get("overall", 0)
    body_size = dna.get("appearance", {}).get("body_size_estimate", "?")
    head_shape = dna.get("face", {}).get("head_shape", {}).get("shape", "?")
    breed = dna.get("breed", {}).get("candidates", [{"name": "?"}])[0]["name"]

    texts = [
        f"Pattern: {pattern}  |  {colors}",
        f"Pose: {pose} ({conf:.0%})  |  Size: {body_size}",
        f"Head: {head_shape}  |  Action: {action}",
        f"Breed: {breed}",
    ]
    for i, text in enumerate(texts):
        cv2.putText(overlay, text, (panel_x + 5, panel_y + 15 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # --- 6. Title ---
    cv2.putText(overlay, "Cat Pet DNA v1.0", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return overlay
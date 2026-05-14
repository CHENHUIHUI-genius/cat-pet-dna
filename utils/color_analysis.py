"""
Cat Pet DNA — Stage 2: Color Feature Extraction
- K-Means clustering for main colors (visual algorithm)
- Color pattern / coat type rule inference (rule-based)
"""

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

# Color name mapping (rule-based lookup)
RGB_COLOR_TABLE = {
    "white":    ([200, 200, 200], [255, 255, 255]),
    "black":    ([0, 0, 0],       [60, 60, 60]),
    "gray":     ([60, 60, 60],    [180, 180, 180]),
    "orange":   ([180, 100, 0],   [255, 200, 80]),
    "brown":    ([80, 40, 20],    [160, 100, 60]),
    "cream":    ([220, 200, 160], [255, 240, 210]),
    "blue":     ([100, 120, 150], [150, 160, 200]),
}


def _rgb_to_name(r: int, g: int, b: int) -> str:
    """Map RGB to human-readable color name using distance rule."""
    min_dist = float("inf")
    best_name = "unknown"
    for name, (low, high) in RGB_COLOR_TABLE.items():
        if all(low[i] <= c <= high[i] for i, c in enumerate((r, g, b))):
            return name
        center = [(low[i] + high[i]) / 2 for i in range(3)]
        dist = sum((c - center[i]) ** 2 for i, c in enumerate((r, g, b)))
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _compute_color_entropy(image: np.ndarray, mask: np.ndarray) -> float:
    """HSV hue entropy = color diversity indicator."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    fg_hue = hue[mask > 0]
    if len(fg_hue) < 10:
        return 0.0
    hist = np.histogram(fg_hue, bins=36, range=(0, 180))[0].astype(float)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def _infer_pattern_type(palette: list, entropy: float, mask: np.ndarray,
                        image: np.ndarray) -> str:
    """Rule-based coat pattern inference from palette + statistics."""
    n_colors = len(palette)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    fg_pixels = image[mask > 0]
    if len(fg_pixels) == 0:
        return "unknown"

    fg_hsv = hsv[mask > 0]
    white_mask = (fg_hsv[:, 2] > 200) & (fg_hsv[:, 1] < 40)
    white_ratio = float(np.mean(white_mask)) if len(white_mask) > 0 else 0.0

    black_mask = (fg_pixels[:, 0] < 60) & (fg_pixels[:, 1] < 60) & (fg_pixels[:, 2] < 60)
    black_ratio = float(np.mean(black_mask)) if len(black_mask) > 0 else 0.0

    if n_colors <= 2 and entropy < 2.0:
        return "solid"
    if white_ratio > 0.20 and n_colors >= 3:
        return "calico"
    if black_ratio + white_ratio > 0.70 and n_colors <= 3:
        return "bicolor"
    if 2.0 <= entropy <= 4.0 and n_colors >= 2:
        return "tabby"
    if n_colors >= 3 and entropy > 3.0:
        return "tortie"
    return "unknown"


def extract_color_palette(image: np.ndarray, mask: np.ndarray,
                          n_colors: int = 5) -> dict:
    """
    Extract dominant colors using MiniBatch K-Means.
    Returns dict with palette, entropy, primary_names, pattern_type, confidence.
    """
    fg_pixels = image[mask > 0]
    if len(fg_pixels) < 10:
        return {
            "palette": [],
            "entropy": 0.0,
            "primary_names": ["unknown"],
            "pattern_type": "unknown",
            "confidence": 0.0,
        }

    # Downsample for speed
    if len(fg_pixels) > 50000:
        idx = np.random.choice(len(fg_pixels), 50000, replace=False)
        fg_pixels = fg_pixels[idx]

    k = min(n_colors, max(2, len(np.unique(fg_pixels, axis=0))))
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, random_state=42,
                             n_init=3, max_iter=10)
    labels = kmeans.fit_predict(fg_pixels)
    centers = kmeans.cluster_centers_.astype(int)

    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    sorted_idx = np.argsort(-counts)

    palette = []
    for idx in sorted_idx:
        rgb = centers[idx].tolist()
        ratio = float(counts[idx]) / total
        name = _rgb_to_name(*rgb)
        palette.append({
            "rgb": rgb,
            "hex": _rgb_to_hex(*rgb),
            "ratio": round(ratio, 4),
            "name": name,
        })

    entropy = _compute_color_entropy(image, mask)
    pattern_type = _infer_pattern_type(palette, entropy, mask, image)

    seen = set()
    primary_names = []
    for c in palette:
        if c["name"] not in seen:
            seen.add(c["name"])
            primary_names.append(c["name"])

    # Confidence
    pixel_conf = min(1.0, len(fg_pixels) / 10000)
    if len(centers) >= 2:
        dists = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dists.append(np.linalg.norm(centers[i].astype(float) - centers[j].astype(float)))
        sep_conf = min(1.0, np.mean(dists) / 150.0) if dists else 0.5
    else:
        sep_conf = 0.5
    confidence = 0.5 * pixel_conf + 0.5 * sep_conf

    return {
        "palette": palette,
        "entropy": round(entropy, 3),
        "primary_names": primary_names,
        "pattern_type": pattern_type,
        "confidence": round(confidence, 3),
    }
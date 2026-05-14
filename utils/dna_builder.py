"""
Cat Pet DNA — Stage 6: DNA Aggregator & JSON Builder
Assembles all module outputs into the final Pet DNA JSON structure.
"""

import json
from datetime import datetime


def build_pet_dna(
    image_size: tuple,
    processing_time_ms: float,
    segmentation_quality: float,
    color_result: dict,
    body_geometry: dict,
    body_size: str,
    head_shape_result: dict,
    ear_result: dict,
    eye_result: dict,
    face_symmetry: float,
    pose_result: dict,
    action_result: dict,
) -> dict:
    """
    Assemble all module outputs into the final Pet DNA JSON.

    Args:
        image_size: (width, height)
        processing_time_ms: total pipeline time
        segmentation_quality: 0~1
        color_result: from color_analysis.extract_color_palette()
        body_geometry: from contour_features.extract_body_geometry()
        body_size: from contour_features.estimate_body_size()
        head_shape_result: from face_analysis.analyze_head_shape()
        ear_result: from face_analysis.detect_ears()
        eye_result: from face_analysis.detect_eyes()
        face_symmetry: from face_analysis.compute_face_symmetry()
        pose_result: from pose_estimation.classify_pose()
        action_result: from pose_estimation.infer_action_tendency()

    Returns:
        dict: complete Pet DNA JSON
    """
    # --- Breed inference (pure rule-based) ---
    breed_candidates = _infer_breed(
        color_result.get("primary_names", []),
        color_result.get("pattern_type", ""),
        ear_result.get("ear_type", ""),
        head_shape_result.get("shape", ""),
        body_size,
    )

    # --- Overall confidence ---
    confidences = [
        segmentation_quality,
        color_result.get("confidence", 0.5),
        pose_result.get("pose_confidence", 0.5),
        head_shape_result.get("confidence", 0.5),
    ]
    overall_conf = sum(confidences) / len(confidences) if confidences else 0.0

    dna = {
        "pet_type": "cat",
        "appearance": {
            "color_palette": color_result.get("palette", []),
            "color_pattern": {
                "type": color_result.get("pattern_type", "unknown"),
                "inference_method": "rule-based: color_entropy + palette_analysis",
            },
            "primary_colors": color_result.get("primary_names", ["unknown"]),
            "body_geometry": body_geometry,
            "body_size_estimate": body_size,
        },
        "breed": {
            "candidates": breed_candidates,
            "inference_method": "rule-based: color+ear+shape matching table",
            "note": "Breed inference is based on surface visual features only. "
                    "It is an estimate, not a genetic test result.",
        },
        "face": {
            "head_shape": {
                "shape": head_shape_result.get("shape", "unknown"),
                "confidence": head_shape_result.get("confidence", 0),
            },
            "ears": {
                "tips_detected": ear_result.get("tips_detected", 0),
                "ear_type": ear_result.get("ear_type", "unknown"),
                "ear_spread_ratio": ear_result.get("ear_spread_ratio", 0),
            },
            "eyes_detected": eye_result.get("eyes_detected", False),
            "face_symmetry": face_symmetry,
        },
        "pose": {
            "pose_type": pose_result.get("pose_type", "unknown"),
            "pose_confidence": pose_result.get("pose_confidence", 0),
            "body_angle": body_geometry.get("body_angle", 0),
            "inference_method": "rule-based: aspect_ratio + extent + body_angle",
        },
        "action_tendency": {
            "state": action_result.get("state", "unknown"),
            "confidence": action_result.get("confidence", 0),
            "inference_method": "rule-based: pose_type + extent_range",
            "evidence": action_result.get("evidence", []),
        },
        "confidence": {
            "overall": round(overall_conf, 3),
            "segmentation_quality": round(segmentation_quality, 3),
            "color_extraction": color_result.get("confidence", 0),
            "pose_estimation": pose_result.get("pose_confidence", 0),
            "face_analysis": head_shape_result.get("confidence", 0),
        },
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "image_size": list(image_size),
            "processing_time_ms": round(processing_time_ms, 1),
            "pipeline_version": "1.0.0",
            "commercial_safe": True,
        },
    }

    return dna


def _infer_breed(colors: list, pattern: str, ear_type: str,
                 head_shape: str, body_size: str) -> list:
    """
    Rule-based breed candidate inference.
    Returns up to 3 candidates with confidence scores.
    """
    candidates = []

    # Rule 1: Solid white + round head
    if "white" in colors and pattern == "solid" and head_shape == "round":
        candidates.append({"name": "Persian / Exotic Shorthair", "confidence": 0.5})
        candidates.append({"name": "Ragdoll", "confidence": 0.3})

    # Rule 2: Tabby + pointed ears
    if pattern == "tabby" and ear_type == "pointed":
        candidates.append({"name": "Domestic Shorthair (Tabby)", "confidence": 0.6})
        if body_size == "large":
            candidates.append({"name": "Maine Coon", "confidence": 0.3})

    # Rule 3: Calico
    if pattern == "calico":
        candidates.append({"name": "Calico (Domestic Shorthair)", "confidence": 0.7})

    # Rule 4: Bicolor (black + white)
    if pattern == "bicolor":
        candidates.append({"name": "Tuxedo (Domestic Shorthair)", "confidence": 0.5})
        if head_shape == "round":
            candidates.append({"name": "British Shorthair", "confidence": 0.3})

    # Rule 5: Solid orange
    if "orange" in colors and pattern == "solid":
        candidates.append({"name": "Orange Tabby (Domestic Shorthair)", "confidence": 0.5})

    # Rule 6: Solid black
    if "black" in colors and pattern == "solid":
        candidates.append({"name": "Bombay / Domestic Black", "confidence": 0.4})

    # Rule 7: Folded ears
    if ear_type == "folded":
        candidates.append({"name": "Scottish Fold", "confidence": 0.6})

    # Rule 8: Wedge head + pointed ears
    if head_shape == "wedge" and ear_type == "pointed":
        candidates.append({"name": "Siamese / Oriental", "confidence": 0.4})
        candidates.append({"name": "Domestic Shorthair", "confidence": 0.3})

    # Fallback
    if not candidates:
        candidates.append({"name": "Domestic Shorthair", "confidence": 0.5})

    # Sort by confidence descending, take top 3
    candidates.sort(key=lambda c: -c["confidence"])
    return candidates[:3]


def dna_to_json(dna: dict, indent: int = 2) -> str:
    """Serialize Pet DNA dict to pretty JSON string."""
    return json.dumps(dna, indent=indent, ensure_ascii=False)
"""
Cat Pet DNA — Stage 5: Pose & Action Tendency
- Pose classification via geometry rules (rule-based)
- Action tendency inference from pose + extent (rule-based)
"""


def classify_pose(aspect_ratio: float, extent: float, body_angle: float) -> dict:
    """
    Classify cat pose from body geometry features.

    Args:
        aspect_ratio: bounding rect height/width (or width/height, whichever > 1)
        extent: area / bounding_rect_area
        body_angle: degrees from horizontal

    Returns dict with: pose_type, confidence, evidence
    """
    evidence = []

    # --- Extent-based primary classification ---
    if extent > 0.6:
        # Compact: curled up or sitting
        if aspect_ratio > 1.2:
            pose_type = "sitting"
            confidence = min(0.9, extent)
            evidence.append(f"extent={extent:.2f}>0.6 (compact)")
            evidence.append(f"aspect_ratio={aspect_ratio:.2f}>1.2 (vertical)")
        else:
            pose_type = "curled_up"
            confidence = min(0.85, extent)
            evidence.append(f"extent={extent:.2f}>0.6 (compact)")
            evidence.append(f"aspect_ratio={aspect_ratio:.2f}<=1.2 (rounded)")

    elif extent < 0.4:
        # Extended: stretching or lying side
        if aspect_ratio < 0.8:
            pose_type = "lying_side"
            confidence = min(0.85, 1.0 - extent)
            evidence.append(f"extent={extent:.2f}<0.4 (extended)")
            evidence.append(f"aspect_ratio={aspect_ratio:.2f}<0.8 (horizontal)")
        elif aspect_ratio > 1.3:
            pose_type = "standing"
            confidence = min(0.8, 1.0 - extent)
            evidence.append(f"extent={extent:.2f}<0.4 (extended)")
            evidence.append(f"aspect_ratio={aspect_ratio:.2f}>1.3 (vertical)")
        else:
            pose_type = "stretching"
            confidence = 0.7
            evidence.append(f"extent={extent:.2f}<0.4 (extended)")
            evidence.append(f"aspect_ratio={aspect_ratio:.2f} in [0.8,1.3]")

    else:
        # Normal range
        if aspect_ratio > 1.2:
            pose_type = "sitting"
            confidence = 0.75
            evidence.append(f"extent={extent:.2f} in [0.4,0.6] (normal)")
            evidence.append(f"aspect_ratio={aspect_ratio:.2f}>1.2 (vertical)")
        elif aspect_ratio < 0.8:
            pose_type = "lying_side"
            confidence = 0.7
            evidence.append(f"extent={extent:.2f} in [0.4,0.6] (normal)")
            evidence.append(f"aspect_ratio={aspect_ratio:.2f}<0.8 (horizontal)")
        else:
            pose_type = "sitting"
            confidence = 0.6
            evidence.append(f"extent={extent:.2f} in [0.4,0.6] (normal)")
            evidence.append(f"aspect_ratio={aspect_ratio:.2f} in [0.8,1.2] (balanced)")

    # Body angle refinement
    if abs(body_angle) < 20 or abs(body_angle - 180) < 20:
        evidence.append(f"body_angle={body_angle:.1f}° (near-horizontal)")
    elif abs(body_angle - 90) < 20:
        evidence.append(f"body_angle={body_angle:.1f}° (near-vertical)")

    return {
        "pose_type": pose_type,
        "pose_confidence": round(confidence, 3),
        "evidence": evidence,
    }


def infer_action_tendency(pose_type: str, extent: float) -> dict:
    """
    Infer action tendency from pose and extent.

    Returns dict with: state, confidence, evidence
    """
    evidence = [f"pose={pose_type}", f"extent={extent:.2f}"]

    # Rule mapping: pose + extent → action state
    if pose_type == "curled_up":
        state = "resting"
        confidence = 0.9
        evidence.append("curled posture → resting state")
    elif pose_type == "lying_side":
        if extent < 0.3:
            state = "active"
            confidence = 0.7
            evidence.append("side-lying with low extent → may be stretching/active")
        else:
            state = "resting"
            confidence = 0.8
            evidence.append("side-lying with normal extent → resting")
    elif pose_type == "sitting":
        if extent > 0.55:
            state = "resting"
            confidence = 0.7
            evidence.append("compact sitting → resting/alert")
        else:
            state = "alert"
            confidence = 0.75
            evidence.append("upright sitting → alert state")
    elif pose_type == "standing":
        if extent < 0.3:
            state = "playful"
            confidence = 0.65
            evidence.append("standing with low extent → playful/hunting")
        else:
            state = "alert"
            confidence = 0.7
            evidence.append("standing → alert state")
    elif pose_type == "stretching":
        state = "active"
        confidence = 0.8
        evidence.append("stretching posture → active state")
    else:
        state = "unknown"
        confidence = 0.3
        evidence.append("unknown pose → cannot infer")

    return {
        "state": state,
        "confidence": round(confidence, 3),
        "evidence": evidence,
    }
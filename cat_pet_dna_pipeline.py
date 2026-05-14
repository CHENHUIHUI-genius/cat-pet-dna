#!/usr/bin/env python3
"""
Cat Pet DNA — Main Pipeline Entry Point
Low-compute visual system for generating structured Pet DNA from cat images.

Usage:
    # Single image
    python3 cat_pet_dna_pipeline.py --input cat.jpg --output ./output/

    # Batch mode
    python3 cat_pet_dna_pipeline.py --input ./test_images/ --batch --output ./output/

    # Camera mode
    python3 cat_pet_dna_pipeline.py --camera 0
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.segmentation import segment_foreground, extract_head_roi
from utils.color_analysis import extract_color_palette
from utils.contour_features import extract_body_geometry, estimate_body_size
from utils.face_analysis import (
    analyze_head_shape, detect_ears, detect_eyes, compute_face_symmetry,
)
from utils.pose_estimation import classify_pose, infer_action_tendency
from utils.dna_builder import build_pet_dna, dna_to_json
from demo.visualize import draw_dna_overlay


def process_image(image: np.ndarray) -> dict:
    """
    Run full Pet DNA pipeline on a single image.

    Args:
        image: RGB image (H, W, 3)

    Returns:
        dict with: dna (Pet DNA dict), mask, overlay, ear_tips, time_ms
    """
    t_start = time.time()
    h, w = image.shape[:2]

    # --- Stage 1: Segmentation ---
    mask, seg_quality = segment_foreground(image)

    # --- Stage 2: Color Analysis ---
    color_result = extract_color_palette(image, mask)

    # --- Stage 3: Body Geometry ---
    body_geo = extract_body_geometry(mask)
    body_size = estimate_body_size(body_geo["area_px"], h * w)

    # --- Stage 4: Face Analysis ---
    head_mask, head_img = extract_head_roi(image, mask)
    head_shape_result = analyze_head_shape(head_mask)
    ear_result = detect_ears(head_mask)
    eye_result = detect_eyes(head_img)
    face_symmetry = compute_face_symmetry(head_mask)

    # --- Stage 5: Pose & Action ---
    pose_result = classify_pose(
        body_geo["aspect_ratio"],
        body_geo["extent"],
        body_geo["body_angle"],
    )
    action_result = infer_action_tendency(
        pose_result["pose_type"],
        body_geo["extent"],
    )

    # --- Stage 6: DNA Assembly ---
    t_elapsed = (time.time() - t_start) * 1000

    dna = build_pet_dna(
        image_size=(w, h),
        processing_time_ms=t_elapsed,
        segmentation_quality=seg_quality,
        color_result=color_result,
        body_geometry=body_geo,
        body_size=body_size,
        head_shape_result=head_shape_result,
        ear_result=ear_result,
        eye_result=eye_result,
        face_symmetry=face_symmetry,
        pose_result=pose_result,
        action_result=action_result,
    )

    # --- Visualization ---
    ear_tips = ear_result.get("ear_tips", [])
    overlay = draw_dna_overlay(image, mask, dna, ear_tips)

    return {
        "dna": dna,
        "mask": mask,
        "overlay": overlay,
        "ear_tips": ear_tips,
        "time_ms": t_elapsed,
    }


def save_results(output_dir: str, basename: str, result: dict):
    """Save DNA JSON, overlay image, and report."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f"{basename}_dna.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(dna_to_json(result["dna"]))
    print(f"  [OK] DNA JSON → {json_path}")

    # Overlay image
    overlay_path = os.path.join(output_dir, f"{basename}_visualized.jpg")
    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, overlay_bgr)
    print(f"  [OK] Visualized → {overlay_path}")

    # Mask
    mask_path = os.path.join(output_dir, f"{basename}_mask.png")
    cv2.imwrite(mask_path, result["mask"])
    print(f"  [OK] Mask → {mask_path}")


def run_single(input_path: str, output_dir: str):
    """Process a single image file."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"{'='*60}")

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f"  [ERROR] Cannot read image: {input_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    result = process_image(img_rgb)

    basename = Path(input_path).stem
    save_results(output_dir, basename, result)

    # Print summary
    dna = result["dna"]
    print(f"\n  ⏱  {result['time_ms']:.1f}ms")
    print(f"  🎨 Pattern: {dna['appearance']['color_pattern']['type']}")
    print(f"     Colors: {', '.join(dna['appearance']['primary_colors'])}")
    print(f"  🧍 Pose: {dna['pose']['pose_type']} ({dna['pose']['pose_confidence']:.0%})")
    print(f"  🎯 Action: {dna['action_tendency']['state']} ({dna['action_tendency']['confidence']:.0%})")
    print(f"  🐱 Breed: {dna['breed']['candidates'][0]['name']}")
    print(f"  📊 Confidence: {dna['confidence']['overall']:.1%}")


def run_batch(input_dir: str, output_dir: str):
    """Process all images in a directory."""
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(extensions)
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"\nBatch processing {len(image_files)} images from {input_dir}")
    total_time = 0

    for fname in image_files:
        fpath = os.path.join(input_dir, fname)
        run_single(fpath, output_dir)
        # Read time from last result
        result_path = os.path.join(output_dir, f"{Path(fname).stem}_dna.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                dna = json.load(f)
                total_time += dna["meta"]["processing_time_ms"]

    avg_time = total_time / len(image_files)
    print(f"\n{'='*60}")
    print(f"Batch complete: {len(image_files)} images")
    print(f"Total: {total_time:.0f}ms | Avg: {avg_time:.1f}ms per image")
    print(f"{'='*60}")


def run_camera(camera_id: int):
    """Real-time camera demo."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_id}")
        return

    print(f"\nCamera {camera_id} opened. Press 'q' to quit, 's' to save snapshot.")
    frame_count = 0
    process_every = 5  # Process every Nth frame
    last_result = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Process every N frames
        if frame_count % process_every == 0:
            try:
                result = process_image(frame_rgb)
                last_result = result
            except Exception as e:
                print(f"  [WARN] Frame {frame_count}: {e}")
                last_result = None

        # Display
        if last_result:
            display = last_result["overlay"].copy()
            # Add FPS
            fps = 1000.0 / max(last_result["time_ms"], 1)
            cv2.putText(display, f"FPS: {fps:.1f}  |  {last_result['time_ms']:.0f}ms",
                        (w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            display = frame_rgb.copy()
            cv2.putText(display, "Detecting...", (w // 2 - 60, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        cv2.imshow("Cat Pet DNA Live", display_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and last_result:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_results("output", f"snapshot_{timestamp}", last_result)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Cat Pet DNA — Low-compute visual system for cat analysis"
    )
    parser.add_argument("--input", "-i", type=str, help="Input image path or directory")
    parser.add_argument("--output", "-o", type=str, default="./output/",
                        help="Output directory")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: input is a directory")
    parser.add_argument("--camera", "-c", type=int, default=None,
                        help="Camera device ID for live demo")
    args = parser.parse_args()

    if args.camera is not None:
        run_camera(args.camera)
    elif args.input:
        if args.batch:
            run_batch(args.input, args.output)
        else:
            run_single(args.input, args.output)
    else:
        parser.print_help()
        print("\nExample:")
        print("  python3 cat_pet_dna_pipeline.py --input cat.jpg --output ./output/")
        print("  python3 cat_pet_dna_pipeline.py --input ./images/ --batch --output ./output/")
        print("  python3 cat_pet_dna_pipeline.py --camera 0")


if __name__ == "__main__":
    main()
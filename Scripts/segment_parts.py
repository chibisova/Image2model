"""
Open-vocabulary part + clothing segmentation for turntable renders.

Pipeline: YOLO-World (text-prompted detection) -> MobileSAM (box -> mask)
Runs on Apple Silicon via MPS, falls back to CPU automatically elsewhere.

Reads:  <input_dir>/*.png              (your turntable renders)
Writes: <output_dir>/<image_stem>/
            <prompt>_mask.png          (binary mask, white = detected region)
            overlay.png                (all masks color-coded on top of the image)
        <output_dir>/detections.json   (all boxes + scores + prompt labels, one entry per image)

SETUP (run once)
-----------------
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision ultralytics opencv-python pillow numpy
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

Download the MobileSAM checkpoint (mobile_sam.pt) from the MobileSAM repo's
weights folder and set MOBILE_SAM_CHECKPOINT below to its path.

USAGE
-----
python3 segment_parts.py
"""

import os
import json
import numpy as np
import cv2
import torch

from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

# =========================================================
# CONFIG - edit these
# =========================================================
INPUT_DIR = "./outputs/turntable_output/images"
OUTPUT_DIR = "./outputs/segmentation_output"

MOBILE_SAM_CHECKPOINT = "../model/mobile_sam.pt"   # path to downloaded checkpoint
MOBILE_SAM_TYPE = "vit_t"                 # MobileSAM's model type key

# Text prompts for parts/clothing you want segmented.
# Tune these based on what actually localizes well - see note at bottom.
PROMPTS = [
    "dog head",
    "dog ear",
    "dog leg",
    "dog paw",
    "dog tail",
    "sweater",
    "collar",
]

CONFIDENCE_THRESHOLD = 0.15   # YOLO-World tends to need a lower threshold
                               # than closed-vocab YOLO for uncommon prompts
IOU_THRESHOLD = 0.5

# Distinct colors per prompt for the overlay visualization (BGR)
OVERLAY_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (0, 128, 255),
]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models(device):
    yolo = YOLO("yolov8s-world.pt")   # auto-downloads on first run
    yolo.set_classes(PROMPTS)

    sam = sam_model_registry[MOBILE_SAM_TYPE](checkpoint=MOBILE_SAM_CHECKPOINT)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    return yolo, predictor


def run_on_image(image_path, yolo, predictor, device, out_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"WARNING: could not read {image_path}, skipping")
        return []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = yolo.predict(
        image_rgb, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
        device=device, verbose=False,
    )[0]

    predictor.set_image(image_rgb)

    overlay = image.copy()
    detections = []

    for i, box in enumerate(results.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        label = PROMPTS[cls_id]

        mask, score, _ = predictor.predict(
            box=xyxy, multimask_output=False,
        )
        mask = mask[0]  # (H, W) boolean

        mask_path = os.path.join(out_dir, f"{label.replace(' ', '_')}_mask.png")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        x1, y1, x2, y2 = xyxy.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{label} {conf:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        detections.append({
            "label": label,
            "confidence": conf,
            "box_xyxy": xyxy.tolist(),
            "mask_score": float(score[0]),
            "mask_path": os.path.relpath(mask_path, out_dir),
        })

    cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)
    return detections


def main():
    device = get_device()
    print(f"Using device: {device}")

    if device == "cpu":
        print("NOTE: running on CPU. YOLO-World detection will be the slow part; "
              "MobileSAM mask refinement is fast regardless.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yolo, predictor = load_models(device)

    image_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_files:
        raise RuntimeError(f"No images found in {INPUT_DIR}")

    all_detections = {}

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(INPUT_DIR, filename)
        stem = os.path.splitext(filename)[0]
        image_out_dir = os.path.join(OUTPUT_DIR, stem)
        os.makedirs(image_out_dir, exist_ok=True)

        dets = run_on_image(image_path, yolo, predictor, device, image_out_dir)
        all_detections[filename] = dets

        print(f"[{idx + 1}/{len(image_files)}] {filename}: {len(dets)} detections")

    with open(os.path.join(OUTPUT_DIR, "detections.json"), "w") as f:
        json.dump(all_detections, f, indent=2)

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------
# TUNING NOTE
# -----------------------------------------------------------------------
# Check overlay.png for a few views before running the full batch. If a
# prompt like "dog paw" isn't detecting reliably, try:
#   - rephrasing ("paw", "foot" instead of "dog paw")
#   - lowering CONFIDENCE_THRESHOLD further (down to ~0.05)
#   - splitting compound prompts into simpler single words
# YOLO-World's open-vocab detection is noticeably weaker on multi-word or
# uncommon part names than on everyday object categories.

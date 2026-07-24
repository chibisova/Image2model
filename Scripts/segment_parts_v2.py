"""
Interactive point-click segmentation for stylized meshes.

Text-prompted detectors (YOLO-World, Grounding DINO) are trained on real
photos and fail on stylized/toon-shaded renders - they have no strong
"ear"/"leg" features to key off of. This script sidesteps that: you click
points directly on each render, MobileSAM turns clicks into a mask. Mask
quality only depends on SAM's edge/region understanding, not on the model
recognizing "dog" as a concept, so it holds up fine on stylized geometry.

CONTROLS (per image, per label)
--------------------------------
  Left click       add a POSITIVE point (this is part of the region)
  Right click       add a NEGATIVE point (exclude this area, use to fix
                    the mask if it's bleeding into a neighboring part)
  c                 clear all points for this label, start over
  n / Enter         confirm mask, save it, move to next label
  s                 skip this label for this image (part not visible
                    in this view - normal for a turntable, don't force it)
  q                 quit entirely (progress so far is already saved)

Reads:  <input_dir>/*.png
Writes: <output_dir>/<image_stem>/<label>_mask.png
        <output_dir>/<image_stem>/points.json   (so you can redo/audit later)
        <output_dir>/detections.json             (same shape as the
                                                    YOLO-World script, so the
                                                    downstream projection step
                                                    doesn't need to change)

SETUP
-----
Same environment as segment_parts.py (torch, opencv-python, mobile_sam).
No ultralytics/YOLO needed for this script.

USAGE
-----
python3 segment_parts_interactive.py
"""

import os
import json
import numpy as np
import cv2
import torch

from mobile_sam import sam_model_registry, SamPredictor

# =========================================================
# CONFIG - edit these
# =========================================================
INPUT_DIR = "./outputs/turntable_output/images"
OUTPUT_DIR = "./outputs/segmentation_output_v2"

MOBILE_SAM_CHECKPOINT = "../model/mobile_sam.pt"
MOBILE_SAM_TYPE = "vit_t"

LABELS = [
    "dog head",
    "dog ear",
    "dog leg",
    "dog paw",
    "dog tail",
    "sweater",
    "collar",
]

DISPLAY_MAX_DIM = 900   # downscale large renders for on-screen clicking;
                          # clicks are mapped back to full resolution
MASK_ALPHA = 0.5
POS_COLOR = (0, 255, 0)   # BGR
NEG_COLOR = (0, 0, 255)
MASK_COLOR = (255, 200, 0)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_predictor(device):
    sam = sam_model_registry[MOBILE_SAM_TYPE](checkpoint=MOBILE_SAM_CHECKPOINT)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


class ClickState:
    def __init__(self):
        self.points = []   # list of (x, y) in full-res image coords
        self.labels = []   # 1 = positive, 0 = negative
        self.done = None   # "confirm" | "skip" | "quit" | None


def make_mouse_callback(state, scale):
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.points.append((x / scale, y / scale))
            state.labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.points.append((x / scale, y / scale))
            state.labels.append(0)
    return callback


def render_display(image, scale, state, mask):
    disp = cv2.resize(image, None, fx=scale, fy=scale)

    if mask is not None:
        mask_disp = cv2.resize(mask.astype(np.uint8), (disp.shape[1], disp.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = disp.copy()
        overlay[mask_disp] = MASK_COLOR
        disp = cv2.addWeighted(overlay, MASK_ALPHA, disp, 1 - MASK_ALPHA, 0)

    for (x, y), lab in zip(state.points, state.labels):
        color = POS_COLOR if lab == 1 else NEG_COLOR
        cv2.circle(disp, (int(x * scale), int(y * scale)), 5, color, -1)

    return disp


def segment_label_interactive(image, predictor, label, window_name):
    h, w = image.shape[:2]
    scale = min(1.0, DISPLAY_MAX_DIM / max(h, w))

    state = ClickState()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, make_mouse_callback(state, scale))

    mask = None
    score = None

    while True:
        if state.points:
            pts = np.array(state.points)
            labs = np.array(state.labels)
            masks, scores, _ = predictor.predict(
                point_coords=pts, point_labels=labs, multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask, score = masks[best], float(scores[best])
        else:
            mask, score = None, None

        disp = render_display(image, scale, state, mask)
        cv2.putText(disp, f"label: {label}  (n=confirm  s=skip  c=clear  q=quit)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("n") or key == 13:  # Enter
            if mask is None:
                continue  # need at least one point before confirming
            state.done = "confirm"
            break
        elif key == ord("s"):
            state.done = "skip"
            break
        elif key == ord("c"):
            state.points, state.labels = [], []
        elif key == ord("q"):
            state.done = "quit"
            break

    return state.done, mask, score, state.points, state.labels


def main():
    device = get_device()
    print(f"Using device: {device}")
    predictor = load_predictor(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_files:
        raise RuntimeError(f"No images found in {INPUT_DIR}")

    all_detections = {}
    window_name = "click to segment"

    quit_all = False
    for idx, filename in enumerate(image_files):
        if quit_all:
            break

        image_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"WARNING: could not read {image_path}, skipping")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        stem = os.path.splitext(filename)[0]
        image_out_dir = os.path.join(OUTPUT_DIR, stem)
        os.makedirs(image_out_dir, exist_ok=True)

        detections = []
        all_points_log = {}

        print(f"\n[{idx + 1}/{len(image_files)}] {filename}")

        for label in LABELS:
            action, mask, score, points, plabels = segment_label_interactive(
                image, predictor, label, window_name
            )

            if action == "quit":
                quit_all = True
                break
            elif action == "skip":
                continue
            elif action == "confirm":
                mask_path = os.path.join(image_out_dir, f"{label.replace(' ', '_')}_mask.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

                detections.append({
                    "label": label,
                    "confidence": 1.0,     # human-confirmed, not a model score
                    "mask_score": score,
                    "mask_path": os.path.relpath(mask_path, image_out_dir),
                })
                all_points_log[label] = {"points": points, "point_labels": plabels}

        all_detections[filename] = detections
        with open(os.path.join(image_out_dir, "points.json"), "w") as f:
            json.dump(all_points_log, f, indent=2)

        # save progress after every image so a mid-batch quit doesn't lose work
        with open(os.path.join(OUTPUT_DIR, "detections.json"), "w") as f:
            json.dump(all_detections, f, indent=2)

    cv2.destroyAllWindows()
    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

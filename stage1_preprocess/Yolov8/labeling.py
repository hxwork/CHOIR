"""
Interactive video annotation with SAM2.1 on the first frame and optional
full-video tracking.

Default workflow (RUN_TRACKING=False):
    Annotate the first frame, infer a SAM2 mask, and save the mask and prompt
    for the downstream tracking pipeline.

Optional local tracking workflow:
    Use first_mask.png as the initial SAM2 prompt and propagate it through the
    complete video.

Usage:
    python labeling.py
    python labeling.py --track
    python labeling.py --video-id video_001 video_002
    python labeling.py --ckpt /path/to/sam2.1_hiera_large.pt

Input and annotation layout:
    data/<video_id>.mp4
    output/<video_id>/annotations/first_mask.png

Controls:
    Left drag     : Draw the object box in BBox mode
    Left click    : Add a positive point in Point mode
    Right click   : Add a negative point in Point mode
    b             : Switch to BBox mode
    p             : Switch to Point mode
    Enter / Space : Run SAM2 with the current prompts
    s             : Save the current mask
    r             : Reset prompts and mask
    n             : Skip the current video
    q             : Quit
"""

import argparse
import json
import os
import sys
import shutil

import cv2
import numpy as np
import torch

from data_layout import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DATA_DIR,
    annotation_dir_for_video,
    discover_input_videos,
)

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR = os.path.join(SCRIPT_DIR, "sam2")
DATA_DIR = None  # set in main() via --data
OUTPUT_DIR = None  # set in main() via --output

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = os.path.join(SAM2_DIR, "checkpoints", "sam2.1_hiera_large.pt")

# Whether to run full-video tracking locally
RUN_TRACKING = False

# Prefer MPS on Apple Silicon, then CUDA, then CPU.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Using device: {DEVICE}")

# Add the local SAM2 package to the import path.
sys.path.insert(0, SAM2_DIR)

from sam2.build_sam import build_sam2_video_predictor


# Extract video frames because SAM2 init_state requires a JPEG directory.
def extract_frames(video_path: str, out_dir: str) -> list[np.ndarray]:
    """Write JPEG frames to out_dir and return them as BGR arrays."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(out_dir, f"{idx:06d}.jpg")
        cv2.imwrite(path, frame)
        frames.append(frame)
        idx += 1
    cap.release()
    print(f"[INFO] Extracted {len(frames)} frames to {out_dir}")
    return frames


# Interactive annotator
class VideoAnnotator:
    MODE_BBOX = "bbox"
    MODE_POINT = "point"

    def __init__(self, frame: np.ndarray):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.mode = self.MODE_BBOX

        # Bounding-box state
        self.bbox_start = None
        self.bbox_end = None
        self.bbox_final = None  # Confirmed (x1, y1, x2, y2)

        # Point-prompt state
        self.points = []  # [(x, y), ...]
        self.labels = []  # [1 or 0, ...]

        # Drag state
        self._dragging = False

    # Rendering
    def _redraw(self):
        img = self.frame.copy()
        # Confirmed bounding box
        if self.bbox_final is not None:
            x1, y1, x2, y2 = self.bbox_final
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # Bounding box currently being dragged
        if self._dragging and self.bbox_start and self.bbox_end:
            cv2.rectangle(img, self.bbox_start, self.bbox_end, (0, 200, 200), 1)
        # Positive and negative points
        for (x, y), lbl in zip(self.points, self.labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(img, (x, y), 6, color, -1)
            cv2.circle(img, (x, y), 6, (255, 255, 255), 1)

        # Status bar
        mode_txt = f"Mode: {'BBox' if self.mode == self.MODE_BBOX else 'Point'}"
        hint = "b=BBox  p=Point  Enter=Run SAM2  s=Save  r=Reset  n=Skip  q=Quit"
        h = img.shape[0]
        cv2.rectangle(img, (0, h - 50), (img.shape[1], h), (30, 30, 30), -1)
        cv2.putText(img, mode_txt, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(img, hint, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        self.display = img

    # Mouse callback
    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == self.MODE_BBOX:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.bbox_start = (x, y)
                self.bbox_end = None
                self._dragging = True
                self.bbox_final = None
            elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
                self.bbox_end = (x, y)
                self._redraw()
            elif event == cv2.EVENT_LBUTTONUP:
                self._dragging = False
                if self.bbox_start:
                    x1 = min(self.bbox_start[0], x)
                    y1 = min(self.bbox_start[1], y)
                    x2 = max(self.bbox_start[0], x)
                    y2 = max(self.bbox_start[1], y)
                    if x2 - x1 > 5 and y2 - y1 > 5:
                        self.bbox_final = (x1, y1, x2, y2)
                self._redraw()
        elif self.mode == self.MODE_POINT:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                self.labels.append(1)
                self._redraw()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.points.append((x, y))
                self.labels.append(0)
                self._redraw()

    # Prompt state
    def has_prompt(self) -> bool:
        return self.bbox_final is not None or len(self.points) > 0

    # Reset
    def reset(self):
        self.bbox_start = None
        self.bbox_end = None
        self.bbox_final = None
        self.points = []
        self.labels = []
        self._dragging = False
        self._redraw()

    # Convert prompts to SAM2 inputs.
    def get_sam2_inputs(self):
        box = np.array(self.bbox_final, dtype=np.float32) if self.bbox_final else None
        points = np.array(self.points, dtype=np.float32) if self.points else None
        labels = np.array(self.labels, dtype=np.int32) if self.labels else None
        return box, points, labels


# Overlay a segmentation mask for visualization.
def overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    out = frame.copy()
    m = mask.squeeze().astype(bool)
    overlay = out.copy()
    overlay[m] = (np.array(color) * alpha + overlay[m] * (1 - alpha)).astype(np.uint8)
    # Object contour
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


# Process one video.
def process_video(predictor, video_path: str):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = str(annotation_dir_for_video(OUTPUT_DIR, video_path))
    frames_dir = os.path.join(out_dir, "_frames")

    print(f"\n{'='*60}")
    print(f"[INFO] Processing video: {video_name}")

    # Extract all frames because optional tracking requires the complete video.
    frames = extract_frames(video_path, frames_dir)
    if not frames:
        print("[WARN] Could not read any video frames; skipping.")
        return False

    first_frame = frames[0]

    # Interactive windows
    win_name = f"SAM2 Annotator - {video_name}"
    preview_win = f"Video Preview - {video_name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)

    h, w = first_frame.shape[:2]
    scale = min(1400 / w, 900 / h, 1.0)
    win_w, win_h = int(w * scale), int(h * scale)
    cv2.resizeWindow(win_name, win_w, win_h)
    cv2.resizeWindow(preview_win, win_w, win_h)
    # Place the annotation and preview windows side by side.
    cv2.moveWindow(win_name, 0, 50)
    cv2.moveWindow(preview_win, win_w + 10, 50)

    annotator = VideoAnnotator(first_frame)
    cv2.setMouseCallback(win_name, annotator.mouse_callback)
    annotator._redraw()

    # Preview playback state at approximately 30 fps.
    play_idx = 0
    tick_count = 0
    TICKS_PER_FRAME = max(1, round(33 / 20))  # waitKey=20ms

    # Initialize once so prompts can be adjusted and rerun.
    print("[INFO] Initializing SAM2 inference state...")
    inference_state = predictor.init_state(video_path=frames_dir)

    current_mask = None  # H x W bool array; None until the first inference.
    obj_id = 1

    result = "skip"
    while True:
        cv2.imshow(win_name, annotator.display)
        cv2.imshow(preview_win, frames[play_idx])

        tick_count += 1
        if tick_count >= TICKS_PER_FRAME:
            tick_count = 0
            play_idx = (play_idx + 1) % len(frames)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('b'):
            annotator.mode = VideoAnnotator.MODE_BBOX
            annotator._redraw()
        elif key == ord('p'):
            annotator.mode = VideoAnnotator.MODE_POINT
            annotator._redraw()
        elif key == ord('r'):
            # Reset prompts and remove the current mask overlay.
            current_mask = None
            annotator.frame = first_frame.copy()
            annotator.reset()
            print("[INFO] Prompt reset.")
        elif key == ord('n'):
            print("[INFO] Skip.")
            result = "skip"
            break
        elif key == ord('q'):
            result = "quit"
            break
        elif key in (13, 32):  # Enter or Space: run or rerun SAM2.
            if not annotator.has_prompt():
                print("[WARN] Please add a BBox or Point first!")
                continue
            print("[INFO] Running SAM2...")
            predictor.reset_state(inference_state)
            box, points, labels = annotator.get_sam2_inputs()
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=box,
                points=points,
                labels=labels,
            )
            current_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            # Use the mask overlay as the background while retaining prompts.
            annotator.frame = overlay_mask(first_frame, current_mask[np.newaxis])
            annotator._redraw()
            print("[INFO] Mask updated. Adjust prompts and press Enter to rerun, or press s to save.")
        elif key == ord('s'):  # Save the confirmed mask.
            if current_mask is None:
                print("[WARN] Run SAM2 first (press Enter)!")
                continue
            result = "run"
            break

    cv2.destroyWindow(win_name)
    cv2.destroyWindow(preview_win)

    if result == "quit":
        return "quit"
    if result == "skip":
        shutil.rmtree(frames_dir, ignore_errors=True)
        return True

    mask0 = current_mask  # Inferred in the interaction loop.

    # Save first-frame annotation artifacts.
    os.makedirs(out_dir, exist_ok=True)

    # Original first frame
    cv2.imwrite(os.path.join(out_dir, "frame_000000.jpg"), first_frame)

    # Binary first-frame mask used as the downstream SAM2 prompt.
    mask0_img = (mask0 * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "first_mask.png"), mask0_img)

    # First-frame mask overlay
    cv2.imwrite(os.path.join(out_dir, "first_mask_overlay.jpg"), overlay_mask(first_frame, mask0[np.newaxis]))

    # Prompt metadata for inspection or rerunning the annotation.
    prompt_meta = {
        "video": os.path.basename(video_path),
        "frame": 0,
        "obj_id": obj_id,
        "bbox": annotator.bbox_final,  # (x1,y1,x2,y2) or null
        "points": annotator.points,  # [[x,y], ...]
        "labels": annotator.labels,  # [1/0, ...]
    }
    with open(os.path.join(out_dir, "prompt.json"), "w") as f:
        json.dump(prompt_meta, f, indent=2)

    print(f"[INFO] Saved first-frame annotation to: {out_dir}")
    print("       first_mask.png        <- downstream tracking prompt")
    print("       prompt.json           <- prompt metadata")

    # Optionally propagate the mask through the full video.
    if not RUN_TRACKING:
        print("[INFO] RUN_TRACKING=False; skipping full-video tracking.")
        shutil.rmtree(frames_dir, ignore_errors=True)
        return True

    print("[INFO] Starting full-video mask propagation...")
    mask_dir = os.path.join(out_dir, "masks")
    overlay_dir = os.path.join(out_dir, "overlay")
    rgba_dir = os.path.join(out_dir, "rgba")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(rgba_dir, exist_ok=True)

    all_masks = {}

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
        all_masks[frame_idx] = mask

        alpha = (mask * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(mask_dir, f"{frame_idx}.png"),
            alpha,
        )
        cv2.imwrite(
            os.path.join(overlay_dir, f"{frame_idx:06d}.jpg"),
            overlay_mask(frames[frame_idx], mask[np.newaxis]),
        )
        # RGBA output: RGB stores the source frame and alpha stores the mask.
        bgra = np.dstack([frames[frame_idx], alpha])
        cv2.imwrite(
            os.path.join(rgba_dir, f"{frame_idx}.png"),
            bgra,
        )
        if frame_idx % 50 == 0:
            print(f"  -> Processed frame {frame_idx}/{len(frames)-1}")

    np.savez_compressed(
        os.path.join(out_dir, "masks.npz"),
        **{
            str(k): v for k, v in all_masks.items()
        },
    )
    print(f"[INFO] Full-video tracking complete. Results saved to: {out_dir}")
    print("       masks/    <- binary mask PNG files")
    print("       overlay/  <- mask-overlay JPEG files")
    print("       rgba/     <- foreground RGBA PNG files")

    shutil.rmtree(frames_dir, ignore_errors=True)
    return True


# Command-line entry point
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Interactive SAM2.1 video annotation")
    parser.add_argument("--track", action="store_true", help="Also propagate the mask through the full video.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR), help=f"Directory containing flat <video_id>.mp4 inputs (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DATA_DIR), help=f"Root for output/<video_id>/ artifacts (default: {DEFAULT_OUTPUT_DATA_DIR})")
    parser.add_argument("--video-id", "--video_id", nargs="+", default=None, help="Only process these video IDs, without file extensions.")
    parser.add_argument("--ckpt", default=SAM2_CKPT, help="Path to sam2.1_hiera_large.pt.")
    return parser


def main():
    global RUN_TRACKING, DATA_DIR, OUTPUT_DIR, SAM2_CKPT

    args = build_arg_parser().parse_args()
    if args.track:
        RUN_TRACKING = True
    SAM2_CKPT = args.ckpt if os.path.isabs(args.ckpt) else os.path.abspath(args.ckpt)

    data = args.data
    DATA_DIR = data if os.path.isabs(data) else os.path.join(SCRIPT_DIR, data)
    output = args.output
    OUTPUT_DIR = output if os.path.isabs(output) else os.path.join(SCRIPT_DIR, output)

    print(f"[INFO] RUN_TRACKING = {RUN_TRACKING}")

    videos = [str(video_path) for video_path in discover_input_videos(DATA_DIR, args.video_id)]

    if not videos:
        if args.video_id:
            print(f"[ERROR] Requested video IDs were not found in {DATA_DIR}: {args.video_id}")
        else:
            print(f"[ERROR] No input videos found in {DATA_DIR}.")
        sys.exit(1)

    if args.video_id:
        found_video_ids = {os.path.splitext(os.path.basename(video_path))[0] for video_path in videos}
        missing_video_ids = [video_id for video_id in args.video_id if video_id not in found_video_ids]
        if missing_video_ids:
            print(f"[WARN] Video IDs not found and skipped: {missing_video_ids}")

    print(f"[INFO] Found {len(videos)} videos: {[os.path.basename(v) for v in videos]}")

    # Load the SAM2 model.
    if not os.path.isfile(SAM2_CKPT):
        print(f"[ERROR] SAM2 checkpoint not found: {SAM2_CKPT}")
        print("        Download the checkpoint or pass its path with --ckpt.")
        sys.exit(1)

    print(f"[INFO] Loading SAM2.1 Large on {DEVICE}...")
    predictor = build_sam2_video_predictor(
        config_file=SAM2_CONFIG,
        ckpt_path=SAM2_CKPT,
        device=DEVICE,
    )
    print("[INFO] Model loaded.")

    for video_path in videos:
        ret = process_video(predictor, video_path)
        if ret == "quit":
            print("[INFO] User requested exit.")
            break

    print("\n[INFO] Finished processing all videos.")


if __name__ == "__main__":
    main()

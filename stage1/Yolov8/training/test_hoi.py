import json
import multiprocessing as mp
import os
import random
import shutil
from pathlib import Path

import cv2
import imageio
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Global cache for models in worker processes
_WORKER_MODELS = {}

if torch.cuda.is_available():
    # no need to define autocast, directly use torch.amp.autocast
    from torch.amp import autocast
else:
    # fallback for CPU
    class autocast:

        def __init__(self, device_type='cpu', enabled=True):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


def process_video_to_frames(video_path, output_folder):
    """
    extract frames from video and save as images
    
    Args:
        video_path: video file path
        output_folder: output folder for images
    
    Returns:
        imgfiles: list of image file paths
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    imgfiles = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # save frame as image
        img_path = os.path.join(output_folder, f'{frame_idx:06d}.jpg')
        cv2.imwrite(img_path, frame)
        imgfiles.append(img_path)
        frame_idx += 1

    cap.release()
    return imgfiles


def draw_detections(img, right_hand_info=None, left_hand_info=None, object_infos=None):
    """
    Draw hand and object detection results on image.

    Args:
        img: input image (numpy array)
        right_hand_info: right hand info [class_id, x_center, y_center, w, h, kpt1_x, kpt1_y, kpt1_conf, ...]
        left_hand_info: left hand info [class_id, x_center, y_center, w, h, kpt1_x, kpt1_y, kpt1_conf, ...]
        object_infos: list of object infos, each is [x_center, y_center, w, h]
    
    Returns:
        img: image with detection results drawn
    """
    img_h, img_w = img.shape[:2]

    # Define colors (BGR format)
    RIGHT_HAND_COLOR = (0, 255, 0)  # Green for right hand
    LEFT_HAND_COLOR = (0, 0, 255)  # Red for left hand
    OBJECT_COLOR = (255, 0, 0)  # Blue for object
    KEYPOINT_COLOR = (255, 255, 0)  # Cyan for keypoints

    def draw_hand(hand_info, color, hand_name):
        if hand_info is None or len(hand_info) < 5:
            return

        # Extract bbox info (normalized)
        _, x_center, y_center, w, h = hand_info[:5]

        # Convert to pixel coordinates
        x_center_px = int(x_center * img_w)
        y_center_px = int(y_center * img_h)
        w_px = int(w * img_w)
        h_px = int(h * img_h)

        # Calculate bbox corners
        x1 = int(x_center_px - w_px / 2)
        y1 = int(y_center_px - h_px / 2)
        x2 = int(x_center_px + w_px / 2)
        y2 = int(y_center_px + h_px / 2)

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = hand_name
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 10, label_size[1])
        cv2.rectangle(img, (x1, label_y - label_size[1] - 5), (x1 + label_size[0], label_y + 5), color, -1)
        cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw keypoints (starting from index 5, each keypoint has [x, y, conf])
        keypoints_data = hand_info[5:]
        num_keypoints = len(keypoints_data) // 3

        for i in range(num_keypoints):
            kpt_x = keypoints_data[i * 3]
            kpt_y = keypoints_data[i * 3 + 1]
            kpt_conf = keypoints_data[i * 3 + 2]

            if kpt_conf > 0:  # Only draw if confident
                kpt_x_px = int(kpt_x * img_w)
                kpt_y_px = int(kpt_y * img_h)
                cv2.circle(img, (kpt_x_px, kpt_y_px), 3, KEYPOINT_COLOR, -1)
                cv2.circle(img, (kpt_x_px, kpt_y_px), 4, color, 1)

    def draw_object(object_info, color):
        if object_info is None or len(object_info) < 5:
            return

        # Extract bbox info (normalized)
        x_center, y_center, w, h, conf = object_info[:5]

        # Convert to pixel coordinates
        x_center_px = int(x_center * img_w)
        y_center_px = int(y_center * img_h)
        w_px = int(w * img_w)
        h_px = int(h * img_h)

        # Calculate bbox corners
        x1 = int(x_center_px - w_px / 2)
        y1 = int(y_center_px - h_px / 2)
        x2 = int(x_center_px + w_px / 2)
        y2 = int(y_center_px + h_px / 2)

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"Object {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 10, label_size[1])
        cv2.rectangle(img, (x1, label_y - label_size[1] - 5), (x1 + label_size[0], label_y + 5), color, -1)
        cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw right hand (green)
    if right_hand_info is not None:
        draw_hand(right_hand_info, RIGHT_HAND_COLOR, "Right Hand")

    # Draw left hand (red)
    if left_hand_info is not None:
        draw_hand(left_hand_info, LEFT_HAND_COLOR, "Left Hand")

    # Draw objects (blue)
    if object_infos is not None:
        for obj_info in object_infos:
            draw_object(obj_info, OBJECT_COLOR)

    return img


def detect_track(hand_det_model, imgfiles, conf=0.3, tracker='configs/bytetrack.yaml'):
    # Run
    right_hand_tracks = {}
    left_hand_tracks = {}
    object_tracks = {}

    for t, imgpath in enumerate(imgfiles):
        img_cv2 = cv2.imread(imgpath)

        # --- Detection ---
        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device):
                results = hand_det_model.track(img_cv2, conf=conf, tracker=tracker, persist=True, verbose=False)

                if results[0] is None:
                    continue

                boxes = results[0].boxes.xywhn.cpu().numpy()  # normalized to 0-1
                confs = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                keypoints = results[0].keypoints.xyn.cpu().numpy() if results[0].keypoints is not None else None
                keypoints_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints is not None else None

                # Find best detection for each class
                best_right_hand = None
                best_left_hand = None
                current_objects = []
                max_conf_right = 0.0
                max_conf_left = 0.0

                for idx, box in enumerate(boxes):
                    class_id = int(class_ids[idx])
                    confidence = confs[idx]

                    if class_id == 0:  # Left Hand
                        if confidence > max_conf_left:
                            max_conf_left = confidence
                            best_left_hand = (box, keypoints[idx], keypoints_conf[idx])
                    elif class_id == 1:  # Right Hand
                        if confidence > max_conf_right:
                            max_conf_right = confidence
                            best_right_hand = (box, keypoints[idx], keypoints_conf[idx])
                    elif class_id == 2:  # Object
                        current_objects.append(box.tolist() + [confidence])

                # Process and store best hand detections for the frame
                if best_left_hand:
                    box, kpts, kpts_conf = best_left_hand
                    kpts_flat = [coord for kpt, c in zip(kpts, kpts_conf) for coord in [kpt[0], kpt[1], c > 0.5]]
                    left_hand_tracks[t] = [0] + box.tolist() + kpts_flat

                if best_right_hand:
                    box, kpts, kpts_conf = best_right_hand
                    kpts_flat = [coord for kpt, c in zip(kpts, kpts_conf) for coord in [kpt[0], kpt[1], c > 0.5]]
                    right_hand_tracks[t] = [1] + box.tolist() + kpts_flat

                if current_objects:
                    object_tracks[t] = current_objects

    hand_det_model.predictor.trackers[0].reset()

    return right_hand_tracks, left_hand_tracks, object_tracks


def process_video_with_visualization(video_path, hand_det_model, output_base_folder, model_name, conf):
    """
    Process a single video, visualize tracking results, and save as video
    
    Args:
        video_path: video path
        hand_det_model: loaded YOLO model instance
        output_base_folder: temporary output folder
        model_name: name of the model (for output folder)
    """
    video_name = Path(video_path).stem

    # Create temporary frames folder
    temp_frames_folder = os.path.join(output_base_folder, f'temp_{video_name}_frames')
    os.makedirs(temp_frames_folder, exist_ok=True)

    try:
        # Extract frames
        imgfiles = process_video_to_frames(str(video_path), temp_frames_folder)

        # Perform hand detection and tracking
        right_hand_tracks, left_hand_tracks, object_tracks = detect_track(
            hand_det_model,
            imgfiles,
            conf=conf,
            tracker='configs/bytetrack.yaml',
        )

        # Count detected frames
        num_detected_frames = 0
        for frame_idx in range(len(imgfiles)):
            if frame_idx in right_hand_tracks or frame_idx in left_hand_tracks:
                num_detected_frames += 1

        # Determine if detection was successful
        is_successful = num_detected_frames >= 30

        # Create output video with visualization
        # Create output folder based on model name
        vis_output_folder = os.path.join('visualization_hoi_results', model_name)
        os.makedirs(vis_output_folder, exist_ok=True)

        # Create output video filename
        output_video_name = f"{video_name}.mp4"
        output_video_path = os.path.join(vis_output_folder, output_video_name)

        # Get video properties
        fps = 30  # Default fps

        # Collect all frames with visualization
        frames_to_write = []

        # Process each frame and collect for video writing
        for frame_idx, imgpath in enumerate(imgfiles):
            img = cv2.imread(imgpath)

            # Get tracking info for this frame
            right_hand_info = right_hand_tracks.get(frame_idx, None)
            left_hand_info = left_hand_tracks.get(frame_idx, None)
            object_infos = object_tracks.get(frame_idx, None)

            # Draw tracking results
            img_vis = draw_detections(img.copy(), right_hand_info, left_hand_info, object_infos)

            # Add frame info text
            info_text = f"Frame: {frame_idx}/{len(imgfiles)-1}"
            cv2.putText(img_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            # Convert BGR to RGB for imageio
            img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            frames_to_write.append(img_vis_rgb)

        # Write video using imageio
        imageio.mimsave(output_video_path, frames_to_write, fps=fps)

        return video_name, num_detected_frames, is_successful, output_video_path, True

    except Exception as e:
        print(f"Error processing video: {video_name} - {str(e)}")
        return video_name, 0, False, None, False

    finally:
        # Clean up temporary frames folder
        if os.path.exists(temp_frames_folder):
            shutil.rmtree(temp_frames_folder)


def process_video_worker(args):
    """
    Worker function for the multiprocessing pool. Loads model once per worker.
    """
    global _WORKER_MODELS
    video_path, hand_det_model_path, output_base_folder, model_name, conf = args

    # Load model if not already in the worker's cache
    if 'hand_det_model' not in _WORKER_MODELS:
        _WORKER_MODELS['hand_det_model'] = YOLO(hand_det_model_path)
    hand_det_model = _WORKER_MODELS['hand_det_model']

    video_name, num_frames, is_successful, output_video_path, success = process_video_with_visualization(video_path, hand_det_model, output_base_folder,
                                                                                                         model_name, conf)

    if success:
        return {'video_name': video_name, 'num_frames': num_frames, 'is_successful': is_successful, 'output_video_path': output_video_path, 'success': True}
    else:
        return {'video_name': video_name, 'num_frames': 0, 'is_successful': False, 'output_video_path': None, 'success': False}


def worker_main(gpu_id, video_chunk, args, output_base_folder, model_name):
    """
    Main function for a worker process assigned to a specific GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker process for GPU {gpu_id} started, processing {len(video_chunk)} videos.")

    # Prepare arguments for each video in this GPU's chunk
    video_args_chunk = [(str(video), args.model, output_base_folder, model_name, args.conf) for video in video_chunk]

    results = []
    # Use a multiprocessing pool to process videos on this GPU
    with mp.Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_video_worker, video_args_chunk), total=len(video_args_chunk), desc=f"GPU {gpu_id}", position=gpu_id):
            results.append(result)

    # Save results to a temporary file for later aggregation
    temp_results_file = os.path.join(output_base_folder, f'results_gpu_{gpu_id}.json')
    with open(temp_results_file, 'w') as f:
        json.dump(results, f)

    print(f"Worker for GPU {gpu_id} finished. Results saved to {temp_results_file}")


def sample_videos(num_samples=1000):
    """
    Randomly sample videos from the taste_rob dataset
    """
    # Define paths
    base_dir = Path("/mlp_vepfs/share/hpl/project/data/taste_rob")

    # Collect all video paths
    all_videos = []
    for hand_type in ['DoubleHand', 'SingleHand']:
        hand_dir = base_dir / hand_type
        if not hand_dir.exists():
            continue

        for scene_dir in hand_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            video_files = list(scene_dir.glob("*.mp4"))
            for video_path in video_files:
                all_videos.append(video_path)

    print(f"Found {len(all_videos)} videos")

    # Randomly sample videos
    num_samples = min(num_samples, len(all_videos))
    random.seed(43)
    sampled_video_paths = random.sample(all_videos, num_samples)
    print(f"Successfully sampled {num_samples} videos from the taste_rob dataset")
    return sampled_video_paths


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test hand detection on good and bad case videos')
    parser.add_argument('--num_samples', type=str, default='3000', help='Number of videos to sample per category, or "all" to process all videos (default: 10)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers PER GPU (default: 4)')
    parser.add_argument('--model', type=str, default='models/tasterob_hoi_detector.pt', help='Path to the hand and object detection model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection (default: 0.5)')
    parser.add_argument('--visualize', action='store_true', help='Save visualization videos for test cases')
    args = parser.parse_args()

    # Parse num_samples
    if args.num_samples.lower() == 'all':
        num_samples = 'all'
    else:
        try:
            num_samples = int(args.num_samples)
        except ValueError:
            print(f"Error: --num_samples must be a number or 'all', got '{args.num_samples}'")
            return

    # Set multiprocessing start method to spawn (required for CUDA)
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set, skip

    # Set random seed to ensure reproducibility
    # random.seed(42)

    # Sample videos for testing
    print("=" * 80)
    print("Start sampling test videos...")
    print("=" * 80)
    videos_to_process = sample_videos(num_samples=num_samples)

    if not videos_to_process:
        print("No test videos found, please check the dataset path in sample_videos function!")
        return

    output_base_folder = 'output_test_hoi_cases'
    os.makedirs(output_base_folder, exist_ok=True)

    hand_det_model_path = args.model
    print(f"Using model: {hand_det_model_path}")

    # Extract model name from path for visualization output folder
    model_name = Path(hand_det_model_path).parent.parent.name if 'runs_hoi/' in hand_det_model_path else Path(hand_det_model_path).stem

    # --- Multi-GPU Processing Setup ---
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Warning: No GPUs found. Running on CPU, which might be very slow.")
        num_gpus = 1  # Treat as 1 device for CPU-only execution

    print(f"Found {num_gpus} devices. Distributing {len(videos_to_process)} videos...")

    if not args.visualize:
        print("\nProcessing without visualization is not supported in this version. Please add the --visualize flag.")
        return

    # Split videos into chunks for each GPU
    video_chunks = [[] for _ in range(num_gpus)]
    for i, video_file in enumerate(videos_to_process):
        video_chunks[i % num_gpus].append(video_file)

    processes = []
    for gpu_id in range(num_gpus):
        if not video_chunks[gpu_id]:
            continue  # Skip GPU if no videos are assigned
        p = mp.Process(target=worker_main, args=(gpu_id, video_chunks[gpu_id], args, output_base_folder, model_name))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Aggregate results from all temporary files
    print("\nAggregating results from all workers...")
    all_results = []
    for gpu_id in range(num_gpus):
        temp_file = os.path.join(output_base_folder, f'results_gpu_{gpu_id}.json')
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                all_results.extend(json.load(f))
            os.remove(temp_file)  # Clean up the temporary file

    results = all_results

    # Filter successful results
    successful_results = [r for r in results if r['success']]

    # Print individual results
    print("\n" + "=" * 80)
    print("Test Results:")
    print("=" * 80)
    for result in successful_results:
        status = "OK" if result['is_successful'] else "Low detections"
        base_info = f"  Video: {result['video_name']}, detected frames: {result['num_frames']}, status: {status}"
        if args.visualize and 'output_video_path' in result and result['output_video_path']:
            print(f"{base_info}, visualization saved to: {result['output_video_path']}")
        else:
            print(base_info)

    # Statistics results
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)

    total_videos = len(successful_results)
    success_count = sum(1 for r in successful_results if r['is_successful'])
    success_rate = (success_count / total_videos * 100) if total_videos > 0 else 0

    print(f"  Total videos processed: {total_videos}")
    print(f"  Videos with sufficient detections (>= 30 frames): {success_count}")
    print(f"  Videos with low detections (< 30 frames): {total_videos - success_count}")
    print(f"  Success rate: {success_rate:.2f}%")

    # Save detailed results to log file
    log_file_path = f'test_results_{model_name}.log'
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Test results for model: {model_name}\n")
        f.write("=" * 80 + "\n\n")

        for result in successful_results:
            status = "OK" if result['is_successful'] else "Low detections"
            base_info = f"Video: {result['video_name']}, detected frames: {result['num_frames']}, status: {status}"
            if args.visualize and 'output_video_path' in result and result['output_video_path']:
                f.write(f"{base_info}, visualization saved to: {result['output_video_path']}\n")
            else:
                f.write(f"{base_info}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Statistics\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Total videos processed: {total_videos}\n")
        f.write(f"  Videos with sufficient detections (>= 30 frames): {success_count}\n")
        f.write(f"  Videos with low detections (< 30 frames): {total_videos - success_count}\n")
        f.write(f"  Success rate: {success_rate:.2f}%\n")

    print(f"\nDetailed results saved to: {log_file_path}")

    if args.visualize:
        vis_folder = os.path.join('visualization_hoi_results', model_name)
        print(f"Visualization videos saved to: {vis_folder}")

    print("=" * 80)


if __name__ == "__main__":
    main()

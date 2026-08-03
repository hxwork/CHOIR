import os
import random
import shutil
from functools import partial
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from tqdm.contrib.concurrent import process_map
from ultralytics import YOLO

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


def draw_hand_tracking(img, right_hand_info=None, left_hand_info=None):
    """
    Draw hand tracking results on image
    
    Args:
        img: input image (numpy array)
        right_hand_info: right hand info [handedness, x_center, y_center, w, h, kpt1_x, kpt1_y, kpt1_conf, ...]
        left_hand_info: left hand info [handedness, x_center, y_center, w, h, kpt1_x, kpt1_y, kpt1_conf, ...]
    
    Returns:
        img: image with tracking results drawn
    """
    img_h, img_w = img.shape[:2]

    # Define colors (BGR format)
    RIGHT_HAND_COLOR = (0, 255, 0)  # Green for right hand
    LEFT_HAND_COLOR = (0, 0, 255)  # Red for left hand
    KEYPOINT_COLOR = (255, 255, 0)  # Cyan for keypoints

    def draw_hand(hand_info, color, hand_name):
        if hand_info is None or len(hand_info) < 5:
            return

        # Extract bbox info (normalized)
        handedness = hand_info[0]
        x_center, y_center, w, h = hand_info[1:5]

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

    # Draw right hand (green)
    if right_hand_info is not None:
        draw_hand(right_hand_info, RIGHT_HAND_COLOR, "Right Hand")

    # Draw left hand (red)
    if left_hand_info is not None:
        draw_hand(left_hand_info, LEFT_HAND_COLOR, "Left Hand")

    return img


def detect_track(hand_det_model, imgfiles, conf=0.5, conf_threshold=0.7, tracker='configs/bytetrack.yaml'):
    # Run
    boxes_ = []
    tracks = {}
    right_hand_tracks = {}
    left_hand_tracks = {}
    for t, imgpath in enumerate(imgfiles):
        img_cv2 = cv2.imread(imgpath)

        ### --- Detection ---
        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device):
                results = hand_det_model.track(img_cv2, conf=conf, tracker=tracker, persist=True, verbose=False)

                boxes = results[0].boxes.xywhn.cpu().numpy()  # normalized to 0-1
                confs = results[0].boxes.conf.cpu().numpy()
                handedness = results[0].boxes.cls.cpu().numpy()
                keypoints = results[0].keypoints.xyn.cpu().numpy()  # normalized to 0-1
                keypoints_conf = results[0].keypoints.conf.cpu().numpy()
                if not results[0].boxes.id is None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)

                # boxes = np.hstack([boxes, confs[:, None]])
                find_right = False
                find_left = False
                for idx, box in enumerate(boxes):
                    # only process high-confidence detections
                    if confs[[idx]][0] < conf_threshold:
                        continue
                    if track_id[idx] == -1:
                        if handedness[[idx]] > 0:
                            id = int(10000)
                        else:
                            id = int(5000)
                    else:
                        id = track_id[idx]
                    subj = dict()
                    subj['frame'] = t
                    subj['det'] = True
                    subj['det_box'] = boxes[[idx]]
                    subj['det_handedness'] = handedness[[idx]]

                    if (not find_right and handedness[[idx]] > 0) or (not find_left and handedness[[idx]] == 0):
                        cur_keypoints = keypoints[[idx]][0].tolist()
                        cur_keypoints_conf = (keypoints_conf[[idx]][0] > 0.5).astype(int).tolist()
                        cur_keypoints_flat = [coord for kpt, c in zip(cur_keypoints, cur_keypoints_conf) for coord in [kpt[0], kpt[1], c]]
                        x_coords = np.array(cur_keypoints)[:, 0]
                        y_coords = np.array(cur_keypoints)[:, 1]
                        x_min, x_max = x_coords.min(), x_coords.max()
                        y_min, y_max = y_coords.min(), y_coords.max()
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        scale = 1.2
                        cur_bbox = [x_center, y_center, width * scale, height * scale]
                        full_info = handedness[[idx]].tolist() + cur_bbox + cur_keypoints_flat
                        if (id in tracks) and (subj['det_handedness'][0] == tracks[id][-1]['det_handedness'][0]):
                            # make sure the handness is the same
                            tracks[id].append(subj)

                            if handedness[[idx]] > 0:
                                right_hand_tracks[t] = full_info
                                find_right = True
                            elif handedness[[idx]] == 0:
                                left_hand_tracks[t] = full_info
                                find_left = True
                        elif (id not in tracks):
                            tracks[id] = [subj]

                            if handedness[[idx]] > 0:
                                right_hand_tracks[t] = full_info
                                find_right = True
                            elif handedness[[idx]] == 0:
                                left_hand_tracks[t] = full_info
                                find_left = True
                        else:
                            continue
    hand_det_model.predictor.trackers[0].reset()

    return tracks, right_hand_tracks, left_hand_tracks


def save_detection_results(imgfiles, right_hand_tracks, left_hand_tracks, video_name, output_path='./data', fps=30, train_val_ratio=0.9):
    # collect high-confidence frames
    high_conf_frames = {}

    for frame_idx in range(len(imgfiles)):
        frame_data = []

        # check right hand
        if frame_idx in right_hand_tracks:
            full_info = right_hand_tracks[frame_idx]
            frame_data.append(('right', full_info))

        # check left hand
        if frame_idx in left_hand_tracks:
            full_info = left_hand_tracks[frame_idx]
            frame_data.append(('left', full_info))

        if frame_data:
            high_conf_frames[frame_idx] = frame_data

    # check if the video has low detection rate (< 30 frames)
    if len(high_conf_frames) < 30:

        # calculate the frame range to remove the first and last 1.5s
        skip_frames = int(1.5 * fps)  # 1.5 seconds corresponds to frames
        start_frame = skip_frames
        end_frame = len(imgfiles) - skip_frames

        # save all frames (remove the first and last 1.5s) to test
        for frame_idx in range(start_frame, end_frame):
            imgpath = imgfiles[frame_idx]
            # only copy images, no labels
            shutil.copy(imgpath, os.path.join(output_path, 'images', 'test', f'{video_name}_{frame_idx:06d}.jpg'))

        return f"Video {video_name} has less than 30 high-confidence detections ({len(high_conf_frames)} frames), saving to test set..."

    # enough detections, split into train/val

    # get all frame indices and sort
    frame_indices = sorted(high_conf_frames.keys())

    # split by 9:1 ratio
    split_idx = int(len(frame_indices) * train_val_ratio)
    train_frames = set(frame_indices[:split_idx])
    val_frames = set(frame_indices[split_idx:])

    # save train results
    for frame_idx in train_frames:
        imgpath = imgfiles[frame_idx]
        frame_data = high_conf_frames[frame_idx]
        labels = []

        for hand_type, full_info in frame_data:
            labels.append(full_info)

        label_file = os.path.join(output_path, 'labels', 'train', f'{video_name}_{frame_idx:06d}.txt')

        with open(label_file, 'w') as f:
            for label in labels:
                f.write(f"{' '.join(str(x) for x in label)}\n")

        shutil.copy(imgpath, os.path.join(output_path, 'images', 'train', f'{video_name}_{frame_idx:06d}.jpg'))

    # save val data
    for frame_idx in val_frames:
        imgpath = imgfiles[frame_idx]
        frame_data = high_conf_frames[frame_idx]

        labels = []
        for hand_type, full_info in frame_data:
            labels.append(full_info)

        label_file = os.path.join(output_path, 'labels', 'val', f'{video_name}_{frame_idx:06d}.txt')
        with open(label_file, 'w') as f:
            for label in labels:
                f.write(f"{' '.join(str(x) for x in label)}\n")

        shutil.copy(imgpath, os.path.join(output_path, 'images', 'val', f'{video_name}_{frame_idx:06d}.jpg'))

    return f"Successfully saved {len(train_frames)} frames to train, {len(val_frames)} frames to val detection results for video: {video_name}!"


def sample_videos_from_test_cases(num_samples_per_category='all'):
    """
    Sample videos from the good and bad subdirectories in the test_cases directory
    
    Args:
        num_samples_per_category: number of videos sampled per category, or 'all' to process all videos
    
    Returns:
        good_videos: good cases video paths list
        bad_videos: bad cases video paths list
    """
    # Define test_cases path
    test_cases_dir = Path("test_cases")

    good_dir = test_cases_dir / "good_cases"
    bad_dir = test_cases_dir / "bad_cases"

    # Collect good cases videos
    good_videos = []
    if good_dir.exists():
        good_videos = list(good_dir.glob("*.mp4"))
        print(f"Found {len(good_videos)} videos in good cases")
    else:
        print(f"Warning: {good_dir} does not exist!")

    # Collect bad cases videos
    bad_videos = []
    if bad_dir.exists():
        bad_videos = list(bad_dir.glob("*.mp4"))
        print(f"Found {len(bad_videos)} videos in bad cases")
    else:
        print(f"Warning: {bad_dir} does not exist!")

    # Randomly sample or use all videos
    if num_samples_per_category == 'all':
        sampled_good_videos = good_videos
        sampled_bad_videos = bad_videos
        print(f"Using all {len(sampled_good_videos)} videos from good cases")
        print(f"Using all {len(sampled_bad_videos)} videos from bad cases")
    else:
        num_good_samples = min(num_samples_per_category, len(good_videos))
        num_bad_samples = min(num_samples_per_category, len(bad_videos))

        sampled_good_videos = random.sample(good_videos, num_good_samples) if good_videos else []
        sampled_bad_videos = random.sample(bad_videos, num_bad_samples) if bad_videos else []

        print(f"Sampled {num_good_samples} videos from good cases")
        print(f"Sampled {num_bad_samples} videos from bad cases")

    return sampled_good_videos, sampled_bad_videos


def process_single_video_for_test(video_path, hand_det_model_path, output_base_folder):
    """
    Process a single video and return the number of detected frames
    
    Args:
        video_path: video path
        hand_det_model_path: hand detection model path
        output_base_folder: temporary output folder
    
    Returns:
        video_name: video name
        num_detected_frames: number of detected high-confidence frames
        success: whether the video is successfully processed
    """
    # Load model
    hand_det_model = YOLO(hand_det_model_path)

    video_name = video_path.stem

    # Create temporary frames folder
    temp_frames_folder = os.path.join(output_base_folder, f'temp_{video_name}_frames')
    os.makedirs(temp_frames_folder, exist_ok=True)

    try:
        # Extract frames
        imgfiles = process_video_to_frames(str(video_path), temp_frames_folder)

        # Perform hand detection and tracking
        tracks, right_hand_tracks, left_hand_tracks = detect_track(
            hand_det_model,
            imgfiles,
            conf=0.6,
            conf_threshold=0.7,
            tracker='configs/bytetrack.yaml',
        )

        # Count detected frames
        num_detected_frames = 0
        for frame_idx in range(len(imgfiles)):
            if frame_idx in right_hand_tracks or frame_idx in left_hand_tracks:
                num_detected_frames += 1

        return video_name, num_detected_frames, True

    except Exception as e:
        print(f"Error processing video: {video_name} - {str(e)}")
        return video_name, 0, False

    finally:
        # Clean up temporary frames folder
        if os.path.exists(temp_frames_folder):
            shutil.rmtree(temp_frames_folder)


def process_video_with_visualization(video_path, hand_det_model_path, output_base_folder, model_name):
    """
    Process a single video, visualize tracking results, and save as video
    
    Args:
        video_path: video path
        hand_det_model_path: hand detection model path
        output_base_folder: temporary output folder
        model_name: name of the model (for output folder)
    
    Returns:
        video_name: video name
        num_detected_frames: number of detected high-confidence frames
        is_successful: whether detection was successful (>= 30 frames)
        output_video_path: path to the saved visualization video
        success: whether the video is successfully processed
    """
    # Load model
    hand_det_model = YOLO(hand_det_model_path)

    video_name = video_path.stem

    # Create temporary frames folder
    temp_frames_folder = os.path.join(output_base_folder, f'temp_{video_name}_frames')
    os.makedirs(temp_frames_folder, exist_ok=True)

    try:
        # Extract frames
        imgfiles = process_video_to_frames(str(video_path), temp_frames_folder)

        # Perform hand detection and tracking
        tracks, right_hand_tracks, left_hand_tracks = detect_track(
            hand_det_model,
            imgfiles,
            conf=0.6,
            conf_threshold=0.7,
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
        vis_output_folder = os.path.join('visualization_results', model_name)
        os.makedirs(vis_output_folder, exist_ok=True)

        # Create output video filename with success/fail flag
        success_flag = "success" if is_successful else "fail"
        output_video_name = f"{video_name}_{success_flag}.mp4"
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

            # Draw tracking results
            img_vis = draw_hand_tracking(img.copy(), right_hand_info, left_hand_info)

            # Add frame info text
            info_text = f"Frame: {frame_idx}/{len(imgfiles)-1} | Detected frames: {num_detected_frames} | Status: {success_flag.upper()}"
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


def process_video_wrapper(args):
    """
    Wrapper function for multiprocessing
    
    Args:
        args: tuple of (video_path, hand_det_model_path, output_base_folder, case_type)
    
    Returns:
        dict with video_name, num_frames, is_successful, case_type
    """
    video_path, hand_det_model_path, output_base_folder, case_type = args
    video_name, num_frames, success = process_single_video_for_test(video_path, hand_det_model_path, output_base_folder)

    if success:
        is_successful = num_frames >= 30
        return {'video_name': video_name, 'num_frames': num_frames, 'is_successful': is_successful, 'case_type': case_type, 'success': True}
    else:
        return {'video_name': video_name, 'num_frames': 0, 'is_successful': False, 'case_type': case_type, 'success': False}


def process_video_wrapper_with_vis(args):
    """
    Wrapper function for multiprocessing with visualization
    
    Args:
        args: tuple of (video_path, hand_det_model_path, output_base_folder, case_type, model_name)
    
    Returns:
        dict with video_name, num_frames, is_successful, case_type, output_video_path
    """
    video_path, hand_det_model_path, output_base_folder, case_type, model_name = args
    video_name, num_frames, is_successful, output_video_path, success = process_video_with_visualization(video_path, hand_det_model_path, output_base_folder,
                                                                                                         model_name)

    if success:
        return {
            'video_name': video_name,
            'num_frames': num_frames,
            'is_successful': is_successful,
            'case_type': case_type,
            'output_video_path': output_video_path,
            'success': True
        }
    else:
        return {'video_name': video_name, 'num_frames': 0, 'is_successful': False, 'case_type': case_type, 'output_video_path': None, 'success': False}


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test hand detection on good and bad case videos')
    parser.add_argument('--num_samples', type=str, default='10', help='Number of videos to sample per category, or "all" to process all videos (default: 10)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--model',
                        type=str,
                        default='runs/fine_tune5/weights/last.pt',
                        help='Path to the hand detection model (default: runs/fine_tune5/weights/last.pt)')
    parser.add_argument('--visualize_bad_cases', action='store_true', help='Save visualization videos for bad cases')
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
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set, skip

    # Set random seed to ensure reproducibility
    random.seed(42)

    # Sample good and bad cases videos
    print("=" * 80)
    print("Start sampling test videos...")
    print("=" * 80)
    good_videos, bad_videos = sample_videos_from_test_cases(num_samples_per_category=num_samples)

    if not good_videos and not bad_videos:
        print("No test videos found, please check if the test_cases directory exists!")
        return

    output_base_folder = 'output_test_cases'
    os.makedirs(output_base_folder, exist_ok=True)

    hand_det_model_path = args.model
    print(f"Using model: {hand_det_model_path}")

    # Extract model name from path for visualization output folder
    model_name = Path(hand_det_model_path).parent.parent.name if 'runs/' in hand_det_model_path else Path(hand_det_model_path).stem

    # Prepare arguments for multiprocessing
    # Number of workers (adjust based on GPU memory)
    num_workers = args.num_workers
    print(f"Using {num_workers} parallel workers")

    # Process good cases and bad cases
    if args.visualize_bad_cases:
        # # Process good cases without visualization
        # print(f"\nProcessing {len(good_videos)} good case videos...")
        # print("=" * 80)
        # good_video_args = [(video, hand_det_model_path, output_base_folder, 'good') for video in good_videos]
        # good_results_raw = process_map(process_video_wrapper, good_video_args, max_workers=num_workers, desc="Processing good cases", unit="video", chunksize=1)

        # Process bad cases with visualization
        print(f"\nProcessing {len(bad_videos)} bad case videos with visualization...")
        print("=" * 80)
        bad_video_args = [(video, hand_det_model_path, output_base_folder, 'bad', model_name) for video in bad_videos]
        bad_results_raw = process_map(process_video_wrapper_with_vis,
                                      bad_video_args,
                                      max_workers=num_workers,
                                      desc="Processing bad cases with visualization",
                                      unit="video",
                                      chunksize=1)

        # Combine results
        results = bad_results_raw
    else:
        # Process all videos without visualization
        all_video_args = []
        for video in good_videos:
            all_video_args.append((video, hand_det_model_path, output_base_folder, 'good'))
        for video in bad_videos:
            all_video_args.append((video, hand_det_model_path, output_base_folder, 'bad'))

        print(f"\nProcessing {len(all_video_args)} videos using {num_workers} workers...")
        print("=" * 80)

        # Use process_map for parallel processing with progress bar
        results = process_map(process_video_wrapper, all_video_args, max_workers=num_workers, desc="Processing videos", unit="video", chunksize=1)

    # Separate results by case type
    good_results = []
    bad_results = []

    for result in results:
        if result['success']:
            if result['case_type'] == 'good':
                good_results.append(result)
            else:
                bad_results.append(result)

    # Print individual results
    print("\n" + "=" * 80)
    print("Good Cases Results:")
    print("=" * 80)
    for result in good_results:
        status = "Successfully detected" if result['is_successful'] else "Not successfully detected"
        print(f"  Video: {result['video_name']}, detected frames: {result['num_frames']}, status: {status}")

    print("\n" + "=" * 80)
    print("Bad Cases Results:")
    print("=" * 80)
    for result in bad_results:
        status = "Successfully detected" if result['is_successful'] else "Not successfully detected"
        base_info = f"  Video: {result['video_name']}, detected frames: {result['num_frames']}, status: {status}"
        if args.visualize_bad_cases and 'output_video_path' in result and result['output_video_path']:
            print(f"{base_info}, saved to: {result['output_video_path']}")
        else:
            print(base_info)

    # Statistics results
    print("\n" + "=" * 80)
    print("Statistics results")
    print("=" * 80)

    # Good cases statistics
    good_success_count = sum(1 for r in good_results if r['is_successful'])
    good_total = len(good_results)
    good_success_rate = (good_success_count / good_total * 100) if good_total > 0 else 0

    print(f"\nGood Cases:")
    print(f"  Total videos: {good_total}")
    print(f"  Successfully detected: {good_success_count}")
    print(f"  Not successfully detected: {good_total - good_success_count}")
    print(f"  Detection success rate: {good_success_rate:.2f}%")

    # Bad cases statistics
    bad_success_count = sum(1 for r in bad_results if r['is_successful'])
    bad_total = len(bad_results)
    bad_success_rate = (bad_success_count / bad_total * 100) if bad_total > 0 else 0

    print(f"\nBad Cases:")
    print(f"  Total videos: {bad_total}")
    print(f"  Successfully detected: {bad_success_count}")
    print(f"  Not successfully detected: {bad_total - bad_success_count}")
    print(f"  Detection success rate: {bad_success_rate:.2f}%")

    # Save detailed results to log file
    with open('test_cases_results.log', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Test results details\n")
        f.write("=" * 80 + "\n\n")

        f.write("Good Cases detailed results:\n")
        f.write("-" * 80 + "\n")
        for result in good_results:
            status = "Successfully detected" if result['is_successful'] else "Not successfully detected"
            f.write(f"Video: {result['video_name']}, detected frames: {result['num_frames']}, status: {status}\n")

        f.write(f"\nGood Cases statistics:\n")
        f.write(f"  Total videos: {good_total}\n")
        f.write(f"  Successfully detected: {good_success_count}\n")
        f.write(f"  Not successfully detected: {good_total - good_success_count}\n")
        f.write(f"  Detection success rate: {good_success_rate:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("Bad Cases detailed results:\n")
        f.write("-" * 80 + "\n")
        for result in bad_results:
            status = "Successfully detected" if result['is_successful'] else "Not successfully detected"
            base_info = f"Video: {result['video_name']}, detected frames: {result['num_frames']}, status: {status}"
            if args.visualize_bad_cases and 'output_video_path' in result and result['output_video_path']:
                f.write(f"{base_info}, saved to: {result['output_video_path']}\n")
            else:
                f.write(f"{base_info}\n")

        f.write(f"\nBad Cases statistics:\n")
        f.write(f"  Total videos: {bad_total}\n")
        f.write(f"  Successfully detected: {bad_success_count}\n")
        f.write(f"  Not successfully detected: {bad_total - bad_success_count}\n")
        f.write(f"  Detection success rate: {bad_success_rate:.2f}%\n")

    print(f"\nDetailed results saved to: test_cases_results.log")

    if args.visualize_bad_cases:
        vis_folder = os.path.join('visualization_results', model_name)
        print(f"Visualization videos for bad cases saved to: {vis_folder}")

    print("=" * 80)


if __name__ == "__main__":
    main()

import os
import random
import shutil
from functools import partial
from pathlib import Path

import cv2
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


def sample_videos(num_samples=1000):
    """
    Randomly sample videos from the taste_rob dataset
    """
    # Define paths
    base_dir = Path("data/taste_rob")

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
    sampled_video_paths = random.sample(all_videos, num_samples)
    print(f"Successfully sampled {num_samples} videos from the taste_rob dataset")
    return sampled_video_paths


def process_single_video(video_path, hand_det_model_path, output_data_folder, output_base_folder):
    """
    Process a single video
    """
    # load model
    hand_det_model = YOLO(hand_det_model_path)

    video_name = video_path.stem

    # create temporary frames folder
    temp_frames_folder = os.path.join(output_base_folder, f'temp_{video_name}_frames')
    os.makedirs(temp_frames_folder, exist_ok=True)

    try:
        # extract frames
        imgfiles = process_video_to_frames(str(video_path), temp_frames_folder)

        # perform hand detection and tracking
        tracks, right_hand_tracks, left_hand_tracks = detect_track(
            hand_det_model,
            imgfiles,
            conf=0.6,
            conf_threshold=0.7,
            tracker='configs/bytetrack.yaml',
        )

        # save results
        print_info = save_detection_results(
            imgfiles,
            right_hand_tracks,
            left_hand_tracks,
            video_name,
            output_path=output_data_folder,
            fps=30,
            train_val_ratio=0.9,
        )

        return print_info

    except Exception as e:
        return f"Error processing video: {video_name} - {str(e)}"

    finally:
        # clean up temporary frames folder
        if os.path.exists(temp_frames_folder):
            shutil.rmtree(temp_frames_folder)


def main():
    # set multiprocessing start method to spawn (CUDA required)
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # if already set, skip

    # sample videos
    video_files = sample_videos(num_samples=30000)

    output_base_folder = 'output_tracks'
    output_data_folder = './data'

    # make output base folder
    os.makedirs(output_base_folder, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_data_folder, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_data_folder, 'labels', split), exist_ok=True)

    # for vi, video_path in enumerate(video_files):
    #     video_name = video_path.stem  # get video name (without extension)
    #     print(f"\nProcessing video: {video_name}")

    #     # create temp frames folder for each video
    #     temp_frames_folder = os.path.join('output_tracks', f'temp_{video_name}_frames')
    #     os.makedirs(temp_frames_folder, exist_ok=True)

    #     # extract frames from video
    #     print(f"  extract frames from video...")
    #     imgfiles = process_video_to_frames(str(video_path), temp_frames_folder)

    #     # perform hand detection and tracking
    #     print(f"  perform hand detection and tracking...")
    #     tracks, right_hand_tracks, left_hand_tracks = detect_track(
    #         hand_det_model,
    #         imgfiles,
    #         conf=0.6,
    #         conf_threshold=0.7,
    #         tracker='configs/bytetrack.yaml',
    #     )

    #     save_detection_results(
    #         imgfiles,
    #         right_hand_tracks,
    #         left_hand_tracks,
    #         video_name,
    #         output_path=output_data_folder,
    #         fps=30,
    #         train_val_ratio=0.9,
    #     )

    #     # clean up temp frames folder
    #     print(f"  clean up temp frames folder...")
    #     shutil.rmtree(temp_frames_folder)

    #     print(f"  video {video_name} processed!")

    # use multiprocessing to process videos
    num_workers = 6  # run 5 processes on one GPU

    # create partial function, fix some parameters
    process_func = partial(
        process_single_video,
        hand_det_model_path='models/wilor_hand_detector.pt',
        output_data_folder=output_data_folder,
        output_base_folder=output_base_folder,
    )

    # use process pool to process videos
    results = process_map(
        process_func,
        video_files,
        max_workers=num_workers,
        desc="Processing videos",  # description of progress bar
        unit="video"  # unit of progress bar
    )

    with open('prepare_data.log', 'w') as f:
        for result in results:
            f.write(f"{result}\n")
    print('Save log file to prepare_data.log!')


if __name__ == "__main__":
    main()

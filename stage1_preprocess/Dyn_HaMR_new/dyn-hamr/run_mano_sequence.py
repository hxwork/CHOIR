#!/usr/bin/env python3
"""
Batch-run Dyn-HaMR optimization across one or more GPUs.

Reads CHOIR per-video folders under output/<video_id>/inputs/video.mp4 and writes
Hydra results to output/<video_id>/stage1/dynhamr/ (see confs/config.yaml).
"""
import argparse
import logging
import multiprocessing as mp
import os
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from tqdm.contrib.concurrent import process_map

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CHOIR repo root: stage1_preprocess/Dyn_HaMR_new/dyn-hamr/run_mano_sequence.py -> ../../..
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = str(REPO_ROOT / "output")


def get_available_gpus():
    """Return the number of visible CUDA devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def process_video(video_path, gpu_id=0, is_static=True, data_root=None, error_log_file=None):
    """
    Process a single video file.

    Args:
        video_path: Path to output/<video_id>/<video_id>.mp4
        gpu_id: GPU device ID to use
        is_static: Whether to use static camera mode
        data_root: CHOIR output root (default: REPO_ROOT/output)
        error_log_file: Path to error log file for real-time logging

    Returns:
        tuple: (video_name, success, error_message)
    """
    video_name = Path(video_path).parents[1].name  # .../<video_id>/inputs/video.mp4
    if data_root is None:
        data_root = DEFAULT_OUTPUT_ROOT

    # Hydra writes to ${data.root}/${data.video_dir}/stage1/dynhamr
    module_root = Path(__file__).resolve().parents[1]  # Dyn_HaMR_new/
    vipe_root = module_root / "third-party" / "vipe"
    python_args = [
        'run_opt.py',
        'data=video_vipe',
        'run_opt=True',
        f'data.root={data_root}',
        f'data.video_dir={video_name}',
        f'data.seq={video_name}',
        f'data.vipe_root={vipe_root}',
        f'is_static={is_static}',
    ]

    cmd = f"export CUDA_VISIBLE_DEVICES={gpu_id} && python {' '.join(python_args)}"
    logger.info(f"Running command: {cmd}")

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
            shell=True,
            executable='/bin/bash',
        )

        if result.returncode == 0:
            logger.info(f"Successfully processed: {video_name}")
            return video_name, True, None

        error_msg = result.stderr if result.stderr else result.stdout
        logger.error(f"Failed to process: {video_name}")
        logger.error(f"  Error: {error_msg[:200]}")
        if error_log_file:
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Video: {video_name}\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write(f"Error:\n{error_msg}\n")
        return video_name, False, error_msg

    except subprocess.TimeoutExpired:
        error_msg = "Timeout (>1 hour)"
        logger.error(f"Timeout: {video_name}")
        if error_log_file:
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Video: {video_name}\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write(f"Error: {error_msg}\n")
        return video_name, False, error_msg

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Exception: {video_name} - {error_msg}")
        if error_log_file:
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Video: {video_name}\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write(f"Exception: {error_msg}\n")
        return video_name, False, error_msg


def process_videos_on_gpu(gpu_id, video_files, num_workers, is_static, data_root, error_log_file=None):
    """Process a batch of videos on a specific GPU."""
    logger.info(f"GPU {gpu_id}: Processing {len(video_files)} videos with {num_workers} workers")
    process_func = partial(
        process_video,
        gpu_id=gpu_id,
        is_static=is_static,
        data_root=data_root,
        error_log_file=error_log_file,
    )
    return process_map(
        process_func,
        video_files,
        max_workers=num_workers,
        desc=f"GPU {gpu_id}",
        unit="video",
        chunksize=1,
        position=gpu_id,
    )


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='Batch Dyn-HaMR optimization with multi-GPU support')
    parser.add_argument(
        '--video_dir',
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help='CHOIR output root containing <video_id>/inputs/video.mp4 (default: repo output/).',
    )
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes per GPU')
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU IDs (e.g., "0,1,2"). Default: all GPUs')
    parser.add_argument('--is_static', action='store_true', default=True, help='Use static camera mode')
    parser.add_argument('--video_id', type=str, nargs='+', default=None, help='Video ID(s) to process')
    parser.add_argument('--total_parts', type=int, default=1, help='Split videos across machines')
    parser.add_argument('--part_id', type=int, default=0, help='Which part to process (0-indexed)')
    args = parser.parse_args()

    data_root = os.path.abspath(args.video_dir)

    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        num_gpus = get_available_gpus()
        if num_gpus == 0:
            logger.error("No GPU available!")
            return
        gpu_ids = list(range(num_gpus))

    logger.info(f"Detected/Using GPUs: {gpu_ids}")
    logger.info(f"Data / output root: {data_root}")

    video_files = []
    subdirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    for video_id in subdirs:
        video_path = os.path.join(data_root, video_id, "inputs", "video.mp4")
        if os.path.exists(video_path):
            video_files.append(video_path)

    if args.video_id is not None:
        video_files = [f for f in video_files if Path(f).parents[1].name in args.video_id]

    if args.total_parts > 1:
        if args.part_id < 0 or args.part_id >= args.total_parts:
            logger.error(f"Invalid part_id {args.part_id}. Must be in range [0, {args.total_parts-1}]")
            return
        total_videos = len(video_files)
        video_files = [v for i, v in enumerate(video_files) if i % args.total_parts == args.part_id]
        logger.info(f"Part {args.part_id + 1}/{args.total_parts}: Selected {len(video_files)} out of {total_videos} videos")

    if not video_files:
        logger.error("No matching video files found")
        return

    num_gpus = len(gpu_ids)
    logger.info(f"Found {len(video_files)} video files to process")
    logger.info(f"Using {num_gpus} GPU(s), {args.num_workers} workers per GPU")
    logger.info(f"Total parallel processes: {num_gpus * args.num_workers}")
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep batch logs next to this entry script (not under output/<video_id>/).
    log_dir = Path(__file__).resolve().parent
    error_log_file = str(log_dir / f"batch_test_errors_{timestamp}.txt")
    with open(error_log_file, 'w', encoding='utf-8') as f:
        f.write(f"Batch Test Error Log - {datetime.now()}\n")
        f.write(f"Part {args.part_id + 1}/{args.total_parts}\n")
        f.write("=" * 60 + "\n")
    logger.info(f"Error log file: {error_log_file}")

    videos_per_gpu = [[] for _ in range(num_gpus)]
    for i, video_file in enumerate(video_files):
        videos_per_gpu[i % num_gpus].append(video_file)

    for gpu_id, videos in zip(gpu_ids, videos_per_gpu):
        logger.info(f"GPU {gpu_id}: {len(videos)} videos assigned")

    if num_gpus == 1:
        process_func = partial(
            process_video,
            gpu_id=gpu_ids[0],
            is_static=args.is_static,
            data_root=data_root,
            error_log_file=error_log_file,
        )
        results = process_map(
            process_func,
            video_files,
            max_workers=args.num_workers,
            desc="Processing videos",
            unit="video",
            chunksize=1,
        )
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id, videos in zip(gpu_ids, videos_per_gpu):
                if videos:
                    futures.append(
                        executor.submit(
                            process_videos_on_gpu,
                            gpu_id,
                            videos,
                            args.num_workers,
                            args.is_static,
                            data_root,
                            error_log_file,
                        )
                    )
            results = []
            for future in futures:
                results.extend(future.result())

    success_count = 0
    failed_videos = []
    for video_name, success, error_msg in results:
        if success:
            success_count += 1
        else:
            failed_videos.append((video_name, error_msg))

    logger.info("=" * 60)
    logger.info("Processing completed!")
    logger.info(f"Total: {len(video_files)} videos")
    logger.info(f"Success: {success_count} videos")
    logger.info(f"Failed: {len(failed_videos)} videos")

    if failed_videos:
        logger.info("\nFailed videos:")
        for video_name, error_msg in failed_videos:
            logger.info(f"  - {video_name}")
            if error_msg:
                logger.info(f"    Error: {error_msg[:100]}")

    result_file = str(log_dir / f"batch_test_results_{timestamp}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Batch test results - {datetime.now()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {len(video_files)} videos\n")
        f.write(f"Success: {success_count} videos\n")
        f.write(f"Failed: {len(failed_videos)} videos\n\n")
        if failed_videos:
            f.write("Failed videos:\n")
            for video_name, error_msg in failed_videos:
                f.write(f"\nVideo: {video_name}\n")
                f.write(f"Error: {error_msg}\n")

    logger.info(f"\nThe results have been saved to: {result_file}")


if __name__ == '__main__':
    main()

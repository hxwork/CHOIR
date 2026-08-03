import glob
import os
import random
import shutil
import subprocess
from pathlib import Path


def get_video_ids_from_images(image_dir):
    """extract all unique video_ids from image directory"""
    video_ids = set()
    if not os.path.exists(image_dir):
        print(f"Directory does not exist: {image_dir}")
        return video_ids

    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg'):
            # extract video_id before underscore
            video_id = img_file.split('_')[0]
            video_ids.add(video_id)

    return list(video_ids)


def find_video_path(video_id):
    """find the full path of the video file by video_id"""
    # use glob to find video file
    pattern = f"/mlp_vepfs/share/hpl/project/data/taste_rob/*/*/{video_id}.mp4"
    matches = glob.glob(pattern)

    if matches:
        return matches[0]  # return the first matching path
    else:
        return None


def main():
    # define paths
    base_dir = "/mlp_vepfs/share/hpl/project/code/Yolov8"
    test_images_dir = os.path.join(base_dir, "data/images/test")
    val_images_dir = os.path.join(base_dir, "data/images/val")

    good_cases_output = os.path.join(base_dir, "test_cases/good_cases")
    bad_cases_output = os.path.join(base_dir, "test_cases/bad_cases")

    # create output directories
    os.makedirs(good_cases_output, exist_ok=True)
    os.makedirs(bad_cases_output, exist_ok=True)

    print("Extracting bad case video_ids from test directory...")
    bad_video_ids = get_video_ids_from_images(test_images_dir)
    print(f"Found {len(bad_video_ids)} bad case video_ids")

    print("Extracting good case video_ids from val directory...")
    good_video_ids = get_video_ids_from_images(val_images_dir)
    print(f"Found {len(good_video_ids)} good case video_ids")

    # if more than 500, randomly sample
    max_samples = 500
    if len(bad_video_ids) > max_samples:
        print(f"Bad cases more than {max_samples}, randomly sampling...")
        bad_video_ids = random.sample(bad_video_ids, max_samples)

    if len(good_video_ids) > max_samples:
        print(f"Good cases more than {max_samples}, randomly sampling...")
        good_video_ids = random.sample(good_video_ids, max_samples)

    # process bad cases
    print(f"\nCopying {len(bad_video_ids)} bad case videos...")
    bad_success_count = 0
    bad_fail_count = 0

    for i, video_id in enumerate(bad_video_ids, 1):
        video_path = find_video_path(video_id)
        if video_path:
            dest_path = os.path.join(bad_cases_output, f"{video_id}.mp4")
            try:
                shutil.copy2(video_path, dest_path)
                print(f"[{i}/{len(bad_video_ids)}] Copied: {video_id}.mp4")
                bad_success_count += 1
            except Exception as e:
                print(f"[{i}/{len(bad_video_ids)}] Copy failed {video_id}: {e}")
                bad_fail_count += 1
        else:
            print(f"[{i}/{len(bad_video_ids)}] Video not found: {video_id}")
            bad_fail_count += 1

    # process good cases
    print(f"\nCopying {len(good_video_ids)} good case videos...")
    good_success_count = 0
    good_fail_count = 0

    for i, video_id in enumerate(good_video_ids, 1):
        video_path = find_video_path(video_id)
        if video_path:
            dest_path = os.path.join(good_cases_output, f"{video_id}.mp4")
            try:
                shutil.copy2(video_path, dest_path)
                print(f"[{i}/{len(good_video_ids)}] Copied: {video_id}.mp4")
                good_success_count += 1
            except Exception as e:
                print(f"[{i}/{len(good_video_ids)}] Copy failed {video_id}: {e}")
                good_fail_count += 1
        else:
            print(f"[{i}/{len(good_video_ids)}] Video not found: {video_id}")
            good_fail_count += 1

    # print summary
    print("\n" + "=" * 60)
    print("Copy completed!")
    print(f"Bad cases: {bad_success_count} success, {bad_fail_count} failed")
    print(f"Good cases: {good_success_count} success, {good_fail_count} failed")
    print(f"Bad cases output directory: {bad_cases_output}")
    print(f"Good cases output directory: {good_cases_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

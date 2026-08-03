import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh


def process_objects(video_ids=None):
    """
    处理 sam-3d-objects 数据集中的文件并存入新的 meshdata 目录:
    - obj_canonical.obj -> 加载，变换顶点 (x,y坐标取反)，另存为 decomposed.obj
    - obj_00000.json -> 提取 "scale" 键值存为 scale.json

    Args:
        video_ids: 指定要处理的 video id 列表，为 None 时处理所有目录
    """
    source_base_dir = Path("/vepfs_default/chanxueyan/lhp/xh/code/sam-3d-objects/input_data/")
    dest_base_dir = Path("meshdata/sam3d/")
    exclude_dirs = ["dynhamr", "images"]

    if not source_base_dir.is_dir():
        print(f"error: source base directory {source_base_dir} not found")
        return

    print(f"scanning {source_base_dir} for video directories...")

    # 获取所有需要处理的 video_id 目录
    if video_ids:
        video_dirs = []
        for vid in video_ids:
            d = source_base_dir / vid
            if d.is_dir():
                video_dirs.append(d)
            else:
                print(f"warning: video_id '{vid}' not found in {source_base_dir}, skipping")
    else:
        video_dirs = [d for d in source_base_dir.iterdir() if d.is_dir() and d.name not in exclude_dirs]

    processed_files_count = 0
    processed_dirs_count = 0
    not_found_dirs = []

    for video_dir in video_dirs:
        video_id = video_dir.name
        source_seq_dir = video_dir / "optimized_hoi_seq"

        source_obj_path = source_seq_dir / "obj_canonical.obj"
        source_json_path = source_seq_dir / "obj_00000.json"

        obj_exists = source_obj_path.is_file()
        json_exists = source_json_path.is_file()

        if not obj_exists and not json_exists:
            print(f"skip {video_id}: no target files found in {source_seq_dir}")
            not_found_dirs.append(video_id)
            continue

        # 创建目标目录
        dest_video_dir = dest_base_dir / video_id
        dest_video_dir.mkdir(parents=True, exist_ok=True)
        processed_dirs_count += 1

        # 处理 obj_canonical.obj
        if obj_exists:
            dest_obj_path = dest_video_dir / "decomposed.obj"
            print(f"loading, transforming, and saving: {source_obj_path} -> {dest_obj_path}")
            try:
                mesh = trimesh.load(source_obj_path, force='mesh', process=False)

                # 变换顶点
                mesh.vertices = mesh.vertices @ np.diag([-1, -1, 1])

                # 导出变换后的网格
                mesh.export(dest_obj_path)
                processed_files_count += 1
            except Exception as e:
                print(f"error processing obj file {source_obj_path}: {e}")

        # 处理 obj_00000.json
        if json_exists:
            try:
                with source_json_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'scale' in data:
                    scale_value = data['scale']
                    dest_scale_json_path = dest_video_dir / 'scale.json'

                    print(f"extracting scale from {source_json_path} -> {dest_scale_json_path}")

                    with dest_scale_json_path.open('w', encoding='utf-8') as f:
                        json.dump({'scale': scale_value}, f, indent=4)
                    processed_files_count += 1
                else:
                    print(f"warning: 'scale' key not found in {source_json_path}")

            except json.JSONDecodeError:
                print(f"error: could not decode JSON from {source_json_path}")
            except Exception as e:
                print(f"error processing {source_json_path}: {e}")

    print(f"\noperation completed.")
    print(f"processed {processed_files_count} files from {processed_dirs_count} directories.")
    if not_found_dirs:
        print(f"{len(not_found_dirs)} directories did not contain any target files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 sam-3d-objects 数据并存入 meshdata 目录")
    parser.add_argument(
        "--video_id",
        nargs="+",
        metavar="VIDEO_ID",
        default=None,
        help="指定要处理的 video id（可输入多个，空格分隔）。不指定则处理所有目录。",
    )
    args = parser.parse_args()
    process_objects(video_ids=args.video_id)

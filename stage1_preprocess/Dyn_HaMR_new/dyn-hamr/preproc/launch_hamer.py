import multiprocessing as mp
import os
import subprocess
from concurrent import futures
from pathlib import Path

from preproc.datasets import update_args
from preproc.export_hamer import export_sequence_results

ROOT_DIR = os.path.abspath(f"{__file__}/../../../")
SRC_DIR = os.path.join(ROOT_DIR, "third-party/hamer")
# CHOIR shares the Stage-1 WiLoR hand detector with Yolov8 (repo-relative).
DEFAULT_YOLO_MODEL = str(
    Path(ROOT_DIR).resolve().parent / "Yolov8" / "models" / "wilor_hand_detector.pt"
)


def launch_hamer(gpus, seq, img_dir, res_dir, name, datatype, overwrite=False):
    """
    run hamer using GPU pool
    """
    cur_proc = mp.current_process()
    print("PROCESS", cur_proc.name, cur_proc._identity)

    # Prefer CUDA_VISIBLE_DEVICES from the parent process when already set.
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is not None:
        print(f"Using CUDA_VISIBLE_DEVICES from environment: {cuda_visible}")
        gpu_setting = ""
    else:
        gpu = gpus[0]
        print(f"Setting CUDA_VISIBLE_DEVICES to: {gpu}")
        gpu_setting = f"CUDA_VISIBLE_DEVICES={gpu}"

    HAMER_DIR = SRC_DIR
    print("HAMER DIR", HAMER_DIR)

    yolo_model = os.environ.get("CHOIR_HAMER_YOLO_MODEL", DEFAULT_YOLO_MODEL)
    if not os.path.isfile(yolo_model):
        raise FileNotFoundError(
            f"HaMeR YOLO detector not found at {yolo_model}. "
            "Place stage1_preprocess/Yolov8/models/wilor_hand_detector.pt or set CHOIR_HAMER_YOLO_MODEL."
        )
    print(f"Using YOLO hand detector: {yolo_model}")

    cmd_args = [
        f"cd {HAMER_DIR};",
    ]
    if gpu_setting:
        cmd_args.append(gpu_setting)
    cmd_args.extend([
        "python -u run.py",
        f"--img_folder {img_dir} ",
        f"--res_folder {res_dir}/demo_{name}.pkl ",
        f"--batch_size=48 --side_view --save_mesh --full_frame",
        f"--type {datatype}",
        f"--checkpoint {ROOT_DIR}",
        f"--yolo_model {yolo_model}",
    ])

    cmd = " ".join(cmd_args)
    print(cmd)
    return subprocess.call(cmd, shell=True)


def process_seq(
    gpus,
    out_root,
    seq,
    img_dir,
    out_name="hamer_out",
    datatype=None,
    track_name="track_preds",
    shot_name="shot_idcs",
    overwrite=False,
):
    """
    Run and export HAMER results
    """
    name = os.path.basename(seq)
    res_root = f"{out_root}/{out_name}/{seq}"
    os.makedirs(res_root, exist_ok=True)
    res_dir = os.path.join(res_root, "results")
    res_path = f"{res_root}/{name}.pkl"

    if overwrite or not os.path.isfile(res_path):
        res = launch_hamer(gpus, seq, img_dir, res_dir, name, datatype, overwrite)
        print(f'rename {res_dir}/demo_{name}.pkl into ', res_path)
        os.rename(f"{res_dir}/demo_{name}.pkl", res_path)
        assert res == 0, "HAMER FAILED"

    # export the HAMER predictions
    track_dir = f"{out_root}/{track_name}/{seq}"
    shot_path = f"{out_root}/{shot_name}/{seq}.json"

    export_sequence_results(res_path, track_dir, shot_path)
    return 0


def get_out_dir(src_root, src_dir, src_token, out_token):
    """
    :param src_root (str) root of all data
    :param src_dir (str) img input dir
    :param src_token (str) parent name of image input dir
    :param out_token (str) name of output dir
    """
    src_suffix = src_dir.removeprefix(src_root)
    out_dir = f"{out_root}/{src_suffix}"
    return out_dir.replace(src_token, out_token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="posetrack", help="dataset to process")
    parser.add_argument("--root", default=None, help="root dir of data, default None")
    parser.add_argument("--split", default="val", help="split of dataset, default val")
    parser.add_argument("--img_name", default=None, help="input image directory name, default None")
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--gpus", nargs="*", default=[0])
    parser.add_argument("-y", "--overwrite", action="store_true")

    args = parser.parse_args()
    args = update_args(args)

    out_root = f"{args.root}/slahmr/{args.split}"

    print(f"running phalp on {len(args.img_dirs)} image directories")
    if len(args.gpus) > 1:
        with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as exe:
            for img_dir, seq in zip(args.img_dirs, args.seqs):
                exe.submit(
                    process_seq,
                    args.gpus,
                    out_root,
                    seq,
                    img_dir,
                    overwrite=args.overwrite,
                )
    else:
        for img_dir, seq in zip(args.img_dirs, args.seqs):
            process_seq(args.gpus, out_root, seq, img_dir, overwrite=args.overwrite)

import json
import multiprocessing as mp
import os
import sys
from functools import partial

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

import eval_common

sys.modules['common'] = eval_common
from smplx import MANOLayer

import eval_gt as gt
import eval_modules as eval_m
from eval_common.xdict import xdict

eval_fn_dict = {
    "mpjpe_ra_r": eval_m.eval_mpjpe_right,
    # "mrrpe_ho": eval_m.eval_mrrpe_ho_right,
    # "cd_f_ra": eval_m.eval_cd_f_ra,
    # "cd_f_right": eval_m.eval_cd_f_right,
}


def default_pred_data_path(seq_name=""):
    seq_name_short = seq_name.split('.')[0]
    return f"../../output/{seq_name_short}/eval_data.npy"


def load_data_step(seq_name="", gpu_id=0, pred_data_path=""):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading data for {seq_name}")

    vis_p = pred_data_path or default_pred_data_path(seq_name)
    pred_data = np.load(vis_p, allow_pickle=True).item()

    # hamer_output = f'../../output/{seq_name_short}/processed/hold_fit.slerp.npy'
    # hamer_output = np.load(hamer_output, allow_pickle=True).item()
    # hamer_output = hamer_output['right']
    # hamer_mano_global_orient = torch.from_numpy(hamer_output['global_orient']).to(device)  # (N, 3)
    # hamer_mano_trans = torch.from_numpy(hamer_output['transl']).to(device)  # (N, 3)
    # hamer_mano_pose = torch.from_numpy(hamer_output['hand_pose']).to(device)  # (N, 45)
    # hamer_mano_betas = torch.from_numpy(hamer_output['betas']).to(device)  # (N, 10)
    # from magichoi_mano.body_models import MANO
    # mano_layer = MANO(model_path='../../stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano', is_rhand=True, flat_hand_mean=False,
    #                   use_pca=False).to(device)

    # # hamer_mano_global_orient = axis_angle_to_matrix(hamer_mano_global_orient[:, None])  # (N, 1, 3, 3)
    # # hamer_mano_pose = axis_angle_to_matrix(hamer_mano_pose.reshape(-1, 15, 3))  # (N, 15, 3, 3)
    # mano_output = mano_layer(global_orient=hamer_mano_global_orient, transl=hamer_mano_trans, hand_pose=hamer_mano_pose, betas=hamer_mano_betas)

    # hamer_mano_joints = mano_output.joints
    # hamer_mano_verts = mano_output.vertices
    # hamer_mano_joints = hamer_mano_joints.cpu()
    # pred_data['j3d_ra.right'] = hamer_mano_joints - hamer_mano_joints[:, 0, None]
    # print(hamer_mano_joints.shape)

    out = xdict()
    out.update(pred_data)
    return out


def process_single_sequence(seq_name, gpu_id=0, debug=False, only_eval_hand=False, pred_data_path="", out_dir="", mvs_data_root=""):
    """Evaluate a single sequence"""
    from tqdm import tqdm

    # Set GPU for current process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Use cuda:0 in code since only one device is visible
    torch.cuda.set_device(0)

    try:
        print(f"\n{'='*60}")
        print(f"Processing sequence: {seq_name} [GPU {gpu_id}]")
        print(f"{'='*60}")

        # Build paths
        seq_base = seq_name.split('.')[0]  # e.g., hold_ABF12_ho3d
        seq_suffix = seq_name.split('.')[1] if '.' in seq_name else "0"  # e.g., 180

        mvs_data_root = mvs_data_root or os.environ.get("MVS_DATA_ROOT", "input_data")
        mvs_root = f"{mvs_data_root}/{seq_base}/processed/colmap_{seq_name}/sfm_superpoint+superglue/mvs/"
        out_dir = out_dir or f"HO3Dv3/our_metrics/{seq_base}/"

        # Load data
        data_pred = load_data_step(seq_name=seq_name, gpu_id=gpu_id, pred_data_path=pred_data_path)
        data_pred["out_dir"] = out_dir
        data_gt = gt.load_data_diff_object(
            full_seq_name=seq_name,
            mvs_root=mvs_root,
            debug=debug,
        )

        seq_name_full = data_pred["full_seq_name"]
        out_p = out_dir
        os.makedirs(out_p, exist_ok=True)

        # Set evaluation functions
        current_eval_fn_dict = eval_fn_dict.copy()
        if not only_eval_hand:
            current_eval_fn_dict["icp"] = eval_m.eval_icp_first_frame
            current_eval_fn_dict["cd_f_right"] = eval_m.eval_cd_f_right

        print(f"[{seq_name}][GPU {gpu_id}] Evaluation metrics:")
        for eval_fn_name in current_eval_fn_dict.keys():
            print(f"  - {eval_fn_name}")

        # Initialize metrics dictionary
        metric_dict = {}
        # Evaluate each metric
        pbar = tqdm(current_eval_fn_dict.items(), desc=f"[{seq_name}][GPU {gpu_id}]")
        for eval_fn_name, eval_fn in pbar:
            pbar.set_description(f"[{seq_name}][GPU {gpu_id}] Evaluating {eval_fn_name}")
            metric_dict = eval_fn(data_pred, data_gt, metric_dict)

        # Calculate mean values
        mean_metrics = {}
        for metric_name, values in metric_dict.items():
            mean_value = float(np.nanmean(values))
            mean_metrics[metric_name] = mean_value

        # Sort
        mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))

        print(f"\n[{seq_name}][GPU {gpu_id}] Evaluation results:")
        for metric_name, mean_value in mean_metrics.items():
            print(f"  {metric_name.upper()}: {mean_value:.2f}")

        # Save results
        json_path = out_p + "/metric.json"
        npy_path = out_p + "/metric_all.npy"

        from datetime import datetime
        current_time = datetime.now()
        time_str = current_time.strftime("%m-%d %H:%M")
        mean_metrics["timestamp"] = time_str
        mean_metrics["seq_name"] = seq_name_full

        with open(json_path, "w") as f:
            json.dump(mean_metrics, f, indent=4)
            print(f"[{seq_name}][GPU {gpu_id}] Saved mean metrics to {json_path}")

        np.save(npy_path, metric_dict)
        print(f"[{seq_name}][GPU {gpu_id}] Saved all metrics to {npy_path}")

        print(f"[{seq_name}][GPU {gpu_id}] ✓ Completed!")
        return (seq_name, True, mean_metrics)

    except Exception as e:
        print(f"[{seq_name}][GPU {gpu_id}] ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (seq_name, False, str(e))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parallel evaluation of multiple sequences')
    parser.add_argument("--sequences", type=str, nargs='+', default=None, help="List of sequences to process")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: auto-detect GPU count)")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Specify GPU IDs to use, comma-separated, e.g.: 0,1,2,3")
    parser.add_argument("--mvs_data_root", type=str, default=os.environ.get("MVS_DATA_ROOT", "input_data"), help="Root containing <seq_base>/processed/colmap_<seq>/...")
    parser.add_argument("--pred_data_root", type=str, default="", help="Optional root containing <seq_base>/eval_data.npy predictions")
    parser.add_argument("--out_root", type=str, default="", help="Optional root for per-sequence metric outputs")
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
    parser.add_argument("--only_eval_hand", default=False, action="store_true", help="Only evaluate hand")

    args = parser.parse_args()

    # Detect available GPUs
    env_gpu_ids = os.environ.get("GPU_IDS")
    if args.gpu_ids:
        # User specified GPUs
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    elif env_gpu_ids:
        gpu_ids = [int(x.strip()) for x in env_gpu_ids.replace(",", " ").split()]
    else:
        # Auto-detect all available GPUs
        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            print("Warning: No CUDA devices detected, will use CPU")
            gpu_ids = [0]

    # Set number of parallel workers
    if args.num_workers is None:
        num_workers = len(gpu_ids)
    else:
        num_workers = args.num_workers

    print(f"\n{'='*60}")
    print(f"GPU Configuration:")
    print(f"  Available GPUs: {len(gpu_ids)}")
    print(f"  GPU IDs: {gpu_ids}")
    print(f"  Number of workers: {num_workers}")
    print(f"{'='*60}\n")

    # 默认序列列表
    default_sequences = [
        # "hold_ABF12_ho3d.180",
        # "hold_ABF14_ho3d.180",
        # "hold_GPMF12_ho3d.90",  # NOTE problem!!!
        # "hold_GPMF14_ho3d.90",  # NOTE problem!!!
        # "hold_MC1_ho3d.0",  # NOTE problem!!!
        # "hold_MC4_ho3d.0",
        # "hold_MDF12_ho3d.60",
        # "hold_MDF14_ho3d.300",
        # "hold_ShSu10_ho3d.30",
        # "hold_ShSu12_ho3d.30",
        # "hold_SM2_ho3d.90",
        # "hold_SM4_ho3d.0",
        # "hold_SMu1_ho3d.0",
        "hold_SMu40_ho3d.0",
    ]

    sequences = args.sequences if args.sequences else default_sequences

    print("Sequence list:")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {seq}")
    print()

    # Assign GPU to each sequence (round-robin)
    tasks = []
    for i, seq in enumerate(sequences):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        seq_base = seq.split(".")[0]
        pred_data_path = os.path.join(args.pred_data_root, seq_base, "eval_data.npy") if args.pred_data_root else ""
        out_dir = os.path.join(args.out_root, seq_base) if args.out_root else ""
        tasks.append((seq, gpu_id, args.debug, args.only_eval_hand, pred_data_path, out_dir, args.mvs_data_root))

    print("Task assignment:")
    gpu_task_count = {gpu_id: 0 for gpu_id in gpu_ids}
    for seq, gpu_id, _, _, _, _, _ in tasks:
        gpu_task_count[gpu_id] += 1
    for gpu_id, count in gpu_task_count.items():
        print(f"  GPU {gpu_id}: {count} sequences")
    print()

    # Create process pool for parallel processing
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(process_single_sequence, tasks)

    # Summarize results
    print(f"\n{'='*60}")
    print("All sequences processed!")
    print(f"{'='*60}\n")

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"Successful: {len(successful)}/{len(sequences)}")
    print(f"Failed: {len(failed)}/{len(sequences)}\n")

    if successful:
        print("Successful sequences:")
        for seq_name, _, metrics in successful:
            print(f"  ✓ {seq_name}")

    if failed:
        print("\nFailed sequences:")
        for seq_name, _, error in failed:
            print(f"  ✗ {seq_name}: {error}")

    # Save summary results
    summary_file = "HO3Dv3/our_metrics_summary.json"
    summary = {"total": len(sequences), "successful": len(successful), "failed": len(failed), "results": {}}

    for seq_name, success, data in results:
        if success:
            summary["results"][seq_name] = data
        else:
            summary["results"][seq_name] = {"error": data}

    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSummary results saved to: {summary_file}")


if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    main()

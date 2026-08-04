import json
import os
import sys

import numpy as np
import smplx
import torch
import trimesh

import eval_common

sys.modules['common'] = eval_common
import eval_gt as gt
import eval_modules as eval_m
from eval_common.xdict import xdict
from eval_modules import compute_bounding_box_centers
from utils_simba.geometry import transform_points
from utils_simba.hand import initialize_mano_model

device = "cuda:0"

eval_fn_dict = {
    "mpjpe_ra_r": eval_m.eval_mpjpe_right,
    # "mrrpe_ho": eval_m.eval_mrrpe_ho_right,
    # "cd_f_ra": eval_m.eval_cd_f_ra,
    # "cd_f_right": eval_m.eval_cd_f_right,
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_p", type=str, default="")
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--mvs_root", type=str, default="")
    parser.add_argument("--object_mesh_f", type=str, default="")
    parser.add_argument("--MANO_f", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--only_eval_hand", default=False, action="store_true")

    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args


def load_data_step(
    seq_name="",
    data_root="",
    mvs_root="",
    object_mesh_f="",
    ckpt_p="",
    MANO_f="",
    common_fate=False,
):

    device = "cuda:0"
    print("Loading data")

    # from src.utils.io.optim import load_data
    # ckpt_p='logs/hold_GPMF14_ho3d_pre_train/checkpoints/last.pose_ref'
    # ckpt_p='logs/' + seq_name.split('.')[0] + '_pre_train/checkpoints/last.pose_ref'
    # "data/hold_MC1_ho3d/processed//colmap_hold_MC1_ho3d.80/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy",
    # out, ckpt = load_data(ckpt_p, pose_p=f"{mvs_root}/o2w_normalized_aligned.npy", object_mesh_f=object_mesh_f)
    # breakpoint()
    vis_p = f"{data_root}/hold_fit.aligned.npy"

    data = np.load(vis_p, allow_pickle=True).item()
    intrinsic = data['object']["K"]  # Camera intrinsics
    img_fs = data['object']["im_paths"]

    ckpt = torch.load(ckpt_p, map_location='cpu')
    sd = xdict(ckpt["state_dict"])
    # param_dict = sd.search(".params.")
    hand_scale = np.array(list(sd.search(".hand_scale").values())[0])
    betas = sd['models.right.hand_beta'].clone().cuda()
    betas = torch.tile(betas, (len(img_fs), 1))
    h2c_rot = sd["models.right.hand_rot"].clone().cuda()
    h2c_transl = sd["models.right.hand_transl"].clone().cuda()
    hand_pose = sd["models.right.hand_pose"].clone().cuda()

    # vis_p = f"{data_root}/hold_fit.aligned.npy"
    # data = load_data(vis_p)

    # # K = data['object']["K"].to(device).view(1, 4, 4)[:, :3, :3]
    # scale = 1 / data['object']['obj_scale']
    # scale = torch.tensor([scale]).float().to(device)
    mesh_c_o = trimesh.load(object_mesh_f, process=False)

    # ## log object ##
    # vis_p = f"{data_root}/hold_fit.aligned.npy"
    # data = load_data(vis_p)

    # Extract object data
    # pts_w = data['object']["j3d"]          # Shape: (B, number_3d_points, 3)
    # o2w_all = data['object']["o2w_all"]    # Shape: (B, 4, 4)
    o2c_mat = np.load(f"{mvs_root}/o2w_normalized_aligned.npy")
    c2o_mat = np.linalg.inv(o2c_mat)  # camera to object
    # obj_3d_o = trimesh.load(f"{mvs_root}/sparse_points_normalized_aligned.ply", process=False).vertices
    # obj_3d_o = np.tile(obj_3d_o, (o2c_mat.shape[0], 1, 1))
    # img_fs = data['object']["im_paths"]
    # intrinsic = data['object']["K"]        # Camera intrinsics
    # obj_scale = data['object']['obj_scale']

    ###### log hand ########
    # Initialize MANO model
    f3d_r = initialize_mano_model(MANO_f)

    # betas = data['right']["hand_beta"]
    # betas = torch.tensor(np.tile(betas, (o2c_mat.shape[0], 1))).cuda()
    # h2c_rot = torch.tensor(data['right']["hand_rot"]).cuda()
    # h2c_transl = torch.tensor(data['right']["hand_transl"]).cuda()
    # hand_pose = torch.tensor(data['right']["hand_pose"]).cuda()

    # Extract and transform right hand vertices
    from magichoi_mano.body_models import MANO
    mano_layer = MANO(model_path=MANO_f, is_rhand=True, use_pca=False)
    # mano_layer = smplx.create(model_path=MANO_f, model_type="mano", use_pca=False, is_rhand=True)
    mano_layer.to(torch.device("cuda"))
    ### Note: Hand coordinates to object coordinates only involves translation, not rotation
    # hand vertices in canonical coordinates
    # breakpoint()
    hand_out_can = mano_layer(
        betas=betas,
        hand_pose=hand_pose,
        transl=torch.zeros_like(h2c_transl),
        global_orient=h2c_rot,
    )
    hand_v_can = hand_out_can.vertices.cpu().numpy()
    hand_jnts_can = hand_out_can.joints.cpu().numpy()
    h2c_mat = np.tile(np.eye(4), (o2c_mat.shape[0], 1, 1))
    h2c_mat[:, :3, 3] = h2c_transl.cpu().numpy()
    h2c_mat = h2c_mat * hand_scale  # scale the hand
    h2c_mat[:, 3, 3] = 1
    h2o_mat = c2o_mat @ h2c_mat
    # hand vertices in camera coordinates
    # hand vertices in object coordinates
    hand_v_o = transform_points(hand_v_can, h2o_mat)
    hand_jnts_o = transform_points(hand_jnts_can, h2o_mat)

    hand_v_c = transform_points(hand_v_o, o2c_mat)
    hand_jnts_c = transform_points(hand_jnts_o, o2c_mat)

    object_v_c = transform_points(mesh_c_o.vertices, o2c_mat)

    hand_v_c /= hand_scale
    hand_jnts_c /= hand_scale

    object_v_c = object_v_c / hand_scale

    out = xdict()
    out['verts.right'] = hand_v_c
    out['jnts.right'] = hand_jnts_c
    out['root.right'] = hand_jnts_c[:, 0, :]
    out['j3d_ra.right'] = hand_jnts_can - hand_jnts_can[:, 0:1, :]
    out['verts.object'] = object_v_c
    out['v3d_c.object'] = out['verts.object']
    out['root.object'] = compute_bounding_box_centers(out['verts.object'])
    out['v3d_ra.object'] = out['verts.object'] - out['root.object'][:, None, :]
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]

    faces = {
        'object': np.array(mesh_c_o.faces),
        'right': np.array(f3d_r),
    }
    out["faces"] = faces

    print("Done loading data")
    out = out.to_torch()
    out['verts.right'].float().to(device)
    out['jnts.right'].float().to(device)
    out['verts.object'].float().to(device)

    out_dict = xdict()
    out_dict["fnames"] = img_fs  #.tolist()

    out_dict["K"] = intrinsic[None]
    out_dict["full_seq_name"] = seq_name
    out_dict.merge(out)
    return out_dict


def main():
    from tqdm import tqdm
    args = parse_args()

    data_pred = load_data_step(
        seq_name=args.seq_name,
        data_root=args.data_root,
        mvs_root=args.mvs_root,
        object_mesh_f=args.object_mesh_f,
        ckpt_p=args.sd_p,
        MANO_f=args.MANO_f,
    )
    data_pred["out_dir"] = args.out_dir
    data_gt = gt.load_data_diff_object(
        full_seq_name=args.seq_name,
        mvs_root=args.mvs_root,
        debug=args.debug,
    )

    seq_name = data_pred["full_seq_name"]
    out_p = args.out_dir
    os.makedirs(out_p, exist_ok=True)
    if not args.only_eval_hand:
        eval_fn_dict["icp"] = eval_m.eval_icp_first_frame
        eval_fn_dict["cd_f_right"] = eval_m.eval_cd_f_right

    print("------------------")
    print("Involving the following eval_fn:")
    for eval_fn_name in eval_fn_dict.keys():
        print(eval_fn_name)
    print("------------------")

    # Initialize the metrics dictionaries
    metric_dict = {}
    # Evaluate each metric using the corresponding function
    pbar = tqdm(eval_fn_dict.items())
    for eval_fn_name, eval_fn in pbar:
        pbar.set_description(f"Evaluating {eval_fn_name}")
        metric_dict = eval_fn(data_pred, data_gt, metric_dict)

    # Dictionary to store mean values of metrics
    mean_metrics = {}

    # Print out the mean of each metric and store the results
    for metric_name, values in metric_dict.items():
        mean_value = float(np.nanmean(values))  # Convert mean value to native Python float
        mean_metrics[metric_name] = mean_value

    # sort by key
    mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))

    for metric_name, mean_value in mean_metrics.items():
        print(f"{metric_name.upper()}: {mean_value:.2f}")

    # Define the file paths
    json_path = out_p + "/metric.json"
    npy_path = out_p + "/metric_all.npy"

    from datetime import datetime

    current_time = datetime.now()
    time_str = current_time.strftime("%m-%d %H:%M")
    mean_metrics["timestamp"] = time_str
    mean_metrics["seq_name"] = seq_name
    print("Units: CD (cm**2), F-score (percentage), MPJPE (mm)")

    # Save the mean_metrics dictionary to a JSON file with indentation
    with open(json_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
        print(f"Saved mean metrics to {json_path}")

    # Save the metric_all numpy array
    np.save(npy_path, metric_dict)
    print(f"Saved metric_all numpy array to {npy_path}")


if __name__ == "__main__":
    main()

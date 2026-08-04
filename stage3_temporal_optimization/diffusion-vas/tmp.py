import numpy as np
import torch

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer
from debug_bbox import get_global_amodal_bbox, load_hand_data, load_raw_frames
from utils import farthest_point_sampling_torch

model = CoTrackerPredictor(checkpoint="./checkpoints/scaled_offline.pth").cuda()
raw_rgbs_np = load_raw_frames("../../output/87141/rgbs", frame_type='rgb')
raw_masks_np = load_raw_frames("../../output/87141/obj_masks", frame_type='mask')

valid_points_y, valid_points_x = np.where(raw_masks_np[0])
valid_points_xy = np.stack([valid_points_x, valid_points_y], axis=1)
valid_points_xy_torch = torch.from_numpy(valid_points_xy).float().cuda()
num_queries = 20 * 20

if valid_points_xy_torch.shape[0] > num_queries:
    sampled_points_torch = farthest_point_sampling_torch(valid_points_xy_torch, num_queries)
else:
    sampled_points_torch = valid_points_xy_torch

sampled_points_np = sampled_points_torch.cpu().numpy().astype(int)
sampled_x, sampled_y = sampled_points_np[:, 0], sampled_points_np[:, 1]

queries_2d = torch.from_numpy(np.stack([sampled_x, sampled_y], axis=1)).float().cuda()

# 1. run CoTracker
N = queries_2d.shape[0]
t_col = torch.zeros((N, 1)).cuda()  # frame index is always 0
queries = torch.cat([t_col, queries_2d], dim=1)[None]  # (1, N, 3)

import ipdb

ipdb.set_trace()
raw_rgbs_tensor = torch.from_numpy(np.stack(raw_rgbs_np, axis=0)).permute(0, 3, 1, 2).unsqueeze(0).cuda().contiguous().float()
with torch.no_grad():
    pred_tracks, pred_vis = model(
        raw_rgbs_tensor,
        queries=queries,
        backward_tracking=True,
    )
vis = Visualizer(save_dir="./debug_output")
vis.visualize(raw_rgbs_tensor, pred_tracks, pred_vis, filename="tracking_visualization")

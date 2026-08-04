import os

import numpy as np
import torch
from manotorch.manolayer import ManoLayer as AMANOLayer
from pytorch3d.transforms import axis_angle_to_matrix
from scipy.spatial.transform import Rotation as R
from smplx import MANOLayer

# 1. 实例化两个 Layer
mano_dir = '../../stage1_preprocess/Dyn_HaMR_new/_DATA/data'
user_layer = AMANOLayer(mano_assets_root=os.path.join(mano_dir, 'mano'), flat_hand_mean=True, side='right')  # 你的
gt_layer = MANOLayer(model_path=os.path.join(mano_dir, 'mano'), is_rhand=True, flat_hand_mean=True)  # GT的

# 2. 构造全零输入
batch_size = 1
zeros_pose = torch.zeros(batch_size, 48)  # 3 global + 45 hand

# 3. 获取输出的 Global Rotation Matrix
# 注意：我们只关心 Global Rotation 的差异
with torch.no_grad():
    # 你的输出 (Flat Hand)
    user_out = user_layer(zeros_pose)
    # 获取手腕(Root)的旋转矩阵 [1, 3, 3]
    # 注意：你需要从你的 user_out.transforms_abs 或 similar 拿到 Root Global Rot
    R_user = user_out.transforms_abs[:, 0, :3, :3].cpu().numpy()
    print('R_user:', user_out.joints)

    # GT 输出 (Relaxed Hand)
    gt_out = gt_layer(global_orient=axis_angle_to_matrix(torch.zeros(1, 1, 3)), hand_pose=axis_angle_to_matrix(torch.zeros(1, 15, 3)))
    # 同样获取 Root Global Rot
    R_gt = gt_out.global_orient.cpu().numpy()  # 假设 GT 有这个或通过 axisang 转
    print('R_gt:', gt_out.joints)

#

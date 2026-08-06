import torch.nn as nn
import torch.nn.functional as F

from model.pointnet2_utils import PointNetSetAbstractionMsg


class PointNet2TokenEncoder(nn.Module):

    def __init__(self, output_dim, normal_channel=True):
        super(PointNet2TokenEncoder, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1, 0.2], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.out_projection = nn.Conv1d(640, output_dim, 1)

    def forward(self, xyz):
        # More robust layout check:
        # If last dim is small (<=6) and middle dim is large, input is (B, N, C) and needs transpose
        if xyz.shape[-1] <= 6 and xyz.shape[1] > 6:
            xyz = xyz.transpose(1, 2).contiguous()

        # xyz is now guaranteed (B, C, N)
        B, C, N = xyz.shape

        if self.normal_channel:
            # PointNet++ utils assume first 3 channels are XYZ, rest are features
            norm = xyz[:, 3:, :]  # normals / colors
            xyz = xyz[:, :3, :]  # xyz coordinates
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # (B, 640, 128) -> (B, output_dim, 128)
        tokens = self.out_projection(l2_points)

        # Back to Transformer layout (B, N, Dim)
        tokens = tokens.transpose(1, 2).contiguous()
        l2_xyz = l2_xyz.transpose(1, 2).contiguous()

        return tokens, l2_xyz

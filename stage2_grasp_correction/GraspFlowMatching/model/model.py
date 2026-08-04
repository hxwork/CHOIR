import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

from model.pointnet2_cls_msg import PointNet2TokenEncoder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ScaleEmbedder(nn.Module):
    """Map scalar obj_scale (B, 1) to (B, D)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, scale):
        return self.mlp(scale)


class CameraRayEmbedder(nn.Module):
    """Map camera ray (B, 3) to (B, D)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, camera_ray):
        return self.mlp(camera_ray)


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class GraspSiTBlock(nn.Module):
    """SiT block with self-attention and cross-attention to object tokens."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))

    def forward(self, x, c, obj_feats, obj_padding_mask=None):
        """
        x: (B, L_hand, D)
        c: (B, D) global condition (time)
        obj_feats: (B, L_obj, D)
        obj_padding_mask: (B, L_obj) True = padding
        """
        shift_msa, scale_msa, gate_msa, \
        shift_cross, scale_cross, gate_cross, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        x_norm = modulate(self.norm_cross(x), shift_cross, scale_cross)
        attn_out, _ = self.cross_attn(query=x_norm, key=obj_feats, value=obj_feats, key_padding_mask=obj_padding_mask)
        x = x + gate_cross.unsqueeze(1) * attn_out

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class GraspDepthMagSiT(nn.Module):

    def __init__(
        self,
        input_dim=1,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=False,
    ):
        super().__init__()

        self.class_dropout_prob = class_dropout_prob
        self.learn_sigma = learn_sigma

        self.point_encoder = PointNet2TokenEncoder(output_dim=hidden_size, normal_channel=True)

        self.obj_pos_embedder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.x_embedder = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.hand_embedder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.scale_embedder = ScaleEmbedder(hidden_size)
        self.camera_ray_embedder = CameraRayEmbedder(hidden_size)

        # Tokens: Depth(1) + Scale(1) + Ray(1) + Hand(21) = 24
        self.total_tokens = 1 + 1 + 1 + 21
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_tokens, hidden_size))

        self.blocks = nn.ModuleList([GraspSiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])

        self.out_channels = input_dim * 2 if learn_sigma else input_dim
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        self.null_obj_feat = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.null_scale_embed = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.null_ray_embed = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.null_hand_embed = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.null_obj_feat, std=0.02)
        nn.init.normal_(self.null_scale_embed, std=0.02)
        nn.init.normal_(self.null_ray_embed, std=0.02)
        nn.init.normal_(self.null_hand_embed, std=0.02)

        nn.init.normal_(self.obj_pos_embedder[0].weight, std=0.02)
        nn.init.normal_(self.obj_pos_embedder[2].weight, std=0.02)
        nn.init.normal_(self.x_embedder[0].weight, std=0.02)
        nn.init.normal_(self.x_embedder[2].weight, std=0.02)
        nn.init.normal_(self.hand_embedder[0].weight, std=0.02)
        nn.init.normal_(self.hand_embedder[2].weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.scale_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.scale_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.camera_ray_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.camera_ray_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, cond_x, cond_obj_scale, cond_obj_pointcloud, cond_camera_ray, use_cfg=True, force_drop=None):
        B = x.shape[0]

        if force_drop is None:
            final_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        elif isinstance(force_drop, bool):
            final_mask = torch.full((B,), force_drop, dtype=torch.bool, device=x.device)
        else:
            final_mask = force_drop

        if use_cfg and self.training and self.class_dropout_prob > 0:
            rand_mask = torch.rand(B, device=x.device) < self.class_dropout_prob
            final_mask = final_mask | rand_mask

        x_token = self.x_embedder(x)
        hand_token = self.hand_embedder(cond_x)

        obj_token, obj_xyz = self.point_encoder(cond_obj_pointcloud)
        obj_pos_emb = self.obj_pos_embedder(obj_xyz)
        obj_token = obj_token + obj_pos_emb

        t_emb = self.t_embedder(t)

        scale_token = self.scale_embedder(cond_obj_scale).unsqueeze(1)
        camera_ray_token = self.camera_ray_embedder(cond_camera_ray).unsqueeze(1)

        if final_mask.any():
            N_tokens = obj_token.shape[1]
            null_feats_expanded = self.null_obj_feat.expand(1, N_tokens, -1)
            obj_token[final_mask] = null_feats_expanded.type_as(obj_token)

            scale_token[final_mask] = self.null_scale_embed.type_as(scale_token)
            camera_ray_token[final_mask] = self.null_ray_embed.type_as(camera_ray_token)

            N_hand = hand_token.shape[1]
            null_hand_expanded = self.null_hand_embed.expand(1, N_hand, -1)
            hand_token[final_mask] = null_hand_expanded.type_as(hand_token)

        c = t_emb
        tokens = torch.cat([x_token, scale_token, camera_ray_token, hand_token], dim=1)
        tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens, c, obj_token)

        x = tokens[:, 0:1, :]
        x = self.final_layer(x, c)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=-1)

        return x

    def forward_with_cfg(self, x, t, cond_x, cond_obj_scale, cond_obj_pointcloud, cond_camera_ray, cfg_scale):
        combined = torch.cat([x, x], dim=0)
        t_combined = torch.cat([t, t], dim=0)

        cond_x_combined = torch.cat([cond_x, cond_x], dim=0)
        cond_scale_combined = torch.cat([cond_obj_scale, cond_obj_scale], dim=0)
        cond_obj_pointcloud_combined = torch.cat([cond_obj_pointcloud, cond_obj_pointcloud], dim=0)
        cond_camera_ray_combined = torch.cat([cond_camera_ray, cond_camera_ray], dim=0)

        B = combined.shape[0]
        force_drop_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        force_drop_mask[B // 2:] = True

        model_out = self.forward(
            combined,
            t_combined,
            cond_x_combined,
            cond_scale_combined,
            cond_obj_pointcloud_combined,
            cond_camera_ray_combined,
            use_cfg=False,
            force_drop=force_drop_mask,
        )
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        final_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return final_eps

    @torch.no_grad()
    def encode_condition_features(self, cond_obj_pointcloud):
        """Cache object tokens once before the sampling loop."""
        cond_obj_token, cond_obj_xyz = self.point_encoder(cond_obj_pointcloud)
        cond_obj_pos_emb = self.obj_pos_embedder(cond_obj_xyz)
        cond_obj_token = cond_obj_token + cond_obj_pos_emb
        return cond_obj_token

    def forward_inference(self, x, t, cond_x, cond_obj_scale, cond_obj_pointcloud_emb, cond_camera_ray, force_drop=None):
        """Inference forward that takes precomputed object embeddings."""
        B = x.shape[0]

        if force_drop is None:
            final_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        elif isinstance(force_drop, bool):
            final_mask = torch.full((B,), force_drop, dtype=torch.bool, device=x.device)
        else:
            final_mask = force_drop

        x_token = self.x_embedder(x)
        hand_token = self.hand_embedder(cond_x)

        t_emb = self.t_embedder(t)
        scale_token = self.scale_embedder(cond_obj_scale).unsqueeze(1)
        camera_ray_token = self.camera_ray_embedder(cond_camera_ray).unsqueeze(1)

        obj_token = cond_obj_pointcloud_emb.clone()
        if final_mask.any():
            N_tokens = obj_token.shape[1]
            null_feats_expanded = self.null_obj_feat.expand(1, N_tokens, -1)
            obj_token[final_mask] = null_feats_expanded.type_as(obj_token)

            scale_token[final_mask] = self.null_scale_embed.type_as(scale_token)
            camera_ray_token[final_mask] = self.null_ray_embed.type_as(camera_ray_token)

            N_hand = hand_token.shape[1]
            null_hand_expanded = self.null_hand_embed.expand(1, N_hand, -1)
            hand_token[final_mask] = null_hand_expanded.type_as(hand_token)

        c = t_emb
        tokens = torch.cat([x_token, scale_token, camera_ray_token, hand_token], dim=1)
        tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens, c, obj_token)

        x = tokens[:, 0:1, :]
        x = self.final_layer(x, c)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=-1)

        return x

    def forward_inference_with_cfg(self, x, t, cond_x, cond_obj_scale, cond_obj_pointcloud_emb, cond_camera_ray, cfg_scale):
        """CFG sampling with precomputed object embeddings."""
        combined = torch.cat([x, x], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        cond_x_combined = torch.cat([cond_x, cond_x], dim=0)
        cond_scale_combined = torch.cat([cond_obj_scale, cond_obj_scale], dim=0)
        cond_camera_ray_combined = torch.cat([cond_camera_ray, cond_camera_ray], dim=0)
        cond_obj_emb_combined = torch.cat([cond_obj_pointcloud_emb, cond_obj_pointcloud_emb], dim=0)

        B = combined.shape[0]
        force_drop_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        force_drop_mask[B // 2:] = True

        model_out = self.forward_inference(
            combined,
            t_combined,
            cond_x_combined,
            cond_scale_combined,
            cond_obj_pointcloud_emb=cond_obj_emb_combined,
            cond_camera_ray=cond_camera_ray_combined,
            force_drop=force_drop_mask,
        )

        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        final_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return final_eps

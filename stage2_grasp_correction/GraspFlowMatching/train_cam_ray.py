# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch.distributed as dist
import trimesh
from diffusers.models import AutoencoderKL
from PIL import Image
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb_utils
from data_layout import DEFAULT_MANO_ASSETS
from data_loader.GraspPair import GraspPair
from manotorch.manolayer import ManoLayer
from model.model import GraspDepthMagSiT
from train_utils import parse_transport_args
from transport import Sampler, create_transport

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(level=logging.INFO,
                            format='[\033[34m%(asctime)s\033[0m] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")])
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def decode_mano_params(hand_tensor, depth_mag=None, camera_ray=None):
    """
    Decode a (17, 6) hand representation into MANO params (translation, global rotation, pose).
    """
    if not isinstance(hand_tensor, torch.Tensor):
        hand_tensor = torch.from_numpy(hand_tensor)

    trans = hand_tensor[0, :3].unsqueeze(0)  # (1, 3)
    if depth_mag is not None and camera_ray is not None:
        trans = trans + depth_mag * camera_ray
    root_rot_6d = hand_tensor[1, :].unsqueeze(0)  # (1, 6)
    pose_6d = hand_tensor[2:, :]  # (15, 6)

    # 6D -> axis-angle
    root_rot_mat = rotation_6d_to_matrix(root_rot_6d)
    root_rot_aa = matrix_to_axis_angle(root_rot_mat)  # (1, 3)

    pose_mat = rotation_6d_to_matrix(pose_6d)
    pose_aa = matrix_to_axis_angle(pose_mat)  # (15, 3)

    # Concatenate into the format expected by ManoLayer
    pose_coeffs = torch.cat([root_rot_aa, pose_aa.view(1, 45)], dim=1)  # (1, 48)

    return trans, pose_coeffs


def get_hand_mesh(trans, pose_coeffs, mano_layer):
    """
    Generate a hand mesh with ManoLayer.
    """
    mano_output = mano_layer(pose_coeffs)
    verts = mano_output.verts + trans
    verts = verts[0].detach().cpu().numpy()
    faces = mano_layer.th_faces.detach().cpu().numpy()
    return verts, faces


def create_ray_mesh(vec, radius=0.001, color=[255, 255, 0, 255], length_scale=5.0):
    """
    Create a cylinder mesh along vector vec.
    """
    length = np.linalg.norm(vec)

    # 1. Create a canonical cylinder along Z, centered at the origin
    # sections=8 is enough; no need for high tessellation
    mesh = trimesh.creation.cylinder(radius=radius, height=length * length_scale, sections=8)

    # 2. Rotate so Z aligns with vec
    # trimesh.geometry.align_vectors returns a 4x4 transform
    direction = vec / length
    # Canonical Z axis is [0, 0, 1]
    rot_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
    mesh.apply_transform(rot_matrix)

    # 3. Set color
    mesh.visual.vertex_colors = color

    return mesh


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())
    mano_layer = None

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_name = f"{experiment_index:03d}-{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info("Initializing MANO layer...")
        mano_layer = ManoLayer(
            mano_assets_root=str(DEFAULT_MANO_ASSETS),
            side='right',
            use_pca=False,
            flat_hand_mean=True,
        )

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    model = GraspDepthMagSiT().to(device)

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)

    train_steps = 0
    start_epoch = 0
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        assert os.path.isfile(ckpt_path), f'Could not find SiT checkpoint at {ckpt_path}'
        state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        args = state_dict["args"]
        if "epoch" in state_dict:
            start_epoch = state_dict["epoch"]
        if "train_steps" in state_dict:
            train_steps = state_dict["train_steps"]

    requires_grad(ema, False)

    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)  # default: velocity;
    transport_sampler = Sampler(transport)
    logger.info(f"GraspSiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    train_dataset = GraspPair(data_split='train', debug=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    train_loader = DataLoader(train_dataset,
                              batch_size=local_batch_size,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    val_dataset = GraspPair(data_split='val', debug=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False, seed=args.global_seed)
    val_loader = DataLoader(val_dataset,
                            batch_size=local_batch_size,
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False)
    logger.info(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with :
    use_cfg = args.cfg_scale > 1.0

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        model.train()
        for input_dict in train_loader:
            x = input_dict['noise_depth_mag']
            cond_x = input_dict['cond_x']
            gt_hand = input_dict['gt_hand']
            cond_hand = input_dict['cond_hand']
            cond_obj_scale = input_dict['cond_obj_scale']
            cond_obj_pointcloud = input_dict['cond_obj_pointcloud']
            cond_camera_ray = input_dict['cond_camera_ray']

            x = x.to(device)
            cond_x = cond_x.to(device)
            gt_hand = gt_hand.to(device)
            cond_hand = cond_hand.to(device)
            cond_obj_scale = cond_obj_scale.to(device)
            cond_obj_pointcloud = cond_obj_pointcloud.to(device)
            cond_camera_ray = cond_camera_ray.to(device)
            model_kwargs = dict(cond_x=cond_hand, cond_obj_scale=cond_obj_scale, cond_obj_pointcloud=cond_obj_pointcloud, cond_camera_ray=cond_camera_ray)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log({"train loss": avg_loss, "train steps/sec": steps_per_sec}, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            # === C. Sampling / Visualization Phase ===
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info(f"Generating EMA samples at step {train_steps}...")

                # 1. Create sampling noise
                # Noise shape (local_batch_size, 17, 6)
                # # NOTE for debugging
                # g = torch.Generator(device=device)
                # g.manual_seed(42)
                # zs = torch.randn(local_batch_size, 17, 6, device=device, generator=g)
                zs = torch.randn(local_batch_size, 1, 1, device=device)

                # 2. Prepare conditioning inputs
                base_kwargs = dict(cond_x=cond_hand, cond_obj_scale=cond_obj_scale, cond_obj_pointcloud=cond_obj_pointcloud, cond_camera_ray=cond_camera_ray)

                # 3. Configure model call
                if use_cfg:
                    # [Important]
                    # forward_with_cfg internally cats [zs, zs] and [cond, cond]
                    sample_model_kwargs = dict(cfg_scale=args.cfg_scale, **base_kwargs)
                    model_fn = ema.forward_with_cfg
                else:
                    # Without CFG, pass single-copy noise and conditions
                    sample_model_kwargs = base_kwargs
                    model_fn = ema.forward

                # 4. Run ODE sampling
                sample_fn = transport_sampler.sample_ode()
                # sample output follows zs shape
                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                dist.barrier()

                # --- Save / log samples ---
                # Allocate gather buffers
                all_preds = torch.zeros((args.global_batch_size, *samples.shape[1:]), device=device, dtype=samples.dtype)
                all_gts = torch.zeros((args.global_batch_size, *gt_hand.shape[1:]), device=device, dtype=gt_hand.dtype)
                all_conds = torch.zeros((args.global_batch_size, *cond_x.shape[1:]), device=device, dtype=cond_x.dtype)
                all_cond_obj_scales = torch.zeros((args.global_batch_size, *cond_obj_scale.shape[1:]), device=device, dtype=cond_obj_scale.dtype)
                all_cond_obj_pointclouds = torch.zeros((args.global_batch_size, *cond_obj_pointcloud.shape[1:]), device=device, dtype=cond_obj_pointcloud.dtype)
                all_cond_camera_rays = torch.zeros((args.global_batch_size, *cond_camera_ray.shape[1:]), device=device, dtype=cond_camera_ray.dtype)

                # All-gather (contiguous() for safety)
                dist.all_gather_into_tensor(all_preds, samples.contiguous())
                dist.all_gather_into_tensor(all_gts, gt_hand.contiguous())
                dist.all_gather_into_tensor(all_conds, cond_x.contiguous())
                dist.all_gather_into_tensor(all_cond_obj_scales, cond_obj_scale.contiguous())
                dist.all_gather_into_tensor(all_cond_obj_pointclouds, cond_obj_pointcloud.contiguous())
                dist.all_gather_into_tensor(all_cond_camera_rays, cond_camera_ray.contiguous())
                if rank == 0:
                    # Include step in filename to avoid overwrite
                    save_path = f"{checkpoint_dir}/samples_epoch_{epoch}_step_{train_steps}.pt"
                    torch.save(
                        {
                            "pred_hand": all_preds.cpu(),
                            "gt_hand": all_gts.cpu(),
                            "cond_hand": all_conds.cpu(),
                            "cond_obj_scale": all_cond_obj_scales.cpu(),
                            "cond_obj_pointcloud": all_cond_obj_pointclouds.cpu(),
                            "cond_camera_ray": all_cond_camera_rays.cpu()
                        }, save_path)
                    logger.info(f"Saved samples to {save_path}")

                    # Visualize one sample from the batch and save as .ply
                    sample_idx = 0
                    pred_depth_mag = all_preds[sample_idx].cpu()
                    gt_hand = all_gts[sample_idx].cpu()
                    cond_hand = all_conds[sample_idx].cpu()
                    obj_pc = all_cond_obj_pointclouds[sample_idx].cpu()[:, :3]
                    obj_scale = all_cond_obj_scales[sample_idx].cpu()
                    camera_ray = all_cond_camera_rays[sample_idx].cpu()
                    # --- Process hands ---
                    # GT Hand (green)
                    gt_trans, gt_pose_coeffs = decode_mano_params(gt_hand)
                    gt_trans = gt_trans * obj_scale
                    gt_hand_verts, gt_faces = get_hand_mesh(gt_trans, gt_pose_coeffs, mano_layer)
                    gt_mesh = trimesh.Trimesh(vertices=gt_hand_verts, faces=gt_faces)
                    gt_mesh.visual.vertex_colors = [0, 255, 0, 200]

                    # Pred Hand (red)
                    pred_trans, pred_pose_coeffs = decode_mano_params(cond_hand, pred_depth_mag, camera_ray)
                    pred_trans = pred_trans * obj_scale
                    pred_hand_verts, pred_faces = get_hand_mesh(pred_trans, pred_pose_coeffs, mano_layer)
                    pred_mesh = trimesh.Trimesh(vertices=pred_hand_verts, faces=pred_faces)
                    pred_mesh.visual.vertex_colors = [255, 0, 0, 200]

                    # Cond Hand (blue)
                    cond_trans, cond_pose_coeffs = decode_mano_params(cond_hand)
                    cond_trans = cond_trans * obj_scale
                    cond_hand_verts, cond_faces = get_hand_mesh(cond_trans, cond_pose_coeffs, mano_layer)
                    cond_mesh = trimesh.Trimesh(vertices=cond_hand_verts, faces=cond_faces)
                    cond_mesh.visual.vertex_colors = [0, 0, 255, 200]

                    # Cond Camera Ray (yellow)
                    cond_camera_ray_mesh = create_ray_mesh(camera_ray, radius=0.005, color=[255, 255, 0, 200])

                    # --- Process object ---
                    obj_pc_rescaled = obj_pc.numpy() * obj_scale.numpy()
                    if obj_pc_rescaled.shape[0] > 0:
                        sphere = trimesh.creation.icosphere(radius=0.002, subdivisions=1)
                        base_vertices = sphere.vertices
                        base_faces = sphere.faces
                        all_vertices = []
                        all_faces = []
                        vertex_offset = 0
                        for point in obj_pc_rescaled:
                            all_vertices.append(base_vertices + point)
                            all_faces.append(base_faces + vertex_offset)
                            vertex_offset += len(base_vertices)
                        final_vertices = np.concatenate(all_vertices, axis=0)
                        final_faces = np.concatenate(all_faces, axis=0)
                        object_mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
                        object_mesh.visual.vertex_colors = [128, 128, 128, 255]
                    else:
                        object_mesh = trimesh.Trimesh()

                    # --- Combine and save ---
                    scene = trimesh.Scene()
                    scene.add_geometry(object_mesh)
                    scene.add_geometry(gt_mesh)
                    scene.add_geometry(pred_mesh)
                    scene.add_geometry(cond_mesh)
                    scene.add_geometry(cond_camera_ray_mesh)

                    output_filename = f"samples_epoch_{epoch}_step_{train_steps}_sample_{sample_idx}.ply"
                    output_path = f"{checkpoint_dir}/{output_filename}"

                    scene.export(output_path)
                    logger.info(f"Saved visualization to {output_path}")

        # # === B. Validation Phase (End of Epoch) ===
        # # Run validation once per epoch
        # logger.info(f"Starting validation for epoch {epoch}...")
        # model.eval()  # switch to eval
        # val_running_loss = 0.0
        # val_steps = 0

        # with torch.no_grad():
        #     for x, cond_x, cond_obj_scale, cond_obj_pointcloud in val_loader:
        #         x = x.to(device)
        #         cond_x = cond_x.to(device)
        #         cond_obj_scale = cond_obj_scale.to(device)
        #         cond_obj_pointcloud = cond_obj_pointcloud.to(device)

        #         model_kwargs = dict(cond_x=cond_x, cond_obj_scale=cond_obj_scale, cond_obj_pointcloud=cond_obj_pointcloud)

        #         # Validation loss (same transport loss path)
        #         loss_dict = transport.training_losses(model, x, model_kwargs)
        #         val_running_loss += loss_dict["loss"].mean().item()
        #         val_steps += 1

        # # Aggregate val loss
        # avg_val_loss = torch.tensor(val_running_loss / val_steps, device=device)
        # dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
        # avg_val_loss = avg_val_loss.item() / dist.get_world_size()

        # logger.info(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}")
        # if args.wandb and rank == 0:
        #     wandb_utils.log({"val_loss": avg_val_loss}, step=train_steps)

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=512)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--ckpt_every", type=int, default=5_000)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom SiT checkpoint")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)

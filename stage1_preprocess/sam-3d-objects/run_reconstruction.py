"""Stage 1 entry: reconstruct a 3D object mesh from Yolov8 RGB + object mask."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import trimesh
from einops import repeat
from PIL import Image
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.transforms import Transform3d, quaternion_to_matrix
from tqdm import tqdm

sys.path.append("notebook")
from inference import Inference, load_image, load_mask

from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from sam3d_objects.pipeline.layout_post_optimization_utils import apply_transform, denormalize_f, get_mesh
from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer

# CHOIR repo root: stage1_preprocess/sam-3d-objects/run_reconstruction.py -> ../..
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = str(REPO_ROOT / "output")


def compute_loss(rendered, mask_gt, loss_weights, quat, translation, scale):
    """Silhouette MSE plus light pose regularizers."""
    pred_mask = rendered[..., 3][0]
    loss_mask = F.mse_loss(pred_mask, mask_gt[0, 0])

    quat_normalized = quat / quat.norm()
    loss_reg_q = F.mse_loss(quat_normalized, torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device))
    loss_reg_t = torch.norm(translation) ** 2
    loss_reg_s = (scale - 1.0) ** 2

    return (
        loss_weights["mask"] * loss_mask
        + loss_weights["reg_q"] * loss_reg_q
        + loss_weights["reg_t"] * loss_reg_t
        + loss_weights["reg_s"] * loss_reg_s
    )


def run_render_compare(mesh, center, intrinsics_np, H, W, mask, device):
    """Refine pose with a soft-silhouette renderer against the object mask."""
    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[intrinsics_np[0, 0], intrinsics_np[1, 1]]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[intrinsics_np[0, 2], intrinsics_np[1, 2]]], device=device, dtype=torch.float32),
        image_size=torch.tensor([[H, W]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )

    H, W = mask.shape[-2:]
    opt_raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=1e-6,
        faces_per_pixel=50,
        max_faces_per_bin=50000,
    )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=opt_raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    quat = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, requires_grad=True))
    translation = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True))
    scale = torch.nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))

    def get_optimizer(_stage):
        return torch.optim.Adam([quat, translation, scale], lr=5e-3)

    loss_weights = {"mask": 200, "reg_q": 0.1, "reg_t": 0.05, "reg_s": 0.05}
    prev_loss = None

    for stage in [1, 2]:
        optimizer = get_optimizer(stage)
        iters = [5, 25]
        for _ in range(iters[stage - 1]):
            optimizer.zero_grad()
            transformed = apply_transform(mesh, center, quat, translation, scale)
            rendered = silhouette_renderer(transformed)
            loss = compute_loss(rendered, mask, loss_weights, quat, translation, scale)
            loss.backward()
            optimizer.step()
            if prev_loss is not None and abs(loss.item() - prev_loss) < 1e-5:
                break
            prev_loss = loss.item()

    quat, translation, scale = quat.detach(), translation.detach(), scale.detach()
    quat_normalized = quat / quat.norm()
    R = quaternion_to_matrix(quat_normalized)
    return quat, translation, scale, R


def visualize_render_on_image(processed_outputs, image_path, intrinsics_np, H, W, output_path="rendered_on_image.png"):
    """Composite original and (optionally) refined meshes onto the input image and save."""
    print("Rendering meshes on input image...")

    if not processed_outputs:
        print("No outputs to render.")
        return

    device = processed_outputs[0]["mesh"].device

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_float = image_np.astype(np.float32) / 255.0

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[intrinsics_np[0, 0], intrinsics_np[1, 1]]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[intrinsics_np[0, 2], intrinsics_np[1, 2]]], device=device, dtype=torch.float32),
        image_size=torch.tensor([[H, W]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -5.0]])
    rgb_raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    rgb_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rgb_raster_settings),
        shader=shader,
    )

    original_meshes_to_render = []
    transformed_meshes_to_render = []

    for item in processed_outputs:
        mesh = item["mesh"]
        transformed_mesh = item["transformed_mesh"]
        color = item["color"]

        verts_rgb = repeat(
            torch.tensor(color, device=device, dtype=torch.float32),
            "c -> v c",
            v=mesh.verts_list()[0].shape[0],
        )
        mesh.textures = TexturesVertex(verts_features=[verts_rgb.clone()])
        transformed_mesh.textures = TexturesVertex(verts_features=[verts_rgb.clone()])

        original_meshes_to_render.append(mesh)
        transformed_meshes_to_render.append(transformed_mesh)

    if original_meshes_to_render:
        combined_original_mesh = join_meshes_as_scene(original_meshes_to_render)
        with torch.no_grad():
            rendered_original_image = rgb_renderer(combined_original_mesh)
    else:
        rendered_original_image = torch.zeros((1, H, W, 4), device=device)

    if transformed_meshes_to_render:
        combined_transformed_mesh = join_meshes_as_scene(transformed_meshes_to_render)
        with torch.no_grad():
            rendered_transformed_image = rgb_renderer(combined_transformed_mesh)
    else:
        rendered_transformed_image = torch.zeros((1, H, W, 4), device=device)

    rendered_rgba_orig = rendered_original_image[0, ..., :4].cpu().numpy()
    rendered_alpha_orig = rendered_rgba_orig[..., 3:4]
    composite_image_orig = rendered_rgba_orig[..., :3] * rendered_alpha_orig + image_float * (1 - rendered_alpha_orig)
    composite_image_orig_uint8 = (composite_image_orig * 255).astype(np.uint8)

    rendered_rgba_trans = rendered_transformed_image[0, ..., :4].cpu().numpy()
    rendered_alpha_trans = rendered_rgba_trans[..., 3:4]
    composite_image_trans = rendered_rgba_trans[..., :3] * rendered_alpha_trans + image_float * (1 - rendered_alpha_trans)
    composite_image_trans_uint8 = (composite_image_trans * 255).astype(np.uint8)

    concatenated_image_np = np.concatenate((image_np, composite_image_orig_uint8, composite_image_trans_uint8), axis=1)
    Image.fromarray(concatenated_image_np).save(output_path)
    print(f"Saved concatenated rendered image to {output_path}")


def visualize_scene_with_plotly(output, image_np):
    """Write an interactive Plotly HTML for mesh / pointmap alignment checks."""
    print("Generating scene visualization with Plotly...")

    points_local = output["mesh"][0].vertices.cpu()
    rotation = output["rotation"].cpu()
    translation = output["translation"].cpu()
    scale = output["scale"].cpu()
    pointmap = output.get("pointmap")
    image_tensor = torch.from_numpy(image_np[..., :3]).float()

    if pointmap is not None:
        pointmap = pointmap.cpu()
        # Visualizer expects (H, W, C); convert from (C, H, W) if needed.
        if pointmap.dim() == 3 and pointmap.shape[0] == 3:
            pointmap = pointmap.permute(1, 2, 0)

        pm_h, pm_w = pointmap.shape[:2]
        image_for_viz = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_for_viz = torch.nn.functional.interpolate(
            image_for_viz, size=(pm_h, pm_w), mode="bilinear", align_corners=False
        ).squeeze(0)
    else:
        image_for_viz = image_tensor

    if rotation.dim() == 1:
        rotation = rotation.unsqueeze(0)
    if translation.dim() == 1:
        translation = translation.unsqueeze(0)
    if scale.dim() == 1:
        scale = scale.unsqueeze(0)

    fig = SceneVisualizer.plot_scene(
        points_local=points_local,
        instance_quaternions_l2c=rotation,
        instance_positions_l2c=translation,
        instance_scales_l2c=scale,
        pointmap=pointmap,
        image=image_for_viz,
        title="Alignment Check",
        show_pointmap_as_mesh=True,
    )
    fig.write_html("scene_visualization.html")
    print("Saved interactive scene to scene_visualization.html")


def process_video_directory(video_dir, inference_model, output_root_dir, post_optimize, args):
    """Run reconstruction for one Yolov8 video folder (frame 0 RGB + obj mask)."""
    import sys
    from pathlib import Path as _Path
    _repo = _Path(__file__).resolve().parents[2]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from output_layout import VideoLayout

    video_id = os.path.basename(video_dir)
    if args.video_id is not None and video_id not in args.video_id:
        return
    print(f"--- Processing video ID: {video_id} ---")

    layout = VideoLayout.from_output_root(output_root_dir, video_id)
    layout.ensure_stage_dirs()
    layout.object_init_dir.mkdir(parents=True, exist_ok=True)
    layout.camera_dir.mkdir(parents=True, exist_ok=True)
    layout.env_depths_dir.mkdir(parents=True, exist_ok=True)

    image_path = os.path.join(str(layout.frames_rgb_dir), "0.png")
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}, skipping.")
        return

    print("Running inference for object mask...")
    single_mask_color = (0.0, 1.0, 0.0)
    mask_path = os.path.join(str(layout.object_masks_dir), "0.png")
    print(f"Processing {mask_path}...")
    image_np_full = load_image(image_path)
    mask_np = load_mask(mask_path)

    mask_idx = 0
    glb_path = str(layout.object_init_dir / f"glb_{mask_idx}.glb")
    intrinsics_path = str(layout.intrinsics_json)

    output = inference_model(image_np_full, mask_np, seed=42)
    outputs_data = [{
        "output": output,
        "color": single_mask_color,
        "mask_path": mask_path,
    }]

    output["glb"].export(glb_path)

    output_depth_dir = str(layout.env_depths_dir)
    depth_path = os.path.join(output_depth_dir, f"depth_{mask_idx}.exr")
    depth_map = output["ori_pointmap"][:, :, -1].cpu().numpy()
    cv2.imwrite(depth_path, depth_map, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

    pc_path = os.path.join(output_depth_dir, f"pc_{mask_idx}.ply")
    pc = output["ori_pointmap"][:, :, :3].cpu().numpy().reshape(-1, 3)
    # Convert OpenCV-style axes (x left, y up) to right/down for export.
    flat_mat = np.diag([-1.0, -1.0, 1.0])
    pc = pc @ flat_mat
    pc = trimesh.PointCloud(pc)
    pc.export(pc_path)

    if len(pc.vertices) > 100000:
        indices = np.random.choice(pc.vertices.shape[0], 100000, replace=False)
        sampled_pc = pc.copy()
        sampled_pc.vertices = pc.vertices[indices]
    else:
        sampled_pc = pc
    sampled_pc.export(os.path.join(output_depth_dir, f"sampled_pc_{mask_idx}.ply"))
    print(f"Saved intermediate outputs to {glb_path}")

    if not outputs_data:
        print(f"No objects were processed for {video_dir}, skipping saving and rendering.")
        return

    first_output = outputs_data[0]["output"]
    device = first_output["intrinsics"].device
    image_np_full = load_image(image_path)
    H, W = image_np_full.shape[:2]

    intrinsics = first_output["intrinsics"].to(device)
    intrinsics_np = denormalize_f(intrinsics.cpu().numpy(), H, W)
    with open(intrinsics_path, "w") as f:
        json.dump({"intrinsics": intrinsics_np.tolist()}, f, indent=4)
    print(f"Saved shared intrinsics to {intrinsics_path}")

    processed_outputs_for_viz = []
    for item in outputs_data:
        output = item["output"]
        mask_path = item["mask_path"]

        mesh_glb = output["glb"]
        rotation = output["rotation"].to(device)
        translation = output["translation"].to(device)
        scale = output["scale"].to(device)

        rotation_matrix = quaternion_to_matrix(rotation.squeeze(1))
        tfm_ori = compose_transform(scale=scale, rotation=rotation_matrix, translation=translation)
        mesh, _, _ = get_mesh(mesh_glb, tfm_ori, device)

        # Optional silhouette pose refinement (disabled unless --post_optimize).
        if post_optimize:
            print(f"Running post-optimization for mask: {mask_path}")
            mask = torch.from_numpy(load_mask(mask_path)).float().to(device)
            center = translation[0].clone()
            quat, new_translation, new_scale, R = run_render_compare(
                mesh, center, intrinsics_np, H, W, mask[None, None, ...], device
            )
            delta_transform = (
                Transform3d(device=device)
                .translate(-center.unsqueeze(0))
                .scale(new_scale.unsqueeze(0))
                .rotate(R.transpose(0, 1).unsqueeze(0))
                .translate(center.unsqueeze(0))
                .translate(new_translation.unsqueeze(0))
            )
        else:
            print("Post-optimization is disabled. Saving original transform.")
            delta_transform = Transform3d(device=device)

        final_transform = tfm_ori.compose(delta_transform)
        mask_idx = os.path.splitext(os.path.basename(mask_path))[0]
        transform_info_path = str(layout.object_init_dir / f"transform_{mask_idx}.json")
        with open(transform_info_path, "w") as f:
            json.dump({"transform": final_transform.get_matrix().squeeze(0).cpu().numpy().tolist()}, f, indent=4)
        print(f"Saved transformation info to {transform_info_path}")

        with torch.no_grad():
            new_verts = delta_transform.transform_points(mesh.verts_list()[0].unsqueeze(0))
            transformed_mesh = Meshes(verts=[new_verts.squeeze(0)], faces=mesh.faces_list())

        processed_outputs_for_viz.append({
            "mesh": mesh,
            "transformed_mesh": transformed_mesh,
            "color": item["color"],
        })

    output_render_path = str(layout.rendered_on_image)
    if processed_outputs_for_viz:
        visualize_render_on_image(
            processed_outputs_for_viz, image_path, intrinsics_np, H, W, output_path=output_render_path
        )
    else:
        print(f"No objects were processed for {video_dir}, skipping rendering.")


def worker_main_gpu(gpu_id, video_chunks, config_path, output_root_dir, post_optimize, args):
    """Per-GPU worker used by torch.multiprocessing.spawn."""
    video_dir_chunk = video_chunks[gpu_id]
    torch.cuda.set_device(gpu_id)
    print(f"Worker on GPU {gpu_id} started, processing {len(video_dir_chunk)} directories.")

    try:
        inference_model = Inference(config_path, compile=False)
        for video_dir in tqdm(video_dir_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
            try:
                process_video_directory(video_dir, inference_model, output_root_dir, post_optimize, args)
            except Exception as e:
                print(f"Error processing {video_dir} on GPU {gpu_id}: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"FATAL: Worker for GPU {gpu_id} failed during initialization: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Run 3D object reconstruction on one or more GPUs.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root with Yolov8 per-video folders (default: CHOIR output/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for writing glb/transforms (default: CHOIR output/).",
    )
    parser.add_argument("--debug", action="store_true", help="Single-process mode without multiprocessing.")
    parser.add_argument("--post_optimize", action="store_true", help="Enable silhouette pose post-optimization.")
    parser.add_argument("--video_id", type=str, default=None, nargs="+", help="Video ID(s) to process.")
    args = parser.parse_args()

    root_data_dir = args.data_dir
    output_root_dir = args.output_dir
    os.makedirs(output_root_dir, exist_ok=True)
    print(f"Reading videos from: {root_data_dir}")
    print(f"Saving all results to: {output_root_dir}")

    try:
        all_dirs = sorted(os.listdir(root_data_dir))
    except FileNotFoundError:
        print(f"Error: Data directory not found at '{root_data_dir}'")
        return

    video_dirs = [os.path.join(root_data_dir, d) for d in all_dirs if os.path.isdir(os.path.join(root_data_dir, d))]
    if args.video_id is not None:
        requested = set(args.video_id)
        video_dirs = [d for d in video_dirs if os.path.basename(d) in requested]

    if not video_dirs:
        print(f"No video directories found in '{root_data_dir}'.")
        return

    config_path = "checkpoints/hf/pipeline.yaml"

    if args.debug:
        print("--- RUNNING IN DEBUG MODE (single process, sequential) ---")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print("Debug mode is using GPU 0.")
        else:
            print("Debug mode is using CPU.")

        inference_model = Inference(config_path, compile=False)
        for video_dir in tqdm(video_dirs, desc="Debug Processing"):
            try:
                process_video_directory(video_dir, inference_model, output_root_dir, args.post_optimize, args)
            except Exception as e:
                print(f"Error processing {video_dir} in debug mode: {e}")
                traceback.print_exc()
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("Error: No GPUs found for multi-GPU mode. Use --debug to run on CPU.")
            return

        if num_gpus > len(video_dirs):
            print(f"Warning: More GPUs ({num_gpus}) than directories ({len(video_dirs)}). Using {len(video_dirs)} GPUs.")
            num_gpus = len(video_dirs)

        video_chunks = [[] for _ in range(num_gpus)]
        for i, video_dir in enumerate(video_dirs):
            video_chunks[i % num_gpus].append(video_dir)

        spawn_args = (video_chunks, config_path, output_root_dir, args.post_optimize, args)
        mp.spawn(worker_main_gpu, args=spawn_args, nprocs=num_gpus, join=True)

    print("\n--- All video directories processed. ---")


if __name__ == "__main__":
    main()

import argparse
import os
import traceback
import warnings

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from scipy.ndimage import binary_dilation
from torchvision import transforms
from tqdm import tqdm

from debug_bbox import get_global_amodal_bbox, load_hand_data, load_raw_frames
from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline

warnings.filterwarnings("ignore")


def init_amodal_segmentation_model(model_path_mask):
    device = f"cuda:{torch.cuda.current_device()}"
    pipeline_mask = DiffusionVASPipeline.from_pretrained(model_path_mask, torch_dtype=torch.float16).to(device)
    # pipeline_mask.enable_model_cpu_offload()
    pipeline_mask.set_progress_bar_config(disable=True)

    return pipeline_mask


def init_rgb_model(model_path_rgb):
    device = f"cuda:{torch.cuda.current_device()}"
    pipeline_rgb = DiffusionVASPipeline.from_pretrained(model_path_rgb, torch_dtype=torch.float16).to(device)
    # pipeline_rgb.enable_model_cpu_offload()
    pipeline_rgb.set_progress_bar_config(disable=True)

    return pipeline_rgb


def init_depth_model(model_path_depth, depth_encoder):
    device = f"cuda:{torch.cuda.current_device()}"

    from models.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

    depth_model_configs = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384]
        },
        'vitb': {
            'encoder': 'vitb',
            'features': 128,
            'out_channels': [96, 192, 384, 768]
        },
        'vitl': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        },
        'vitg': {
            'encoder': 'vitg',
            'features': 384,
            'out_channels': [1536, 1536, 1536, 1536]
        }
    }

    depth_model = DepthAnythingV2(**depth_model_configs[depth_encoder]).to(device)
    depth_model.load_state_dict(torch.load(model_path_depth, map_location=device))
    depth_model.eval()

    return depth_model


def get_raw_depth_maps(raw_rgbs, depth_model):
    """
    Computes depth maps from raw RGB images.
    Returns a list of single-channel float numpy arrays, normalized to [0, 1].
    """
    depth_maps = []
    for rgb_image_np in tqdm(raw_rgbs, desc="Estimating Depth"):
        # depth_model expects a (H, W, 3) uint8 numpy array
        depth_map = depth_model.infer_image(rgb_image_np)  # returns a (H, W) float numpy array
        depth_maps.append(depth_map)

    # Normalize across the entire video sequence
    depth_maps_np = np.array(depth_maps)
    min_val, max_val = depth_maps_np.min(), depth_maps_np.max()
    depth_maps_np = (depth_maps_np - min_val) / (max_val - min_val)

    return depth_maps_np


def crop_and_resize_frames(frames, bboxes, output_size, frame_type='rgb'):
    to_tensor = transforms.ToTensor()
    normalizer = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

    processed_frames = []
    for frame, bbox in zip(frames, bboxes):
        x1, y1, x2, y2 = bbox

        crop_w = x2 - x1
        crop_h = y2 - y1
        img_h, img_w = frame.shape[:2]

        src_x1, src_y1 = max(0, x1), max(0, y1)
        src_x2, src_y2 = min(img_w, x2), min(img_h, y2)

        dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
        dst_x2, dst_y2 = src_x2 - x1, src_y2 - y1

        if frame.ndim == 3:
            canvas = np.zeros((crop_h, crop_w, frame.shape[2]), dtype=frame.dtype)
        else:
            canvas = np.zeros((crop_h, crop_w), dtype=frame.dtype)

        if (src_x2 > src_x1) and (src_y2 > src_y1):
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

        # Pre-resize processing
        if frame_type == 'mask':
            # Binarized 0/1 mask to 0/255 for better resizing interpolation
            canvas = (canvas * 255).astype(np.uint8)

        # --- New resize logic to preserve aspect ratio ---
        h_canvas, w_canvas = canvas.shape[:2]
        h_out, w_out = output_size

        # Calculate scale to fit canvas into output_size while preserving aspect ratio
        scale = min(w_out / w_canvas, h_out / h_canvas)
        new_w, new_h = int(w_canvas * scale), int(h_canvas * scale)

        # Resize with aspect ratio preserved
        interpolation = cv2.INTER_LINEAR if frame_type != 'mask' else cv2.INTER_NEAREST
        resized_canvas = cv2.resize(canvas, (new_w, new_h), interpolation=interpolation)

        # Create a new canvas of the final output size and paste the resized image in the center
        if resized_canvas.ndim == 3:
            final_image = np.zeros((h_out, w_out, resized_canvas.shape[2]), dtype=canvas.dtype)
        else:
            final_image = np.zeros((h_out, w_out), dtype=canvas.dtype)

        pad_top = (h_out - new_h) // 2
        pad_left = (w_out - new_w) // 2

        final_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_canvas

        # Post-resize processing & tensor conversion
        if frame_type in ['rgb', 'mask']:
            pil_image = Image.fromarray(final_image.astype(np.uint8))
            tensor_frame = to_tensor(pil_image)
        elif frame_type == 'depth':
            # Add channel dimension for to_tensor. Input is float [0,1]
            tensor_frame = to_tensor(final_image[:, :, np.newaxis])

        if frame_type != 'rgb':
            tensor_frame = tensor_frame.repeat(3, 1, 1)

        # Normalize from [0, 1] to [-1, 1]
        transformed_frame = normalizer(tensor_frame)

        processed_frames.append(transformed_frame)

    return torch.stack(processed_frames).unsqueeze(0)


def overlay_mask_on_image(rgb_img, mask, cmap_idx=None, random_color=False, boundary_thickness=3, darken_factor=2):
    # Ensure the input image is RGB and in the range [0, 1]
    assert rgb_img.shape[-1] == 3, "Expected RGB image with 3 channels"
    # assert rgb_img.min() >= 0 and rgb_img.max() <= 1, "Expected rgb_img values in the range [0, 1]"

    # Select a color for the mask overlay
    cmap = plt.get_cmap("tab10")

    cmap_idx = 4
    if cmap_idx is None and random_color:
        cmap_idx = np.random.randint(0, cmap.N)  # Randomly choose a colormap index if not provided

    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    boundary_color = color[:3] * darken_factor  # Darken the color by the darken_factor
    boundary_color = np.concatenate([boundary_color, [1.0]])  # Make boundary fully opaque

    # Create a boundary mask
    dilated_mask = binary_dilation(mask, iterations=boundary_thickness)
    boundary_mask = dilated_mask & ~mask

    # Create a colored mask in the range [0, 1]
    mask_image = np.zeros_like(rgb_img, dtype=np.float32)
    boundary_image = np.zeros_like(rgb_img, dtype=np.float32)

    for i in range(3):  # Apply the mask and boundary to each channel
        mask_image[..., i] = mask * color[i]
        boundary_image[..., i] = boundary_mask * boundary_color[i]

    # Combine the RGB image with the colored mask and boundary
    overlayed_image = np.clip(rgb_img * 0.5 + mask_image + boundary_image, 0, 1)

    return overlayed_image


def process_sequence(seq_path, data_output_path, pipeline_mask, pipeline_rgb, depth_model, generator, fps=30):
    seq_name = os.path.basename(seq_path)
    os.makedirs(data_output_path, exist_ok=True)

    # # output video paths
    # pred_amodal_masks_path = f"{output_seq_path}/pred_amodal_masks.mp4"
    # pred_amodal_rgb_path = f"{output_seq_path}/pred_amodal_rgb.mp4"
    # pred_amodal_rgb_overlay_path = f"{output_seq_path}/pred_amodal_rgb_overlay.mp4"

    # load input modal masks and rgb images
    pred_res = (256, 512)  # sometimes a higher resolution (e.g.,512x1024) might produce better results

    # --- New data processing pipeline ---
    # 1. Load raw frames without transformations
    raw_masks = load_raw_frames(seq_path + "/obj_masks", is_mask=True)
    raw_rgbs = load_raw_frames(seq_path + "/rgbs", is_mask=False)
    num_frames = len(raw_masks)

    # 2. Load hand bboxes and get smoothed amodal bbox
    hand_bboxes = load_hand_data(seq_path, num_frames)
    bboxes = get_global_amodal_bbox(raw_masks, hand_bboxes)

    # 3. Estimate depth from ORIGINAL rgbs
    raw_depths = get_raw_depth_maps(raw_rgbs, depth_model)

    # 4. Crop, resize, and transform all frames based on bboxes
    modal_pixels = crop_and_resize_frames(raw_masks, bboxes, pred_res, frame_type='mask')
    rgb_pixels = crop_and_resize_frames(raw_rgbs, bboxes, pred_res, frame_type='rgb')
    depth_pixels = crop_and_resize_frames(raw_depths, bboxes, pred_res, frame_type='depth')

    # --- End of new data processing pipeline ---

    print("amodal segmentation by diffusion-vas ...")
    # predict amodal masks (amodal segmentation)
    pred_amodal_masks = pipeline_mask(
        modal_pixels,
        depth_pixels,
        height=pred_res[0],
        width=pred_res[1],
        num_frames=num_frames,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=8,
        noise_aug_strength=0.02,
        min_guidance_scale=1.5,
        max_guidance_scale=1.5,
        generator=generator,
    ).frames[0]

    pred_amodal_masks = [np.array(img) for img in pred_amodal_masks]

    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')

    # --- New Overlay Logic: Visualize on cropped frames ---

    # The predicted amodal masks are for the CROPPED region.
    # We combine it with the original modal mask (also cropped) for consistency.
    modal_mask_union_cropped = (modal_pixels[0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
    pred_amodal_masks = np.logical_or(pred_amodal_masks, modal_mask_union_cropped).astype('uint8')

    # pred_amodal_masks_tensor = torch.from_numpy(np.where(pred_amodal_masks == 0, -1, 1)).float().unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1)

    # Convert the cropped rgb_pixels and modal_pixels tensors back to a displayable format for visualization
    # The rgb_pixels tensor is in [-1, 1], so we scale it to [0, 1] for overlaying.
    cropped_rgbs_for_viz = (rgb_pixels.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0
    # The modal_pixels tensor is also in [-1, 1], so we check for positive values to get the binary mask.
    cropped_modal_masks_for_viz = (modal_pixels.squeeze(0)[:, 0, :, :].cpu().numpy() > 0).astype(np.uint8)

    tmp_cmap_idx = np.random.randint(0, plt.get_cmap("tab10").N)

    # --- New Combined Video Logic: Modal (Left) vs Amodal (Right) ---
    comparison_video_path = f"{data_output_path}/{seq_name}.mp4"

    combined_frames = []
    for i in range(num_frames):
        # Generate modal overlay frame (left side)
        modal_overlay_frame = overlay_mask_on_image(cropped_rgbs_for_viz[i], cropped_modal_masks_for_viz[i], cmap_idx=tmp_cmap_idx)

        # Generate amodal overlay frame (right side)
        amodal_overlay_frame = overlay_mask_on_image(cropped_rgbs_for_viz[i], pred_amodal_masks[i].astype(np.uint8), cmap_idx=tmp_cmap_idx)

        # Concatenate side-by-side
        combined_frame = np.hstack((modal_overlay_frame, amodal_overlay_frame))
        combined_frames.append(combined_frame)

    # Convert to uint8 and save the combined video
    combined_frames_np = np.stack(combined_frames, axis=0)
    imageio.mimwrite(
        comparison_video_path,
        (combined_frames_np * 255).astype(np.uint8),
        format='ffmpeg',
        fps=float(fps),
        macro_block_size=None,
        ffmpeg_params=['-crf', '28', '-preset', 'veryfast'],
        codec='libx264',
        pixelformat='yuv420p',
    )
    print(f"Saved comparison video to {comparison_video_path}")

    # # save modal_rgb
    # modal_rgb_pixels = rgb_pixels * modal_obj_mask + modal_background
    # modal_rgb_pixels_save = np.array(
    #     [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR) for frame in modal_rgb_pixels[0].cpu().numpy().transpose(0, 2, 3, 1)])
    # imageio.mimwrite(modal_rgb_path, (modal_rgb_pixels_save * 255).astype(np.uint8), format='ffmpeg', fps=float(fps), macro_block_size=macro_block_size, quality=quality, codec='libx264', pixelformat='yuv420p')

    # modal_rgb_pixels = modal_rgb_pixels * 2 - 1

    # print("content completion by diffusion-vas ...")
    # # predict amodal rgb (content completion)
    # pred_amodal_rgb = pipeline_rgb(
    #     modal_rgb_pixels,
    #     pred_amodal_masks_tensor,
    #     height=pred_res[0],  # my_res[0]
    #     width=pred_res[1],  # my_res[1]
    #     num_frames=num_frames,
    #     decode_chunk_size=8,
    #     motion_bucket_id=127,
    #     fps=fps,
    #     noise_aug_strength=0.02,
    #     min_guidance_scale=1.5,
    #     max_guidance_scale=1.5,
    #     generator=generator,
    # ).frames[0]

    # pred_amodal_rgb = [np.array(img) for img in pred_amodal_rgb]

    # # save pred_amodal_rgb
    # pred_amodal_rgb = np.array(pred_amodal_rgb).astype('uint8')
    # pred_amodal_rgb_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR) for frame in pred_amodal_rgb])
    # imageio.mimwrite(pred_amodal_rgb_path, pred_amodal_rgb_save, format='ffmpeg', fps=float(fps), macro_block_size=macro_block_size, quality=quality, codec='libx264', pixelformat='yuv420p')

    # # save pred_amodal_rgb_overlay
    # transparency_factor = 0.5
    # white_background = np.ones_like(raw_rgb_pixels) * 255
    # raw_rgb_semi_transparent = np.clip(raw_rgb_pixels * transparency_factor + white_background * (1 - transparency_factor), 0, 255).astype(np.uint8)
    # pred_amodal_rgb_overlay = np.where(pred_amodal_masks_save[..., None] == 1, pred_amodal_rgb_save, raw_rgb_semi_transparent)
    # imageio.mimwrite(pred_amodal_rgb_overlay_path, pred_amodal_rgb_overlay, format='ffmpeg', fps=float(fps), macro_block_size=macro_block_size, quality=quality, codec='libx264', pixelformat='yuv420p')

    # # save modal_rgb_overlay
    # modal_pixels = np.array(
    #     [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST) for frame in modal_pixels[0].cpu().numpy().transpose(0, 2, 3, 1)])
    # modal_rgb_overlay = np.where(np.array((modal_pixels > 0)[:, :, :, :]) == 1, raw_rgb_pixels, raw_rgb_semi_transparent)
    # imageio.mimwrite(modal_rgb_overlay_path, modal_rgb_overlay, format='ffmpeg', fps=float(fps), macro_block_size=macro_block_size, quality=quality, codec='libx264', pixelformat='yuv420p')


def worker_main_gpu(gpu_id, seq_path_chunks, args):
    torch.cuda.set_device(gpu_id)
    print(f"Worker on GPU {gpu_id} started, processing {len(seq_path_chunks[gpu_id])} sequences.")

    # Each worker process loads its own model instance onto its assigned GPU
    pipeline_mask = init_amodal_segmentation_model(args.model_path_mask)
    # pipeline_rgb = init_rgb_model(args.model_path_rgb)
    pipeline_rgb = None
    depth_model = init_depth_model(args.model_path_depth + f"/depth_anything_v2_{args.depth_encoder}.pth", args.depth_encoder)
    generator = torch.manual_seed(23)

    for seq_path in tqdm(seq_path_chunks[gpu_id], desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            process_sequence(seq_path, args.data_output_path, pipeline_mask, pipeline_rgb, depth_model, generator, fps=30)
        except Exception as e:
            print(f"Error processing {os.path.basename(seq_path)} on GPU {gpu_id}: {e}")
            traceback.print_exc()


def main(args):
    data_path = args.data_path
    subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]

    if args.video_id:
        print(f"--- Processing only specified video IDs: {args.video_id} ---")
        target_ids = set(args.video_id.split(','))
        subfolders = [p for p in subfolders if os.path.basename(p) in target_ids]
        print(f"Found {len(subfolders)} matching video sequences to process.")

    if not subfolders:
        print("Warning: No video sequences found to process with the given criteria.")
        return

    if args.debug:
        print("--- RUNNING IN DEBUG MODE (single process, sequential) ---")
        torch.cuda.set_device(0)

        pipeline_mask = init_amodal_segmentation_model(args.model_path_mask)
        pipeline_rgb = init_rgb_model(args.model_path_rgb)
        depth_model = init_depth_model(args.model_path_depth + f"/depth_anything_v2_{args.depth_encoder}.pth", args.depth_encoder)
        generator = torch.manual_seed(23)

        for seq_path in tqdm(subfolders, desc="Debug Processing"):
            process_sequence(seq_path, args.data_output_path, pipeline_mask, pipeline_rgb, depth_model, generator, fps=30)

    else:
        # --- MULTI-GPU MODE ---
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("Error: No GPUs found for multi-GPU mode. Use --debug to run on single process.")
            return

        if num_gpus > len(subfolders):
            print(f"Warning: More GPUs ({num_gpus}) than directories ({len(subfolders)}). Using {len(subfolders)} GPUs.")
            num_gpus = len(subfolders)

        seq_path_chunks = [[] for _ in range(num_gpus)]
        for i, seq_path in enumerate(subfolders):
            seq_path_chunks[i % num_gpus].append(seq_path)

        spawn_args = (seq_path_chunks, args)
        mp.spawn(worker_main_gpu, args=spawn_args, nprocs=num_gpus, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video amodal segmentation and content completion using Diffusion-VAS.")

    parser.add_argument(
        "--model_path_mask",
        type=str,
        default="checkpoints/diffusion-vas-amodal-segmentation",
        help="Path to diffusion-vas amodal segmentation checkpoint.",
    )

    parser.add_argument(
        "--model_path_rgb",
        type=str,
        default="checkpoints/diffusion-vas-content-completion",
        help="Path to diffusion-vas content completion checkpoint.",
    )

    parser.add_argument(
        "--depth_encoder",
        type=str,
        default="vitl",  # or 'vits', vitl, 'vitg'
        help="Depth encoder type.",
    )

    parser.add_argument(
        "--model_path_depth",
        type=str,
        default="checkpoints/",
        help="Path to depth anything v2's checkpoint's parent folder.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="../../output",
        help="Path to the parent directory containing sequence subfolders.",
    )

    parser.add_argument(
        "--data_output_path",
        type=str,
        default="outputs",
        help="Output path.",
    )

    parser.add_argument('--video_id', type=str, default=None, help="Comma-separated video IDs to process.")
    parser.add_argument('--debug', action='store_true', help='Run in single-process debug mode without multiprocessing.')

    args = parser.parse_args()

    main(args)

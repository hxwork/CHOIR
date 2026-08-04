from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
from typing import NamedTuple
import math
from glob import glob
import pickle
import cv2
import os.path as op
import shutil
import subprocess

class CameraInfo(NamedTuple):
    uid: int
    c2w4x4: np.array
    fl_x: np.array
    fl_y: np.array
    height: int
    width: int
    fov_x: float
    fov_y: float
    cx: float
    cy: float
    rgb_name: str
    rgba_name: str
    mask_name: str
    pts3d: np.array

def save_json(json_f, data, indent=4):
    with open(json_f, 'w') as file:
        json.dump(data, file, indent=indent)

def preprocessCamerasFromBlenderJson(config):  
    path = config.json_path
    transformsfile = config.json_file

    images_dir = os.path.join(path, "rgbs")
    masks_dir = os.path.join(path, "masks")

    os.makedirs(path, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)        
    with open(os.path.join(path, transformsfile)) as json_file:
        meta = json.load(json_file)
        for idx, frame in enumerate(meta["frames"]):
            rgba_f = os.path.join(path, frame["file_path"])
            if not rgba_f:
                num_skipped_image_filenames += 1
            else:
                rgba_image = cv2.imread(rgba_f, cv2.IMREAD_UNCHANGED)
                rgb_image = rgba_image[:, :, :3]
                mask_image = rgba_image[:, :, 3]
                _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
                rgba_base_name = os.path.basename(rgba_f)
                rgb_f = os.path.join(images_dir, rgba_base_name)
                mask_f = os.path.join(masks_dir, rgba_base_name)
                cv2.imwrite(rgb_f, rgb_image)
                cv2.imwrite(mask_f, binary_mask)
                print(f"Writing {rgb_f} and {mask_f}")



def readCamerasFromBlenderJson(config):
    # Read the camera information from the transforms.json file
    cam_type = config.cam_type.lower() # "cvc2cvw" or "cvc2blw" or "blc2blw"
    path = config.json_path
    transformsfile = config.json_file
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        meta = json.load(json_file)
        num_skipped_image_filenames = 0
        fl_x, fl_y = get_focal_lengths(meta)
        width = meta["w"]
        height = meta["h"]
        fov_x = math.atan(width / (2 * fl_y)) * 2
        fov_y = math.atan(height / (2 * fl_y)) * 2
        cx, cy = (width -1) * 0.5, (height -1) * 0.5
        cvc2blc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        blw2cvw = np.array([[0, 1, 0, 0],[0, 0 , -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        blc2cvc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cvw2blw = np.array([[0, 0, -1, 0],[1, 0 , 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        blc2glc = np.array([[1, 0, 0, 0],[0, 1 , 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        glw2blw = np.array([[0, 0, 1, 0],[1, 0 , 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        for idx, frame in enumerate(meta["frames"]):
            rgba_f = os.path.join(path, frame["file_path"])
            rgba_base_name = os.path.basename(rgba_f)
            rgb_f = os.path.join(path, "rgbs", rgba_base_name)
            mask_f = os.path.join(path, "masks", rgba_base_name)
            if not rgba_f:
                num_skipped_image_filenames += 1
            else:
                blc2blw = np.array(frame["transform_matrix"]) # frame["transform_matrix"] is camera.matrix_world in Blender format from the camera to the world
                blw2blc = np.linalg.inv(blc2blw)
                # c2w_cv = gl_to_cv_t @ c2w_gl
                if cam_type == "glw2glc":
                    glw2glc = blc2glc @ blw2blc @ glw2blw
                    c2w_final = glw2glc
                elif cam_type == "cvw2cvc":
                    cvw2cvc = blc2cvc @ blw2blc @ cvw2blw
                    c2w_final = cvw2cvc
                elif cam_type == "cvc2cvw":
                    cvc2cvw = blw2cvw @ blc2blw @ cvc2blc
                    c2w_final = cvc2cvw
                elif cam_type == "cvc2blw":
                    cvc2blw = blc2blw @ cvc2blc
                    c2w_final = cvc2blw
                elif cam_type == "blc2blw":
                    c2w_final = blc2blw
                else:
                    assert "Unknown camera type"
                cam_infos.append(CameraInfo(uid=idx, c2w4x4=c2w_final, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, scale_inpaint=None,
                                            rgb_name=rgb_f, rgba_name=rgba_f, mask_name=mask_f, pts3d=None,
                                            height=height, width=width, fov_x=fov_x, fov_y=fov_y))
        if num_skipped_image_filenames >= 0:
            CONSOLE.print(f"Skipping {num_skipped_image_filenames} files in dataset {path}.")
        assert (
            len(cam_infos) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """               
            
    return cam_infos

def get_focal_lengths(meta):
    """Reads or computes the focal length from transforms dict.
    Args:
        meta: metadata from transforms.json file.
    Returns:
        Focal lengths in the x and y directions. Error is raised if these cannot be calculated.
    """
    fl_x, fl_y = 0, 0

    def fov_to_focal_length(rad, res):
        return 0.5 * res / np.tan(0.5 * rad)

    if "fl_x" in meta:
        fl_x = meta["fl_x"]
    elif "x_fov" in meta:
        fl_x = fov_to_focal_length(np.deg2rad(meta["x_fov"]), meta["w"])
    elif "camera_angle_x" in meta:
        fl_x = fov_to_focal_length(meta["camera_angle_x"], meta["w"])

    if "fl_y" in meta:
        fl_y = meta["fl_y"]
    elif "y_fov" in meta:
        fl_y = fov_to_focal_length(np.deg2rad(meta["y_fov"]), meta["h"])
    elif "camera_angle_y" in meta:
        fl_y = fov_to_focal_length(meta["camera_angle_y"], meta["h"])

    if fl_x == 0 or fl_y == 0:
        raise AttributeError("Focal length cannot be calculated from transforms.json (missing fields).")

    return (fl_x, fl_y)

def PreprocessHO3DFoundationPose(config):
    rgb_all = sorted(glob(f"{config['rgb_path']}/*.jpg"))
    depth_all = sorted(glob(f"{config['depth_path']}/*.png"))
    pose_all = sorted(glob(f"{config['pose_path']}/*.txt"))
    mask_all = sorted(glob(f"{config['mask_path']}/*.png"))
    intrinsic_f = rgb_all[0].replace("rgb", "meta").replace("jpg", "pkl")
    out_dir = config['out_dir']
    images_dir = os.path.join(out_dir, "images")
    depths_dir = os.path.join(out_dir, "depths")
    rgbas_dir = os.path.join(out_dir, "rgbas")
    masks_dir = os.path.join(out_dir, "masks")
    poses_dir = os.path.join(out_dir, "cameras")
    intrinsic_dir = os.path.join(out_dir, "intrinsic")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depths_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(rgbas_dir, exist_ok=True)
    os.makedirs(intrinsic_dir, exist_ok=True)
    if config.end_frame == -1:
        end_fname = rgb_all[-1]
        end = int(op.basename(end_fname).split(".")[0].split("_rgba")[0])
        config.end_frame = end
    assert config.end_frame > config.start_frame    

    frame_range = list(range(config.start_frame, config.end_frame, config.frame_interval))
    valid_frames = []

    for i in frame_range:
        rgb_f = os.path.join(config['rgb_path'], f"{i:04d}.jpg")
        if (rgb_f in rgb_all):
            pose_f = os.path.join(config['pose_path'], f"{i:04d}.txt")
            if pose_f in pose_all:
                mask_f = os.path.join(config['mask_path'], f"{i:05d}.png")
                if mask_f in mask_all:
                    depth_f = os.path.join(config['depth_path'], f"{i:04d}.png")
                    if depth_f in depth_all:
                        if i not in config.exclude_frames:
                            valid_frames.append([rgb_f, pose_f, mask_f, depth_f])
                    else:
                        print(f"Warning: {depth_f} not found")
                else:
                    print(f"Warning: {mask_f} not found")
            else:
                print(f"Warning: {pose_f} not found")
        else:
            print(f"Warning: {rgb_f} not found")
    assert len(valid_frames) > 0

    meta = pickle.load(open(intrinsic_f,'rb'))
    K = meta['camMat']
    fmt = '%.12f'
    np.savetxt(f'{intrinsic_dir}/intrins.txt', K, fmt=fmt, delimiter=' ')
    camera_param = {}
    for ci, [rgb_f, pose_f, mask_f, depth_f] in enumerate(valid_frames):
        rgb_f_out = os.path.join(images_dir, f"{ci:04d}.png")
        depth_f_out = os.path.join(depths_dir, f"{ci:04d}.png")
        mask_f_out = os.path.join(masks_dir, f"{ci:04d}.png")

        cv2.imwrite(rgb_f_out, cv2.imread(rgb_f))
        shutil.copy(depth_f, depth_f_out)
        shutil.copy(mask_f, mask_f_out)
        pose_f_out = os.path.join(poses_dir, f"{ci:04d}.json")
        camera_param['blw2cvc'] = np.loadtxt(pose_f)
        camera_param['K'] = K
        height, width = cv2.imread(rgb_f).shape[:2]
        camera_param['height'] = np.array(height)
        camera_param['width'] = np.array(width)
        camera_param_serializable = {key: value.tolist() for key, value in camera_param.items()}
        save_json(pose_f_out, camera_param_serializable)


        rgb_imgae = cv2.imread(rgb_f)
        mask_image = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask[:, :, np.newaxis]
        masked_image = rgb_imgae * (binary_mask // 255) + 255 * (1 - binary_mask// 255)
        alpha_channel = binary_mask
        rgba_image = cv2.merge((masked_image[:, :, 0], masked_image[:, :, 1], masked_image[:, :, 2], alpha_channel))
        rgba_f = os.path.join(rgbas_dir, f"{ci:04d}.png")
        cv2.imwrite(rgba_f, rgba_image)

    json_file = os.path.join(out_dir, "map.json")
    print(f"Writing to {json_file}")
    with open(json_file, 'w') as f:
        for i, entry in enumerate(valid_frames):
            rgb_f, pose_f, mask_f, depth_f = entry
            f.write(f"{i:04d} {rgb_f} {mask_f} {pose_f} {depth_f}\n")        
# after PreprocessHO3DFoundationPose is called, the following code can be used to read the camera information
def ReadHO3DFoundationPose(config):
    cam_type = config.cam_type.lower() # "cvc2blw" or "blc2blw"
    cam_infos = []
    rgb_all = sorted(glob(f"{config['rgb_path']}/*.png"))
    camera_all = sorted(glob(f"{config['camera_path']}/*.json"))
    # mask_all = sorted(glob(f"{config['mask_path']}/*.png"))
    rgba_all = sorted(glob(f"{config['rgba_path']}/*.png"))

    K = None
    cam_infos = [] 
    # for ci, [rgb_f, rgba_f, mask_f, camera_f] in enumerate(zip(rgb_all, rgba_all, mask_all, camera_all)):
    for ci, camera_f in enumerate(camera_all):
        camera_f_ind = op.basename(camera_f).split(".")[0]
        rgb_f = op.join(config['rgb_path'], f"{camera_f_ind}.png")
        rgba_f = op.join(config['rgba_path'], f"{camera_f_ind}.png")
        mask_f = op.join(config['mask_path'], f"{camera_f_ind}.png")
        if not (rgb_f in rgb_all):
            print(f"Error: {rgb_f} not found")
            assert False
        if not (rgba_f in rgba_all):
            print(f"Error: {rgba_f} not found")
            assert False
        # if not (mask_f in mask_all):
        #     print(f"Error: {mask_f} not found")
        #     assert False
        camera = json.load(open(camera_f, 'r'))
        # K = camera['K_inpaint']
        K = camera['K']
        # K = camera['K_manual']
        # K = camera['K_half_wh']
        fl_x, fl_y = K[0][0], K[1][1]
        cx, cy = int(K[0][2]), int(K[1][2])
        height, width = camera['height'], camera['width']
        fov_x = math.atan(cx / (fl_x)) * 2
        fov_y = math.atan(cy / (fl_y)) * 2
        blc2cvc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cvc2blw = np.linalg.inv(camera['blw2cvc'])
        if cam_type == "blc2blw":
            blc2blw = cvc2blw @ blc2cvc
            c2w_final = blc2blw
        elif cam_type == "cvc2blw":
            c2w_final = cvc2blw
        else:
            assert "Unknown camera type"
        cam_infos.append(CameraInfo(uid=camera_f_ind, c2w4x4=c2w_final, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, 
                                    rgb_name=rgb_f, rgba_name=rgba_f, mask_name=mask_f, pts3d=None,
                                    height=height, width=width, fov_x=fov_x, fov_y=fov_y))
        
    return cam_infos

def PreprocessHO3DGTPose(config):
    rgb_all = sorted(glob(f"{config['rgb_path']}/*.jpg"))
    pose_all = sorted(glob(f"{config['pose_path']}/*.pkl"))
    mask_all = sorted(glob(f"{config['mask_path']}/*.png"))
    out_dir = config['out_dir']
    images_dir = os.path.join(out_dir, "images")
    rgbas_dir = os.path.join(out_dir, "rgbas")
    masks_dir = os.path.join(out_dir, "masks")
    poses_dir = os.path.join(out_dir, "poses")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(rgbas_dir, exist_ok=True)

    if config.end_frame == -1:
        end_fname = rgb_all[-1]
        end = int(op.basename(end_fname).split(".")[0].split("_rgba")[0])
        config.end_frame = end
    assert config.end_frame > config.start_frame    

    frame_range = list(range(config.start_frame, config.end_frame, config.frame_interval))
    valid_frames = []

    for i in frame_range:
        rgb_f = os.path.join(config['rgb_path'], f"{i:04d}.jpg")
        if (rgb_f in rgb_all):
            pose_f = os.path.join(config['pose_path'], f"{i:04d}.pkl")
            if pose_f in pose_all:
                mask_f = os.path.join(config['mask_path'], f"{i:05d}.png")
                if mask_f in mask_all:
                    if i not in config.exclude_frames:
                        meta = pickle.load(open(pose_f,'rb'))
                        if meta['objTrans'] is None or meta['objRot'] is None or meta['camMat'] is None:
                            print(f"Warning: Pose not found for Meta file {pose_f}, and skip frame {rgb_f}")
                            continue

                        valid_frames.append([rgb_f, pose_f, mask_f])
                else:
                    print(f"Warning: {mask_f} not found")
            else:
                print(f"Warning: {pose_f} not found")
        else:
            print(f"Warning: {rgb_f} not found")
    assert len(valid_frames) > 0

    for ci, [rgb_f, pose_f, mask_f] in enumerate(valid_frames):
        rgb_f_out = os.path.join(images_dir, f"{ci:04d}.png")
        mask_f_out = os.path.join(masks_dir, f"{ci:04d}.png")
        pose_f_out = os.path.join(poses_dir, f"{ci:04d}.pkl")
        cv2.imwrite(rgb_f_out, cv2.imread(rgb_f))
        shutil.copy(mask_f, mask_f_out)
        shutil.copy(pose_f, pose_f_out)

        rgb_imgae = cv2.imread(rgb_f)
        mask_image = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask[:, :, np.newaxis]
        masked_image = rgb_imgae * (binary_mask // 255) + 255 * (1 - binary_mask// 255)
        alpha_channel = binary_mask
        rgba_image = cv2.merge((masked_image[:, :, 0], masked_image[:, :, 1], masked_image[:, :, 2], alpha_channel))
        rgba_f = os.path.join(rgbas_dir, f"{ci:04d}.png")
        cv2.imwrite(rgba_f, rgba_image)

    
    json_file = os.path.join(out_dir, "map.json")
    print(f"Writing to {json_file}")
    with open(json_file, 'w') as f:
        for i, entry in enumerate(valid_frames):
            rgb_f, pose_f, mask_f = entry
            f.write(f"{i:04d} {rgb_f} {mask_f} {pose_f}\n")        
# after preprocessCamerasFromRealImageWithGtPose is called, the following code can be used to read the camera information
def ReadHO3DGTPose(config):
    cam_type = config.cam_type.lower() # "cvc2cvw" or "cvc2blw" or "blc2blw"
    cam_infos = []
    rgb_all = sorted(glob(f"{config['rgb_path']}/*.png"))
    pose_all = sorted(glob(f"{config['pose_path']}/*.pkl"))
    mask_all = sorted(glob(f"{config['mask_path']}/*.png"))
    rgba_all = sorted(glob(f"{config['rgba_path']}/*.png"))

    K = None
    cam_infos = [] 
    for ci, [rgb_f, rgba_f, mask_f, pose_f] in enumerate(zip(rgb_all, rgba_all, mask_all, pose_all)):
        meta = pickle.load(open(pose_f,'rb'))
        if meta['objTrans'] is None or meta['objRot'] is None or meta['camMat'] is None:
            print(f"Warning: Pose not found for Meta file {pose_f}, and skip frame {rgba_f}")
            assert False
        if ci == 0:
            K = meta['camMat']
            fl_x, fl_y = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            rgba_imgae = cv2.imread(rgba_f)
            height, width = rgba_imgae.shape[:2]
            fov_x = math.atan(width / (2 * fl_x)) * 2
            fov_y = math.atan(height / (2 * fl_y)) * 2
            cvw2blw = np.array([[0, 0, -1, 0],[1, 0 , 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            cvw2glw = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            glc2cvc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            blc2cvc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cvw2glc = np.eye(4)
        cvw2glc[:3,3] = meta['objTrans']
        cvw2glc[:3,:3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
        cvw2cvc = glc2cvc @ cvw2glc
        cvc2cvw = np.linalg.inv(cvw2cvc)
        if cam_type == "blc2blw":
            blc2blw = cvw2blw @ cvc2cvw @ blc2cvc
            c2w_final = blc2blw
        elif cam_type == "glc2glw":
            glc2glw = cvw2glw @ cvc2cvw @ glc2cvc
            c2w_final = glc2glw
        elif cam_type == "cvc2cvw":
            c2w_final = cvc2cvw
        else:
            assert "Unknown camera type"
        cam_infos.append(CameraInfo(uid=ci, c2w4x4=c2w_final, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, scale_inpaint=None,
                                    rgb_name=rgb_f, rgba_name=rgba_f, mask_name=mask_f, pts3d=None,
                                    height=height, width=width, fov_x=fov_x, fov_y=fov_y))
        
    return cam_infos

def show_mask_and_bbox(mask, bbox):
    # input mask is a numpy array
    # input bbox is a tuple (x1, y1, x2, y2)
    import matplotlib.pyplot as plt
    plt.imshow(mask)
    # plot the bbox as rectangle
    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, color="red", linewidth=2))
    plt.show()

def inpaint_input_views(config, do_inpaint=True, do_mask=True, do_center=True, write_pose=True):
    if config.inpaint_select_strategy == "manual":
        inpaint_f = Path(config.image_dir) / f"{config.cond_view:04d}.png"
    else:
        max_value = 0
        hand_bbox_f = Path(config.image_dir) / "../boxes.npy"
        hand_bboxes = np.load(hand_bbox_f)
        valid_index = -1
        for i, ref_view in enumerate(config.ref_views):
            image_f = Path(config.image_dir) / f"{ref_view:04d}.png"
            # check image_f exists
            if not image_f.exists():
                continue
            valid_index += 1
            mask_f = image_f.parent.parent / "masks" / f"{ref_view:04d}.png"
            mask = cv2.imread(str(mask_f), cv2.IMREAD_UNCHANGED)
            SEGM_IDS = {"bg": 0, "object": 50, "right": 150, "left": 250}
            object_pixels = (mask == SEGM_IDS["object"]).sum()
            hand_pixels = (mask == SEGM_IDS["right"]).sum()
            if config.inpaint_select_strategy == "object_hand_ratio":
                # object_hand_ratio = object_pixels / (hand_pixels in the bbox range)
                # hand_bboxes is a list of bboxes, each bbox is a tuple (x1, y1, x2, y2)
                hand_bbox = hand_bboxes[valid_index].copy()
                # clip the bbox x1, x2 to image width, y1, y2 to image height
                mask_height, mask_width = mask.shape[:2]
                hand_bbox[0] = np.clip(hand_bbox[0], 0, mask_width - 1)
                hand_bbox[2] = np.clip(hand_bbox[2], 0, mask_width - 1)
                hand_bbox[1] = np.clip(hand_bbox[1], 0, mask_height - 1)
                hand_bbox[3] = np.clip(hand_bbox[3], 0, mask_height - 1)
                hand_bbox = hand_bbox.astype(np.int32)
                hand_pixels_in_bbox = (mask[hand_bbox[1]:hand_bbox[3], hand_bbox[0]:hand_bbox[2]] == SEGM_IDS["right"]).sum()
                # show the mask and the bbox by plotting
                # show_mask_and_bbox(mask, hand_bbox)
                object_hand_ratio = object_pixels / hand_pixels_in_bbox
                if object_hand_ratio > max_value:
                    max_value = object_hand_ratio
                    inpaint_f = image_f
            elif config.inpaint_select_strategy == "object_pixel_max":
                if object_pixels > max_value:
                    max_value = object_pixels
                    inpaint_f = image_f
            else:
                # assert and print error message
                assert False, "Unknown inpaint_select_strategy"
    print(f"Selected inpaint view: {inpaint_f}")
    # save the inpaint file index to config.inpaint_select_strategy.txt
    inpaint_f_index = os.path.basename(inpaint_f).split(".")[0].split("_rgba")[0]
    inpaint_f_index_f = f"{config.out_dir}/{config.inpaint_select_strategy}_selected.txt"
    os.makedirs(os.path.dirname(inpaint_f_index_f), exist_ok=True)
    with open(inpaint_f_index_f, "w") as f:
        f.write(f"{inpaint_f_index}")

    InpaintAny_dir = "./third_party/Inpaint-Anything"
    InpaintAny_py = os.environ.get("PYINPAINT")
    InpaintAny_script = os.path.join(InpaintAny_dir, "remove_anything.py")

    Cutie_dir = "./third_party/Cutie"
    Cutie_py = os.environ.get("PYCUTIE")
    Cutie_script = os.path.join(Cutie_dir, "interactive_demo.py")
    Cutie_workspace_dir = os.path.join(Cutie_dir, "workspace")
    # for i, inpaint_f in enumerate(inpaint_views):
    if 1:
        out_dir = config.out_dir
        # cameras_dir = os.path.join(os.path.dirname(inpaint_view), "../cameras")
        os.makedirs(out_dir, exist_ok=True)
        image_name = os.path.basename(inpaint_f).split(".")[0]

        inpaint_video_file = os.path.join(out_dir, image_name, "image", image_name + ".mp4")
        inpaint_mask_file = os.path.join(out_dir, image_name, "mask", f"{image_name}.png")
        inpaint_image_file = os.path.join(out_dir, image_name, "image", f"{image_name}.png")

        masked_ip_rgba_f = os.path.join(out_dir, f"{image_name}_rgba.png")       

        if do_inpaint:
            inpainting_command = [
                InpaintAny_py, InpaintAny_script,
                "--input_img", inpaint_f,
                "--coords_type", "click",
                "--point_coords", "200", "450",
                "--point_labels", "1",
                "--dilate_kernel_size", "15",
                "--output_dir", out_dir,
                "--sam_model_type", "vit_t",
                "--sam_ckpt", "./weights/mobile_sam.pt",
                "--lama_config", "./lama/configs/prediction/default.yaml",
                "--lama_ckpt", "./pretrained_models/big-lama"
            ]
            
            subprocess.run(inpainting_command, check=True, cwd=InpaintAny_dir)
            print(f"Finished inpainting {inpaint_f}")
            print(f"Output saved to {out_dir}/{image_name}")
            selected_number = input("Please enter inpaint selected number [0, 1 or 2]:")
            inpaint_file = os.path.join(out_dir, image_name, f"inpainted_with_mask_{selected_number}.png")
            # Create directories
            inpaint_image_dir = os.path.join(os.path.dirname(inpaint_file), "image")
            inpaint_mask_dir = os.path.join(os.path.dirname(inpaint_file), "mask")
            os.makedirs(inpaint_image_dir, exist_ok=True)
            os.makedirs(inpaint_mask_dir, exist_ok=True)
            subprocess.run(["cp", inpaint_file, inpaint_image_file], check=True)

            # Create video from images using ffmpeg
            command = [
                '/usr/bin/ffmpeg', '-framerate', '5', '-pattern_type', 'glob', '-i', '\"./*.png\"',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', '\"pad=ceil(iw/2)*2:ceil(ih/2)*2\"',
                f"{image_name}.mp4"
            ]
            shell_command = ' '.join(command)
            subprocess.run(shell_command, shell=True, check=True, cwd=inpaint_image_dir)

        if do_mask:
            # Run Cutie interactive demo
            command = [
                Cutie_py, 
                Cutie_script,
                "--video", inpaint_video_file, 
                "--num_objects", "1"
            ]
            subprocess.run(["rm", "-rf", os.path.join(Cutie_workspace_dir, f"{image_name}")], check=True)
            subprocess.run(command, check=True, cwd=Cutie_dir)     
            cutie_mask_file = os.path.join(Cutie_workspace_dir, f"{image_name}", "binary_masks", "0000000.png")
            # Copy the generated mask
            subprocess.run(["cp", "-rf", cutie_mask_file, inpaint_mask_file], check=True)


            inpaint_image = cv2.imread(inpaint_image_file)
            mask_image = cv2.imread(inpaint_mask_file, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask[:, :, np.newaxis]
            binary_mask = binary_mask[:inpaint_image.shape[0],...]
            binary_mask = binary_mask[:inpaint_image.shape[0],:inpaint_image.shape[1],:]
            masked_inpaint = inpaint_image * (binary_mask // 255) + 255 * (1 - binary_mask// 255)

            masked_ip_f = os.path.join(out_dir, f"{image_name}.png")
            cv2.imwrite(masked_ip_f, inpaint_image)

            alpha_channel = binary_mask
            rgba_image = cv2.merge((inpaint_image[:, :, 0], inpaint_image[:, :, 1], inpaint_image[:, :, 2], alpha_channel))
            cv2.imwrite(masked_ip_rgba_f, rgba_image)
       
        if do_center:
            inpaint_rgba = cv2.imread(masked_ip_rgba_f, cv2.IMREAD_UNCHANGED)
            desired_size = int(config['inpaint_size'] * (1 - config['border_ratio']))
            # Center the inpainted view
            mask = inpaint_rgba[:, :, 3]
            coords = np.nonzero(mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            h = y_max - y_min
            w = x_max - x_min
            scale = desired_size / max(h, w)
            h2 = int(h * scale)
            w2 = int(w * scale)
            y2_min = (config['inpaint_size'] - h2) // 2
            y2_max = y2_min + h2
            x2_min = (config['inpaint_size'] - w2) // 2
            x2_max = x2_min + w2
            # mask

            inpaint_rgba_center = np.zeros((config['inpaint_size'], config['inpaint_size'], 4), dtype=np.uint8)
            inpaint_rgba_center[y2_min:y2_max, x2_min:x2_max] = cv2.resize(inpaint_rgba[y_min:y_max, x_min:x_max], (w2, h2), interpolation=cv2.INTER_AREA)
            inpaint_rgba_center_f = os.path.join(out_dir, f"{image_name}_rgba_center.png")
            cv2.imwrite(inpaint_rgba_center_f, inpaint_rgba_center)

            # # algine optical center and scale
            # align_cx = int((x_min + x_max) / 2)
            # align_cy = int((y_min + y_max) / 2)            
            # all_cameras = sorted(glob(f"{cameras_dir}/*.json"))     
            # for camera_f in all_cameras:
            #     camera = json.load(open(camera_f, 'r'))
            #     camera['K_inpaint'] = copy.deepcopy(camera['K'])
            #     camera['K_inpaint'][0][2] = align_cx
            #     camera['K_inpaint'][1][2] = align_cy
            #     camera['K_manual'] = copy.deepcopy(camera['K'])
            #     camera['K_manual'][0][2] = config['manual_cx']
            #     camera['K_manual'][1][2] = config['manual_cy']
            #     camera['K_half_wh'] = copy.deepcopy(camera['K'])
            #     camera['K_half_wh'][0][2] = camera['width'] // 2
            #     camera['K_half_wh'][1][2] = camera['height'] // 2
            #     align_scale = scale * camera['height'] / config['inpaint_size']
            #     camera['scale_inpaint'] = align_scale
            #     save_json(camera_f, camera)

        if write_pose:
            data = {
                "elevation_deg": config['elevation_deg'],
                "azimuth_deg": config['azimuth_deg'],
                "fovy_deg": config['fovy_deg'],
                "distance": config['distance']
            }

            pose_f = os.path.join(out_dir, f"{image_name}.json")
            with open(pose_f, 'w') as f:
                json.dump(data, f, indent=4)



def RunInpaintInputViews(scene):
    scene_name = scene['name']
    inpaint_rgb = scene['inpaint_rgb']
    manual_cx_cy = scene['manual_cx_cy']

    config = {
        "inpaint_f": "",
        "inpaint_size": 256,
        "border_ratio": 0.2,
        "manual_cx": manual_cx_cy[0],
        "manual_cy": manual_cx_cy[1],
        "elevation_deg": 30,
        "azimuth_deg": -30,
        "fovy_deg": 41.15,
        "distance": 2
    }
    from attrdict import AttrDict
    config = AttrDict(config)
    inpaint_input_views(config, do_inpaint=True, do_mask=True, do_center=True, write_pose=True)



if __name__ == "__main__":
    scenes = [
                {"name": "AP10",     "type": "evaluation",        "inpaint_rgb": "0380.png", "manual_cx_cy": [310, 259]},
                {"name": "AP11",     "type": "evaluation",        "inpaint_rgb": "0083.png", "manual_cx_cy": [310, 259]},        
                {"name": "AP11",     "type": "evaluation",        "inpaint_rgb": "0418.png", "manual_cx_cy": [310, 259]},        
                {"name": "MPM10",     "type": "evaluation",        "inpaint_rgb": "0021.png", "manual_cx_cy": [310, 259]},
                {"name": "SB13",     "type": "evaluation",        "inpaint_rgb": "0148.png", "manual_cx_cy": [310, 259]},
                {"name": "SS1",     "type": "train",        "inpaint_rgb": "0260.png", "manual_cx_cy": [310, 259]},                
            ]   
    for scene in scenes:
        RunInpaintInputViews(scene)
 
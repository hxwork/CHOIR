import numpy as np
import math
import copy

def RT_opengl2opencv(RT):
     # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)

    return RT

def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)

    return RT_inv[:3, :]

def to_homo(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    
    return RT_h

def cartesian_to_spherical(xyz, unit="degree"):
    xy = xyz[:,0]**2 + xyz[:,1]**2
    distance = np.sqrt(xy + xyz[:,2]**2)
    #theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    elevation = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    if unit == "degree":
        return {"elevation": elevation.item() * 180 / math.pi,
                "azimuth":  azimuth.item() * 180 / math.pi,
                "distance": distance.item(),
                }
    else:
        return {"elevation": elevation.item(),
                "azimuth":  azimuth.item(),
                "distance": distance.item(),
                }
    
def get_relative_trans(c2w4x4, ref_c2w4x4):
    cN2c0 = np.linalg.inv(ref_c2w4x4) @ c2w4x4
    return cN2c0

def scale_t(cN2c14x4, scale):
    cN2c0_scale = copy.deepcopy(cN2c14x4)
    for cN2c0 in cN2c0_scale:
        cN2c0[:3, 3] *= scale
    return cN2c0_scale

def reposition_camera(cN2c14x4, c02w4x4):
    cN2w = c02w4x4 @ cN2c14x4
    return cN2w

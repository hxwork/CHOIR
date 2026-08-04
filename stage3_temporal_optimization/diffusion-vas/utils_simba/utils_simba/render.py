import torch
import numpy as np
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh
import cv2

glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)

def make_mesh_tensors(mesh, device='cuda', max_tex_size=None):
  mesh_tensors = {}
  if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
    img = np.array(mesh.visual.material.image.convert('RGB'))
    img = img[...,:3]
    if max_tex_size is not None:
      max_size = max(img.shape[0], img.shape[1])
      if max_size>max_tex_size:
        scale = 1/max_size * max_tex_size
        img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
    mesh_tensors['tex'] = torch.as_tensor(img, device=device, dtype=torch.float)[None]/255.0
    mesh_tensors['uv_idx']  = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
    uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
    uv[:,1] = 1 - uv[:,1]
    mesh_tensors['uv']  = uv
  else:
    if mesh.visual.vertex_colors is None:
      print(f"WARN: mesh doesn't have vertex_colors, assigning a pure color")
      mesh.visual.vertex_colors = np.tile(np.array([128,128,128]).reshape(1,3), (len(mesh.vertices), 1))
    mesh_tensors['vertex_color'] = torch.as_tensor(mesh.visual.vertex_colors[...,:3], device=device, dtype=torch.float)/255.0

  mesh_tensors.update({
    'pos': torch.tensor(mesh.vertices, device=device, dtype=torch.float),
    'faces': torch.tensor(mesh.faces, device=device, dtype=torch.int),
    'vnormals': torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
  })
  return mesh_tensors


def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords='y_down'):
  """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

  Ref:
  1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
  2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

  :param K: 3x3 ndarray with the intrinsic camera matrix.
  :param x0 The X coordinate of the camera image origin (typically 0).
  :param y0: The Y coordinate of the camera image origin (typically 0).
  :param w: Image width.
  :param h: Image height.
  :param nc: Near clipping plane.
  :param fc: Far clipping plane.
  :param window_coords: 'y_up' or 'y_down'.
  :return: 4x4 ndarray with the OpenGL projection matrix.
  """
  x0 = 0
  y0 = 0
  w = width
  h = height
  nc = znear
  fc = zfar

  depth = float(fc - nc)
  q = -(fc + nc) / depth
  qn = -2 * (fc * nc) / depth

  # Draw our images upside down, so that all the pixel-based coordinate
  # systems are the same.
  if window_coords == 'y_up':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])

  # Draw the images upright and modify the projection matrix so that OpenGL
  # will generate window coords that compensate for the flipped image coords.
  elif window_coords == 'y_down':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])
  else:
    raise NotImplementedError

  return proj

def projection_matrix_to_intrinsics(proj_mat, W, H):
  """Conversion of OpenGL proj. matrix to Hartley-Zisserman intrinsic matrix.

  :param proj_mat: 4x4 ndarray with the OpenGL projection matrix.
  :param W: Image width.
  :param H: Image height.
  :return: 3x3 ndarray with the intrinsic camera matrix.
  """
  # Extract the relevant components from the projection matrix
  K = np.zeros((3, 3), dtype=np.float32) 
  K[0, 0] = proj_mat[0, 0] * W / 2.0
  K[1, 1] = proj_mat[1, 1] * H / 2.0
  K[0, 2] = (1.0 - proj_mat[0, 2]) * W / 2.0
  K[1, 2] = (proj_mat[1, 2] + 1.0) * H / 2.0
  return K  


def transform_pts(pts,tf):
  """Transform 2d or 3d points
  @pts: (...,N_pts,3)
  @tf: (...,4,4)
  """
  if len(tf.shape)>=3 and tf.shape[-3]!=pts.shape[-2]:
    tf = tf[...,None,:,:]
  return (tf[...,:-1,:-1]@pts[...,None] + tf[...,:-1,-1:])[...,0]

def to_homo_torch(pts):
  '''
  @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
  '''
  ones = torch.ones((*pts.shape[:-1],1), dtype=torch.float, device=pts.device)
  homo = torch.cat((pts, ones),dim=-1)
  return homo

def transform_dirs(dirs,tf):
  """
  @dirs: (...,3)
  @tf: (...,4,4)
  """
  if len(tf.shape)>=3 and tf.shape[-3]!=dirs.shape[-2]:
    tf = tf[...,None,:,:]
  return (tf[...,:3,:3]@dirs[...,None])[...,0]

def nvdiffrast_render(K=None, H=None, W=None, ob_in_cvcams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None, output_size=None, use_light=False, light_color=None, light_dir=np.array([0,0,1]), light_pos=np.array([0,0,0]), w_ambient=0.8, w_diffuse=0.5, extra={}):
  '''Just plain rendering, not support any gradient
  @K: (3,3) np array
  @ob_in_cams: (N,4,4) torch tensor, openCV camera
  @projection_mat: np array (4,4)
  @output_size: (height, width)
  @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
  @light_dir: in cam space
  @light_pos: in cam space
  '''
  if glctx is None:
    if context == 'gl':
      glctx = dr.RasterizeGLContext()
    elif context=='cuda':
      glctx = dr.RasterizeCudaContext()
    else:
      raise NotImplementedError
    print("created context")

  if mesh_tensors is None:
    mesh_tensors = make_mesh_tensors(mesh)
  pos = mesh_tensors['pos']
  vnormals = mesh_tensors['vnormals']
  tri = mesh_tensors['faces']
  has_tex = 'tex' in mesh_tensors

  ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float)[None]@ob_in_cvcams
  if projection_mat is None:
    projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.1, zfar=100)
  projection_mat = torch.as_tensor(projection_mat.reshape(-1,4,4), device='cuda', dtype=torch.float)
  mtx = projection_mat@ob_in_glcams

  if output_size is None:
    output_size = np.asarray([H,W])

  pts_cam = transform_pts(pos, ob_in_cvcams)
  pos_homo = to_homo_torch(pos)
  pos_clip = (mtx[:,None]@pos_homo[None,...,None])[...,0]
  if bbox2d is not None:
    l = bbox2d[:,0]
    t = H-bbox2d[:,1]
    r = bbox2d[:,2]
    b = H-bbox2d[:,3]
    tf = torch.eye(4, dtype=torch.float, device='cuda').reshape(1,4,4).expand(len(ob_in_cvcams),4,4).contiguous()
    tf[:,0,0] = W/(r-l)
    tf[:,1,1] = H/(t-b)
    tf[:,3,0] = (W-r-l)/(r-l)
    tf[:,3,1] = (H-t-b)/(t-b)
    pos_clip = pos_clip@tf
  rast_out, _ = dr.rasterize(glctx, pos_clip, tri, resolution=np.asarray(output_size))
  xyz_map, _ = dr.interpolate(pts_cam, rast_out, tri)
  depth = xyz_map[...,2]
  if has_tex:
    texc, _ = dr.interpolate(mesh_tensors['uv'], rast_out, mesh_tensors['uv_idx'])
    color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
  else:
    color, _ = dr.interpolate(mesh_tensors['vertex_color'], rast_out, tri)

  if use_light:
    get_normal = True
  if get_normal:
    vnormals_cam = transform_dirs(vnormals, ob_in_cvcams)
    normal_map, _ = dr.interpolate(vnormals_cam, rast_out, tri)
    normal_map = F.normalize(normal_map, dim=-1)
    normal_map = torch.flip(normal_map, dims=[1])
  else:
    normal_map = None

  if use_light:
    if light_dir is not None:
      light_dir_neg = -torch.as_tensor(light_dir, dtype=torch.float, device='cuda')
    else:
      light_dir_neg = torch.as_tensor(light_pos, dtype=torch.float, device='cuda').reshape(1,1,3) - pts_cam
    diffuse_intensity = (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1)).sum(dim=-1).clip(0, 1)[...,None]
    diffuse_intensity_map, _ = dr.interpolate(diffuse_intensity, rast_out, tri)  # (N_pose, H, W, 1)
    if light_color is None:
      light_color = color
    else:
      light_color = torch.as_tensor(light_color, device='cuda', dtype=torch.float)
    color = color*w_ambient + diffuse_intensity_map*light_color*w_diffuse

  color = color.clip(0,1)
  color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background using alpha
  color = torch.flip(color, dims=[1])   # Flip Y coordinates
  depth = torch.flip(depth, dims=[1])
  extra['xyz_map'] = torch.flip(xyz_map, dims=[1])
  return color, depth, normal_map

def diff_renderer(verts, tri, color, projection, ob_in_cvcams, resolution, glctx):
  '''
  Render the 3D mesh using the given parameters.
  Args:
      verts: (1, N, 3) torch tensor - Vertex positions.
      tri: (M, 3) torch tensor  - Triangle indices.
      color: (1, N, 3) torch tensor - Vertex colors.
      projection: (4, 4) torch tensor - Projection matrix.
      ob_in_cvcams: (4, 4) torch tensor, openCV camera- Camera extrinsics.
      resolution: (H, W) - Output image resolution.
  Returns:
      img: (H, W, 3) - Rendered image.
  '''
  device = projection.device

  ones = torch.ones(1, verts.shape[1], 1).to(device)
  pos = torch.cat((verts, ones), dim=2).float() # augumented pos

  # try:
  #     view_matrix = torch.inverse(c2ws)
  # except:
  #     view_matrix = torch.linalg.pinv(c2ws)
  pts_cam = transform_pts(pos[..., :3], ob_in_cvcams[None])
  ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float) @ ob_in_cvcams
  mtx = (projection @ ob_in_glcams).unsqueeze(0)
  pos_clip = pos @ mtx.mT
  

  rast_out, _ = dr.rasterize(glctx, pos_clip, tri, resolution)

  xyz_map, _ = dr.interpolate(pts_cam, rast_out, tri)
  depth = xyz_map[...,2]  

  out, _ = dr.interpolate(color, rast_out, tri)
  out = dr.antialias(out, rast_out, pos_clip, tri)
  img = torch.flip(out[0], dims=[0]) # Flip vertically.

  return img, depth

def get_ray_origin_direction(c2w: torch.Tensor, K: torch.Tensor, pixel_coords: torch.Tensor):
    """
    Compute ray origins and directions in world coordinates for given pixel coordinates.

    Args:
        c2w (torch.Tensor): Camera-to-World transformation matrix of shape [4, 4].
        K (torch.Tensor): Camera intrinsic matrix of shape [3, 3].
        pixel_coords (torch.Tensor): Pixel coordinates of shape [N, 2], where each row is (u, v).

    Returns:
        ray_origins (torch.Tensor): Ray origins in world coordinates of shape [N, 3].
        ray_directions (torch.Tensor): Normalized ray directions in world coordinates of shape [N, 3].
    """
    # Ensure inputs are of correct shapes
    assert c2w.shape == (4, 4), "c2w must be a [4, 4] matrix."
    assert K.shape == (3, 3), "K must be a [3, 3] matrix."
    assert pixel_coords.shape[1] == 2, "pixel_coords must have shape [N, 2]."

    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Convert pixel coordinates to camera space
    u = pixel_coords[:, 0]
    v = pixel_coords[:, 1]

    # Compute directions in camera space
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = torch.ones_like(x_cam)

    directions_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)  # Shape: [N, 3]

    # Normalize directions in camera space
    directions_cam_norm = torch.nn.functional.normalize(directions_cam, dim=1)  # Shape: [N, 3]

    # Extract rotation (R) and translation (t) from c2w
    R = c2w[:3, :3]  # Shape: [3, 3]
    t = c2w[:3, 3]   # Shape: [3]

    # Rotate directions to world space
    directions_world = torch.matmul(directions_cam_norm, R.T)  # Shape: [N, 3]

    # Normalize directions in world space
    directions_world_norm = torch.nn.functional.normalize(directions_world, dim=1)  # Shape: [N, 3]

    # Ray origin is the camera position in world space, replicated for each ray
    ray_origins = t.unsqueeze(0).expand_as(directions_world_norm)  # Shape: [N, 3]

    return ray_origins, directions_world_norm
import trimesh
import numpy as np
import open3d as o3d
from tqdm import tqdm

def mesh_to_voxel_grid(mesh, voxel_size):
    """
    Convert a mesh to a voxel occupancy grid.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        voxel_size (float): The size of each voxel.
        
    Returns:
        voxels (numpy.ndarray): 3D boolean array indicating occupied voxels.
        voxel_origin (numpy.ndarray): Origin of the voxel grid.
        grid_size (tuple): Number of voxels along each axis.
    """
    # Voxelization using trimesh
    voxel_grid = mesh.voxelized(voxel_size)
    voxels = voxel_grid.matrix
    voxel_origin = voxel_grid.origin
    grid_size = voxels.shape
    return voxels, voxel_origin, grid_size

def get_camera_pose(eye, target, up):
    """
    Compute the camera pose matrix (camera-to-world transformation) given eye (camera position), target, and up vector.
    
    Returns a 4x4 numpy array.
    """
    F = target - eye
    f = F / np.linalg.norm(F)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    
    # Camera-to-world rotation matrix
    R = np.vstack([s, u, -f]).T
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = eye
    return M

def generate_camera_views(mesh, num_views):
    """
    Generate camera poses uniformly distributed around the mesh.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        num_views (int): Number of camera views to generate.
        
    Returns:
        camera_poses (list): List of 4x4 camera pose matrices.
    """
    # Use spherical coordinates to distribute cameras
    camera_poses = []
    
    radius = mesh.bounding_sphere.primitive.radius * 2  # Position cameras at twice the bounding sphere radius
    center = mesh.bounding_sphere.primitive.center

    for i in range(num_views):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        eye = center + np.array([x, y, z])
        target = center
        up = np.array([0, 0, 1])  # Assuming Z-up coordinate system
        # Adjust up vector if necessary
        if np.allclose(np.cross(target - eye, up), 0):
            up = np.array([0, 1, 0])  # Use Y-up if camera is along Z-axis
        # Compute camera pose matrix
        camera_pose = get_camera_pose(eye, target, up)
        camera_poses.append(camera_pose)
    
    return camera_poses

def voxel_centers(voxels, origin, voxel_size):
    """
    Compute the centers of all occupied voxels.
    
    Parameters:
        voxels (numpy.ndarray): 3D boolean array indicating occupied voxels.
        origin (numpy.ndarray): Origin of the voxel grid.
        voxel_size (float): Size of each voxel.
        
    Returns:
        centers (numpy.ndarray): Nx3 array of voxel center coordinates.
    """
    occupied = np.argwhere(voxels)
    centers = origin + (occupied + 0.5) * voxel_size
    return centers, occupied

def determine_visibility(mesh, voxel_centers, camera_poses, max_distance):
    """
    Determine visibility of each voxel from each camera view.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        voxel_centers (numpy.ndarray): Nx3 array of voxel center coordinates.
        camera_poses (list): List of camera pose matrices.
        max_distance (float): Maximum distance for visibility checks.
        
    Returns:
        visibility (numpy.ndarray): NxM boolean array where N is number of voxels and M is number of views.
    """
    num_voxels = voxel_centers.shape[0]
    num_views = len(camera_poses)
    visibility = np.zeros((num_voxels, num_views), dtype=bool)
    
    # Create a ray-mesh query object
    scene = trimesh.Scene(mesh)
    mesh_trimesh = scene.dump().sum()
    rmi = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_trimesh)
    
    for view_idx in tqdm(range(num_views), desc="Processing views"):
        camera_pose = camera_poses[view_idx]
        cam_pos = camera_pose[:3, 3]
        # Directions from camera to voxel centers
        directions = voxel_centers - cam_pos
        distances = np.linalg.norm(directions, axis=1)
        # Normalize directions
        directions_normalized = directions / distances[:, np.newaxis]
        
        # Limit rays to max_distance
        valid = distances < max_distance
        origins = np.tile(cam_pos, (num_voxels, 1))
        rays = directions_normalized
        # Perform ray intersection
        locations, index_ray, index_tri = rmi.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False  # Correct parameter name
        )
        
        # Initialize all voxels as visible
        visibility[:, view_idx] = True
        # For each ray that intersects the mesh before reaching the voxel
        for i in range(len(index_ray)):
            ray_idx = index_ray[i]
            hit_distance = np.linalg.norm(locations[i] - cam_pos)
            if hit_distance < distances[ray_idx]:
                visibility[ray_idx, view_idx] = False  # Occluded
        
    return visibility

def visualize_voxel_visibility(voxels, voxel_origin, voxel_size, visibility, num_views_to_display=3):
    """
    Visualize voxel occupancy and visibility from selected views.
    
    Parameters:
        voxels (numpy.ndarray): 3D boolean array indicating occupied voxels.
        voxel_origin (numpy.ndarray): Origin of the voxel grid.
        voxel_size (float): Size of each voxel.
        visibility (numpy.ndarray): NxM boolean array indicating visibility.
        num_views_to_display (int): Number of views to visualize.
    """
    # Create Open3D voxel grid
    occupied_indices = np.argwhere(voxels)
    voxel_centers_np = voxel_origin + (occupied_indices + 0.5) * voxel_size
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(voxel_centers_np)
    
    # Color voxels based on visibility from first few views
    colors = np.zeros((voxel_centers_np.shape[0], 3))
    # init the colors to light bue
    # colors = np.ones((voxel_centers_np.shape[0], 3)) * 0.7
    for v in range(min(num_views_to_display, visibility.shape[1])):
        visible = visibility[:, v]
        colors[visible] += np.array([1, 0, 0])  # Red for visibility from view v
    # Normalize colors
    colors = np.clip(colors, 0, 1)

    import rerun as rr
    rr.log(
        f"world/voxel_visibility",
        rr.Points3D(voxel_centers_np, colors=colors),
        timeless=True
    )
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    # o3d.visualization.draw_geometries([pcd])

def create_colored_mesh_from_voxels(voxels, voxel_origin, voxel_size, visibility):
    import open3d as o3d
    
    # Get indices of occupied voxels
    occupied_indices = np.argwhere(voxels)
    voxel_centers_np = voxel_origin + (occupied_indices + 0.5) * voxel_size
    
    # Create a list of colored cubes
    mesh = o3d.geometry.TriangleMesh()
    for idx, center in enumerate(voxel_centers_np):
        # Create a cube for each voxel
        cube = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
        cube.translate(center - np.array([voxel_size / 2] * 3))
        
        # Assign color based on visibility
        visible = np.any(visibility[idx])
        if visible:
            cube.paint_uniform_color([1, 0, 0])  # Red for visible voxels
        else:
            cube.paint_uniform_color([0, 0, 0])  # Black for non-visible voxels
        
        mesh += cube
    return mesh    

def get_camera_intrinsics(image_width, image_height, fx, fy, cx, cy):
    """
    Create an Open3D camera intrinsic object.
    """
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(
        width=image_width,
        height=image_height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    return intrinsics

def render_mesh_and_get_visibility(mesh, camera_intrinsics, c2w):
    """
    Render the mesh from the given camera pose and obtain per-pixel visibility.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The mesh to render.
        camera_intrinsics (o3d.camera.PinholeCameraIntrinsic): Camera intrinsic parameters.
        camera_pose (numpy.ndarray): 4x4 camera pose matrix.

    Returns:
        visibility_image (numpy.ndarray): A 2D boolean array indicating pixel visibility.
        color_image (o3d.geometry.Image): The rendered color image.
    """
    # Set up the renderer
    width = camera_intrinsics.width
    height = camera_intrinsics.height

    # Create an offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Create a default material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"  # Or "defaultUnlit" for unlit rendering

    # Add the mesh to the scene with the material
    renderer.scene.add_geometry("mesh", mesh, material)

    # Set camera parameters
    renderer.setup_camera(camera_intrinsics, np.linalg.inv(c2w))

    # Render the scene
    color_image = renderer.render_to_image()
    depth_image = renderer.render_to_depth_image()

    # Convert depth image to numpy array
    depth = np.asarray(depth_image)

    # Pixels with valid depth are visible
    visibility_image = depth > 0
    return visibility_image, color_image


def main():
    # Parameters
    mesh_file = 'outputs/hold_SM2_ho3d.0/3d_ref/save/it1000-export/model.obj'  # Replace with your mesh file path
    voxel_size = 0.01  # Adjust voxel size as needed
    num_views = 5  # Number of camera views
    max_distance = 1000  # Maximum distance for visibility checks
    
    # Load mesh
    mesh = trimesh.load(mesh_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()  # In case the mesh is a scene with multiple meshes
    
    # Voxelization
    voxels, voxel_origin, grid_size = mesh_to_voxel_grid(mesh, voxel_size)
    print(f"Voxel grid size: {grid_size}")
    
    # Generate camera views
    camera_poses = generate_camera_views(mesh, num_views)
    
    # Compute voxel centers
    voxel_centers_np, occupied_indices = voxel_centers(voxels, voxel_origin, voxel_size)
    print(f"Number of occupied voxels: {voxel_centers_np.shape[0]}")
    
    # Determine visibility
    try:
        # Try to use PyEmbree for faster ray tracing
        visibility = determine_visibility(mesh, voxel_centers_np, camera_poses, max_distance)
    except ImportError:
        print("PyEmbree not installed, falling back to slower ray tracing.")
        trimesh.ray.ray_pyembree.RayMeshIntersector = None
        visibility = determine_visibility(mesh, voxel_centers_np, camera_poses, max_distance)
    
    # Example: Print visibility of first 10 voxels
    for i in range(min(10, visibility.shape[0])):
        print(f"Voxel {i}: Visible from views {np.where(visibility[i])[0]}")
    
    # Visualization (optional)
    visualize_voxel_visibility(voxels, voxel_origin, voxel_size, visibility)

        # Create colored mesh from voxels
    mesh_with_visibility = create_colored_mesh_from_voxels(
        voxels, voxel_origin, voxel_size, visibility
    )
    # Optional: Save the mesh to a file
    o3d.io.write_triangle_mesh("mesh_with_visibility.ply", mesh_with_visibility)

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh_with_visibility])

    # Define camera parameters
    image_width = 640  # Example image width
    image_height = 480  # Example image height
    fx = fy = 525.0  # Example focal length in pixels
    cx = image_width / 2.0  # Principal point x-coordinate
    cy = image_height / 2.0  # Principal point y-coordinate

    camera_intrinsics = get_camera_intrinsics(image_width, image_height, fx, fy, cx, cy)
    breakpoint()

    # Choose a camera pose (e.g., the first one)
    camera_pose = camera_poses[0]

    # Render the mesh and get per-pixel visibility
    breakpoint()
    visibility_image, color_image = render_mesh_and_get_visibility(
        mesh_with_visibility, camera_intrinsics, camera_pose
    )

    # Now, visibility_image is a 2D boolean array indicating visibility per pixel
    # You can save or process this as needed
    import matplotlib.pyplot as plt

    plt.imshow(visibility_image, cmap='gray')
    plt.title('Per-Pixel Visibility')
    plt.show()


if __name__ == "__main__":
    main()

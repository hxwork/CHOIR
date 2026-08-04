import trimesh
import numpy as np
from tqdm import tqdm
import os

class Visibility:
    def __init__(self, voxel_size = 0.01, max_distance = 100.0, out_dir = None):
        self.voxel_size = voxel_size
        self.max_distance = max_distance
        print(f"Voxel size: {self.voxel_size}")
        self.out_dir = out_dir
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

    def mesh_to_voxel_grid(self, mesh):
        """
        Convert a mesh to a voxel occupancy grid.
        
        Parameters:
            mesh (trimesh.Trimesh): The input mesh.
            voxel_size (float): The size of each voxel.
            
        Returns:
            voxel_centers (numpy.ndarray): Nx3 array of voxel center coordinates.
            voxel_origin (numpy.ndarray): Origin of the voxel grid.
        """
        # Voxelization using trimesh
        voxel_grid = mesh.voxelized(self.voxel_size)
        voxels = voxel_grid.matrix
        voxel_origin = np.array(voxel_grid.origin)
        occupied = np.argwhere(voxels)
        voxel_centers = voxel_origin + (occupied + 0.5) * self.voxel_size
        
        return voxel_centers

    def determine_visibility(self, mesh, voxel_centers, camera_poses):
        """
        Determine visibility of each voxel from each camera view.
        
        Parameters:
            mesh (trimesh.Trimesh): The input mesh.
            voxel_centers (numpy.ndarray): Nx3 array of voxel center coordinates.
            camera_poses (list): List of camera pose matrices.  
        Returns:
            visibility (numpy.ndarray): NxM boolean array where N is number of voxels and M is number of views.
        """        
        num_voxels = voxel_centers.shape[0]
        num_views = len(camera_poses)
        visibility = np.zeros((num_views, num_voxels), dtype=bool)
        
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
            valid = distances < self.max_distance
            origins = np.tile(cam_pos, (num_voxels, 1))
            rays = directions_normalized
            # Perform ray intersection
            locations, index_ray, index_tri = rmi.intersects_location(
                ray_origins=origins,
                ray_directions=directions,
                multiple_hits=False  # Correct parameter name
            )
            
            # Initialize all voxels as visible
            visibility[view_idx, :] = True
            # For each ray that intersects the mesh before reaching the voxel
            for i in range(len(index_ray)):
                ray_idx = index_ray[i]
                hit_distance = np.linalg.norm(locations[i] - cam_pos)
                if hit_distance < distances[ray_idx]:
                    visibility[view_idx, ray_idx] = False  # Occluded
            
        return visibility
        
    def save_visibility(self, voxel_centers, visibility, view_idx):

        import open3d as o3d
        
        # Get indices of occupied voxels

        
        # Create a list of colored cubes
        mesh = o3d.geometry.TriangleMesh()
        for idx, center in enumerate(voxel_centers):
            # Create a cube for each voxel
            cube = o3d.geometry.TriangleMesh.create_box(self.voxel_size, self.voxel_size, self.voxel_size)
            cube.translate(center - np.array([self.voxel_size / 2] * 3))
            
            # Assign color based on visibility
            visible = np.any(visibility[idx])
            if visible:
                cube.paint_uniform_color([1, 0, 0])  # Red for visible voxels
            else:
                cube.paint_uniform_color([0, 0, 0])  # Black for non-visible voxels
            
            mesh += cube
        # Optional: Save the mesh to a file
        o3d.io.write_triangle_mesh(f"{self.out_dir}/visibility_{view_idx:03d}.ply", mesh)

    def run(self, mesh, camera_poses):
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
        # mesh = trimesh.load(mesh_file)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()  # In case the mesh is a scene with multiple meshes
        # voxelize the mesh
        voxel_centers = self.mesh_to_voxel_grid(mesh)
        visibility = self.determine_visibility(mesh, voxel_centers, camera_poses)
        if self.out_dir is not None:
            for view_idx in range(visibility.shape[0]):
                self.save_visibility(voxel_centers, visibility[view_idx], view_idx)
        return voxel_centers, visibility


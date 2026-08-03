"""
Last modified date: 2023.02.23
Author: Ruicheng Wang, Jialiang Zhang
Description: Class ObjectModel
"""

import json
import os

import numpy as np
import plotly.graph_objects as go
import pytorch3d.ops
import pytorch3d.structures
import torch
import trimesh as tm
from torchsdf import compute_sdf, index_vertices_by_faces


class ObjectModel:

    def __init__(self, data_root_path, batch_size_each, num_samples=2000, device="cuda"):
        """
        Create a Object Model
        
        Parameters
        ----------
        data_root_path: str
            directory to object meshes
        batch_size_each: int
            batch size for each objects
        num_samples: int
            numbers of object surface points, sampled with fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_code_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None
        self.scale_choice = torch.tensor([0.06, 0.08, 0.1], dtype=torch.float, device=self.device)

    def initialize(self, object_code_list):
        """
        Initialize Object Model with list of objects
        
        Choose scales, load meshes, sample surface points
        
        Parameters
        ----------
        object_code_list: list | str
            list of object codes
        """
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        for object_code in object_code_list:
            self.object_scale_tensor.append(self.scale_choice[torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each,), device=self.device)])
            self.object_mesh_list.append(tm.load(os.path.join(self.data_root_path, object_code, "coacd", "decomposed.obj"), force="mesh", process=False))
            object_verts = torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device)
            object_faces = torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device)
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
            if self.num_samples != 0:
                vertices = torch.tensor(self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(self.object_mesh_list[-1].faces, dtype=torch.float, device=self.device)
                mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)
                surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
                surface_points.to(dtype=float, device=self.device)
                self.surface_points_tensor.append(surface_points)
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)
        if self.num_samples != 0:
            self.surface_points_tensor = torch.stack(self.surface_points_tensor,
                                                     dim=0).repeat_interleave(self.batch_size_each, dim=0)  # (n_objects * batch_size_each, num_samples, 3)

    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from hand contact points to object meshes and return contact normals
        
        Interiors are positive, exteriors are negative
        
        Use our modified Kaolin package
        
        Parameters
        ----------
        x: (B, `n_contact`, 3) torch.Tensor
            hand contact points
        with_closest_points: bool
            whether to return closest points on object meshes
        
        Returns
        -------
        distance: (B, `n_contact`) torch.Tensor
            signed distances from hand contact points to object meshes, inside is positive
        normals: (B, `n_contact`, 3) torch.Tensor
            contact normal vectors defined by gradient
        closest_points: (B, `n_contact`, 3) torch.Tensor
            contact points on object meshes, returned only when `with_closest_points is True`
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)
        x = x / scale.unsqueeze(2)
        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            dis, dis_signs, normal, _ = compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance * scale
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = (torch.stack(closest_points) * scale.unsqueeze(2)).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals

    def get_plotly_data(self, i, color='lightgreen', opacity=0.5, pose=None):
        """
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        color: str
            color of mesh
        opacity: float
            opacity
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        model_index = i // self.batch_size_each
        model_scale = self.object_scale_tensor[model_index, i % self.batch_size_each].detach().cpu().numpy()
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data = go.Mesh3d(x=vertices[:, 0],
                         y=vertices[:, 1],
                         z=vertices[:, 2],
                         i=mesh.faces[:, 0],
                         j=mesh.faces[:, 1],
                         k=mesh.faces[:, 2],
                         color=color,
                         opacity=opacity)
        return [data]


class ObjectMineModel:

    def __init__(self, data_root_path, batch_size_each, num_samples=2000, device="cuda"):
        """
        Create a Object Model
        
        Parameters
        ----------
        data_root_path: str
            directory to object meshes
        batch_size_each: int
            batch size for each objects
        num_samples: int
            numbers of object surface points, sampled with fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None

        self.scale_choice = torch.tensor([0.06, 0.08, 0.1], dtype=torch.float, device=self.device)

    def initialize(self, object_code_list):
        """
        Initialize Object Model with list of object codes
        
        Choose scales, load meshes, sample surface points
        
        Parameters
        ----------
        object_code_list: list | str
            list of object codes
        """
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        for obj_idx, object_dir in enumerate(object_code_list):
            scale_path = os.path.join(object_dir, "scale.json")
            if os.path.exists(scale_path):
                with open(scale_path, 'r') as f:
                    scale_data = json.load(f)
                scale = torch.tensor(scale_data['scale'], dtype=torch.float, device=self.device)
                if scale.ndim == 0:  # It's a scalar
                    scale = scale.repeat(3)
                # After this, scale should be a 1D tensor of size 3
                scale = scale.unsqueeze(0).repeat(self.batch_size_each, 1)  # (batch_size_each, 3)
            else:
                # random scale
                scale = self.scale_choice[torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each,), device=self.device)]
                scale = scale.unsqueeze(-1).repeat(1, 3)  # (batch_size_each, 3)
            self.object_scale_tensor.append(scale)
            obj_path = os.path.join(object_dir, "decomposed.obj")
            obj_mesh = tm.load(obj_path, force="mesh", process=False)
            self.object_mesh_list.append(obj_mesh)
            object_verts = torch.Tensor(obj_mesh.vertices).to(self.device)
            object_faces = torch.Tensor(obj_mesh.faces).long().to(self.device)
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
            if self.num_samples != 0:
                points_path = os.path.join(object_dir, f"obj_points_{self.num_samples}.ply")
                if os.path.exists(points_path):
                    pc = tm.load(points_path)
                    surface_points = torch.tensor(pc.vertices, dtype=torch.float, device=self.device)
                else:
                    print(f"Generating FPS points for {object_dir}...")
                    vertices = torch.tensor(obj_mesh.vertices, dtype=torch.float, device=self.device)
                    faces = torch.tensor(obj_mesh.faces, dtype=torch.long, device=self.device)
                    mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
                    dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)
                    surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]

                    pc_to_save = tm.PointCloud(surface_points.cpu().numpy())
                    pc_to_save.export(points_path)
                    print(f"Saved FPS points to {points_path}")

                surface_points = surface_points.to(dtype=torch.float, device=self.device)
                self.surface_points_tensor.append(surface_points)
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)  # (n_objects, batch_size_each, 3)
        if self.num_samples != 0:
            # (n_objects * batch_size_each, num_samples, 3)
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(self.batch_size_each, dim=0)
            # self.surface_points_tensor = einops.repeat(torch.stack(self.surface_points_tensor,
            #                                          dim=0), 'n s 3 -> (n b) s 3', b=self.batch_size_each)

    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from hand contact points to object meshes and return contact normals
        
        Interiors are positive, exteriors are negative
        
        Use our modified Kaolin package
        
        Parameters
        ----------
        x: (B, `n_contact`, 3) torch.Tensor
            hand contact points
        with_closest_points: bool
            whether to return closest points on object meshes
        
        Returns
        -------
        distance: (B, `n_contact`) torch.Tensor
            signed distances from hand contact points to object meshes, inside is positive
        normals: (B, `n_contact`, 3) torch.Tensor
            contact normal vectors defined by gradient
        closest_points: (B, `n_contact`, 3) torch.Tensor
            contact points on object meshes, returned only when `with_closest_points is True`
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)  # (n_objects * batch_size_each, n_points, 3)
        x = x / scale
        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            dis, dis_signs, normal, _ = compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance * scale.mean(dim=2)
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = (torch.stack(closest_points) * scale.mean(dim=2, keepdim=True)).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals

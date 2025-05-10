import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def prepare_data_from_ply(file_names, num_points=5000, device='cuda', normalize=True):
    """
    Loads and preprocesses point cloud data from PLY files.
    Optionally normalizes each point cloud to [0, 1] and returns a DataLoader.

    Args:
        file_names (list of str): List of PLY file paths.
        num_points (int): Number of points to sample from each file.
        device (str): 'cuda' or 'cpu'.
        normalize (bool): Whether to normalize each point cloud to [0, 1].

    Returns:
        DataLoader: DataLoader containing point cloud tensors.
    """
    input_data_list = []

    for file_name in file_names:
        pcd = o3d.io.read_point_cloud(file_name)
        verts_array = np.asarray(pcd.points)  # (N_points, 3)

        if verts_array.shape[0] < num_points:
            raise ValueError(f"{file_name}: not enough vertices ({verts_array.shape[0]}) for sampling {num_points}")

        sampled_idx = np.random.choice(verts_array.shape[0], num_points, replace=False)
        pointcloud = verts_array[sampled_idx]

        if normalize:
            denom = (pointcloud.max(axis=0) - pointcloud.min(axis=0)) + 1e-6  # prevent divide-by-zero
            pointcloud = (pointcloud - pointcloud.min(axis=0)) / denom

        input_data_list.append(pointcloud)

    input_data = np.stack(input_data_list)  # (N_samples, N_points, 3)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    return DataLoader(TensorDataset(input_tensor), batch_size=1, shuffle=False)

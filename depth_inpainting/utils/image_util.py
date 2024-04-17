import matplotlib
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F



def resize_max_res(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
    Returns:
        `Image.Image`: Resized image.
    """
    
    original_width, original_height = img.size
    
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = img.resize((new_width, new_height))
    return resized_img


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def create_point_cloud(depth_map, camera_matrix, extrinsic_matrix):

    """Create point cloud from depth map and camera parameters."""
    
    # Get shape of depth map
    height, width = depth_map.shape

    # Create meshgrid for pixel coordinates
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    # Normalize pixel coordinates
    normalized_x = (x - camera_matrix[0, 2]) / camera_matrix[0, 0]
    normalized_y = (y - camera_matrix[1, 2]) / camera_matrix[1, 1]
    normalized_z = np.ones_like(x)

    # Homogeneous coordinates in camera frame
    depth_map_reshaped = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
    homogeneous_camera_coords = depth_map_reshaped * np.dstack((normalized_x, 
                                                                normalized_y, 
                                                                normalized_z))

    # Add ones to the last dimension
    ones = np.ones((height, width, 1))
    homogeneous_camera_coords = np.dstack((homogeneous_camera_coords, ones))

    # Transform points to world coordinates
    homogeneous_world_coords = np.dot(homogeneous_camera_coords, 
                                      extrinsic_matrix.T)

    # Divide by the fourth coordinate (homogeneous normalization)
    point_cloud = (homogeneous_world_coords[:, :, :3] / 
                                            homogeneous_world_coords[:, :, 3:])

    point_cloud = point_cloud.reshape(-1, 3)

    return point_cloud
def write_ply_mask(points,colors,path_ply,mask=None):
    if mask is not None:
        num = np.sum(mask)
    else:
        num = points.shape[0]
    ply_header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''.format(num)
        # points.shape[0]
    # import ipdb;ipdb.set_trace()
    # if mask is not None:
    with open(path_ply, 'w') as f:
        f.write(ply_header)
        for i in range(points.shape[0]):
            if mask.reshape(-1)[i]:
                f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2],
                                                                int(colors[i, 2]*255), int(colors[i, 1]*255), int(colors[i, 0]*255)))
def write_ply(points,colors,path_ply,mask=None):
    if mask is not None:
        num = np.sum(mask)
    else:
        num = points.shape[0]
    ply_header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''.format(num)

    with open(path_ply, 'w') as f:
        f.write(ply_header)
        for i in range(points.shape[0]):
            f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2],
                                                                int(colors[i, 2]*255), int(colors[i, 1]*255), int(colors[i, 0]*255)))         



def Disparity_Normalization(disparity):
    min_value = torch.min(disparity)
    max_value = torch.max(disparity)
    normalized_disparity = ((disparity -min_value)/(max_value-min_value+1e-5) - 0.5) * 2    
    return normalized_disparity
def Disparity_Normalization_mask(disparity,min_value,max_value):
    normalized_disparity = ((disparity -min_value)/(max_value-min_value+1e-5) - 0.5) * 2    
    return normalized_disparity
def resize_max_res_tensor(input_tensor,is_disp=False,recom_resolution=768):

    original_H, original_W = input_tensor.shape[2:]
    
    downscale_factor = min(recom_resolution/original_H,
                           recom_resolution/original_W)
    
    resized_input_tensor = F.interpolate(input_tensor,
                                         scale_factor=downscale_factor,mode='bilinear',
                                         align_corners=False)
    if is_disp:
        return resized_input_tensor * downscale_factor,downscale_factor
    else:
        return resized_input_tensor
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
import torch
import os
from os import makedirs, path
from errno import EEXIST
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from simple_knn._C import distCUDA2
from torch import nn
import argparse
C0 = 0.28209479177387814
max_sh_degree=0
def RGB2SH(rgb):
    return (rgb - 0.5) / C0
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def construct_list_of_attributes(features_dc,features_rest,scaling,rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
def create_from_pcd(pcd,path):
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (0 + 1) ** 2)).float().cuda()
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = inverse_sigmoid(0.999 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    scaling = nn.Parameter(scales.requires_grad_(True))
    rotation = nn.Parameter(rots.requires_grad_(True))
    opacity = nn.Parameter(opacities.requires_grad_(True))
    max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")

    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scaling, rotation)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
def load_ply(path):
    plydata = PlyData.read(path)
    
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # xyz features_dc features_extra opacities scales rots
    return xyz, features_dc, features_extra, opacities, scales, rots
def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise
def convert_np_tensor(xyz, features_dc, features_rest, opacity, scaling, rotation):
    xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
    features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    features_rest = torch.tensor(features_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    opacity = torch.tensor(opacity, dtype=torch.float, device="cuda")
    scaling = torch.tensor(scaling, dtype=torch.float, device="cuda")
    rotation = torch.tensor(rotation, dtype=torch.float, device="cuda")
    return xyz, features_dc, features_rest, opacity, scaling, rotation
def construct_list_of_attributes(features_dc, features_rest, scaling, rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

def similar_points_tree(point_cloud_A, point_cloud_B, threshold):

    tree = KDTree(point_cloud_B)

    distances, indices = tree.query(point_cloud_A, k=1)

    similar_indices = np.where(distances < threshold)[0]

    return similar_indices
def composition(xyz_rgb, features_dc_rgb, features_extra_rgb, opacities_rgb, scales_rgb, rots_rgb,path_2):
    xyz_binary, features_dc_binary, features_extra_binary, opacities_binary, scales_binary, rots_binary = load_ply(path_2)
    print(xyz_binary.shape)
    xyz = np.concatenate((xyz_rgb, xyz_binary), axis=0)
    features_dc = np.concatenate((features_dc_rgb,features_dc_binary),axis=0)
    features_rest = np.concatenate((features_extra_rgb,features_extra_binary),axis=0)
    opacity = np.concatenate((opacities_rgb, opacities_binary),axis=0)
    scaling = np.concatenate((scales_rgb, scales_binary),axis=0)
    rotation = np.concatenate((rots_rgb , rots_binary),axis=0)
    return xyz, features_dc, features_rest, opacity, scaling, rotation
def save_ply(xyz, features_dc, features_rest, opacity, scaling, rotation, path_save):
        mkdir_p(os.path.dirname(path_save))

        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scale = scaling.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scaling, rotation)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path_save)
# Main Procedure
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Gaussians.')
    parser.add_argument('--original_ply', type=str, required=True, help='The path to the Original Gaussians.')
    parser.add_argument('--supp_ply', type=str, required=True, help='The path to the Inpainting Gaussians')
    parser.add_argument('--save_ply', type=str, required=True, help='The path to save the Gaussians.')
    parser.add_argument('--temp_ply', type=str, default='temp.ply', help='The path to save the Temporary Gaussians.')
    parser.add_argument('--nb_points', type=int, default=100, help='Number of points for the remove_radius_outlier function.')
    parser.add_argument('--radius', type=float, default=0.1, help='Radius for the remove_radius_outlier function.')
    parser.add_argument('--threshold', type=float, default=1.0, help='Threshold for the similar_points_tree function.')

    args = parser.parse_args()


    origin = args.original_ply
    supp = args.supp_ply
    path_save = args.save_ply
    nb_points = args.nb_points
    radius = args.radius
    threshold = args.threshold
    processed_supp = args.temp_ply

    plydata = PlyData.read(supp)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    r = plydata['vertex']['red']
    g = plydata['vertex']['green']
    b = plydata['vertex']['blue']
    points = np.column_stack([x, y, z])
    colors = np.column_stack([r, g, b])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    normalized_colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(normalized_colors)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=4.0)
    pcd = pcd.select_by_index(ind)
    create_from_pcd(pcd,processed_supp)
    



    # Load Painted Gaussians and Original Gaussians
    xyz, features_dc, features_extra, opacities, scales, rots = load_ply(processed_supp)
    print("There are {} points in the inpainted Gaussians.".format(len(xyz)))
    xyz2, features_dc2, features_extra2, opacities2, scales2, rots2 = load_ply(origin)
    print("There are {} points in the original Gaussians.".format(len(xyz2)))
    
    
    # Calculate points near Inpainting Gaussians
    points = xyz2 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    xyz_similar = similar_points_tree(xyz2,xyz,threshold)
    print("There are {} similar points in the two point clouds.".format(len(xyz_similar)))

    # Remove the floaters point in the complemented area 
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    real = set(xyz_similar.tolist()).difference(set(ind))
    ind = list(real)
    print(len(ind))
    indices = np.array([True] * len(xyz2)) 
    indices[ind] = False 
    xyz_rgb = xyz2[indices]
    features_dc = features_dc2 [indices]
    features_rest = features_extra2[indices]
    opacity = opacities2[indices]
    scaling = scales2[indices]
    rotation = rots2[indices]
    print("{} floaters were removed.".format(len(xyz2)-len(xyz_rgb))) 

    # Compose Inpainting Gaussians and Original Gaussians
    xyz_rgb, features_dc, features_rest, opacity, scaling, rotation = composition(xyz_rgb, features_dc, features_rest,
                                                                                  opacity,scaling,rotation,processed_supp)
    xyz, features_dc, features_rest, opacity, scaling, rotation = convert_np_tensor(xyz_rgb, features_dc, features_rest, 
                                                                                    opacity, scaling, rotation)
    save_ply(xyz, features_dc, features_rest, opacity, scaling, rotation,path_save)

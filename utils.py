# The code for generating new triplets in the GC module is modified from the project "CPP".
# https://github.com/yzhao520/CPP


import numpy as np
import torch
import math
import torch.nn.functional as F

def mono_GC(index,inputs,opt,bound):

    inputs_temp = {}
    K=inputs[("K", 0)][:,:-1,:-1]
    K_inv=inputs[("inv_K", 0)][:,:-1,:-1]
    B, H, W = inputs[("color", 0, 0)].shape[0], inputs[("color", 0, 0)].shape[2], inputs[("color", 0, 0)].shape[3]

    perturbance = bound


    translation_sampler, euler_sampler = sample_group_pose_perturbance(B,index,perturbance)
    pose_perturbed = torch.zeros((B, 6))
    pose_perturbed[:, :3] += translation_sampler.squeeze(1)
    pose_perturbed[:, 3:] += euler_sampler.squeeze(1)

    pose_perturbed=pose_perturbed
    inputs_temp["pose_GC"]=pose_perturbed

    for f_i in opt.frame_ids:
        img = torch.cat([inputs[("color", f_i, 0)],inputs[("color_aug", f_i, 0)]])
        gt_depth = torch.cat([inputs[("pre_dgt", f_i, 0)],inputs[("pre_dgt", f_i, 0)]])
        pose_perturbed_2=torch.cat([pose_perturbed,pose_perturbed])
        K_2=torch.cat([K,K])
        K_inv_2=torch.cat([K_inv,K_inv])
        new_img, new_depth = generate_warped_img_gt(img, gt_depth, pose_perturbed_2, K_2,K_inv_2)
        inputs_temp[("color_GC", f_i, 0)]=new_img[:B]
        inputs_temp[("color_aug_GC", f_i, 0)] = new_img[B:]
        inputs_temp[("pre_dgt_GC", f_i, 0)]=new_depth[:B]


    return inputs_temp



def sample_group_pose_perturbance(B, index,bound0=0.1):
    '''
        input pose: tensor with shape [B, 6]
        return pose_perturbed: pose with shape [B, 6]
    '''
    # translation_sampler=[]
    # euler_sampler=[]

    translation_sampler = torch.rand((B, 1, 3)) * 0.
    euler_sampler = torch.zeros((B, 1, 3), dtype=torch.float32)
    for i in range(B):
        lower, upper = -bound0, bound0
        torch.manual_seed(index[i])
        euler_sampler[i]=torch.zeros((1, 1, 3), dtype=torch.float32).uniform_(lower, upper)

    return translation_sampler, euler_sampler


def generate_warped_img_gt(image, depthGT, pose_perturbed, K,K_inv):
    B, H, W = image.shape[0], image.shape[2], image.shape[3]

    pc = image_to_pointcloud(depthGT, K_inv, homogeneous_coord=True)
    dist_map = _compute_distance_map(pc, H, W)
    pc_perturbed_depth, pc_perturbed_image = _transform_pc_with_poses(pc,pose_perturbed)

    pixel_coords_perturbed = pointcloud_to_pixel_coords(pc_perturbed_image, K, image)
    image_warped = F.grid_sample(image, pixel_coords_perturbed, padding_mode="border",align_corners=True)

    dist_warped = F.grid_sample(dist_map, pixel_coords_perturbed,align_corners=True)
    depth_warped = _distance_2_depth(dist_warped,
                                     K_inv)  # no translation in our case, convert depth to distance to get rid of grid artifacts
    depth_warped[depth_warped > 1e3] = 0.
            # depth_warped= transforms.Normalize((0.5), (0.5))(depth_warped)
            # for i in range(B):
            #     depth_warped[i]=transforms.Normalize((0.5), (0.5))(depth_warped[i])
    return image_warped,depth_warped



def image_to_pointcloud(depth, K_inv, homogeneous_coord=False):
    assert depth.dim() == 4
    assert depth.size(1) == 1

    B, H, W = depth.shape[0], depth.shape[2], depth.shape[3]
    depth_v = depth.reshape(B, 1, -1)

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid_y, grid_x = torch.tensor(grid_y, dtype=torch.float32), torch.tensor(grid_x, dtype=torch.float32)
    q = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.ones_like(grid_x.reshape(-1))), dim=0).unsqueeze(
        0).expand(B, 3, H * W)
    # q = q.cuda()
    pc = torch.bmm(K_inv, q) * depth_v
    if homogeneous_coord:
        pc = torch.cat((pc, torch.ones((B, 1, depth_v.shape[-1]), dtype=pc.dtype)), dim=1)
    return pc


def _compute_distance_map(pc, H, W):
    B = pc.shape[0]
    return torch.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2).reshape(B, 1, H, W)


def _transform_pc_with_poses(pc, pose2):
    B = pose2.shape[0]


    R_1 = torch.eye(3).unsqueeze(0).expand(B, 3, 3)
    t_1 = torch.zeros(3).unsqueeze(-1).expand(B, 3, 1)

    R_2 = torch.bmm(Rotz(pose2[:, 5]), torch.bmm(Roty(pose2[:, 4]), Rotx(pose2[:, 3])))

    t_2 = pose2[:, :3].reshape(-1, 3, 1)

    R0 = torch.eye(3).unsqueeze(0).expand(B, 3, 3)
    R0[:, 0, 0] = -1  # handedness is different than our camera model
    R_1 = torch.bmm(R_1, R0)
    R_2 = torch.bmm(R_2, R0)

    cam_coord = pc[:, :3, :]

    cam_coord_depth = R_2.transpose(1, 2) @ R_1 @ cam_coord + R_2.transpose(1, 2) @ (t_2 - t_1)
    cam_coord_rgb = R_1.transpose(1, 2) @ R_2 @ cam_coord + R_1.transpose(1, 2) @ (t_1 - t_2)

    return cam_coord_depth, cam_coord_rgb



def pointcloud_to_pixel_coords(pc, K, image, normalization=True, eps=1e-8):
    B, H, W = image.shape[0], image.shape[2], image.shape[3]
    pc = pc[:, :3, :]
    pc = pc / (pc[:, -1, :].unsqueeze(1) + eps)
    p_coords = torch.bmm(K, pc)
    p_coords = p_coords[:, :2, :]
    if normalization:
        p_coords_n = torch.zeros_like(p_coords, dtype=torch.float32)
        p_coords_n[:, 0, :] = p_coords[:, 0, :] / (W - 1.)
        p_coords_n[:, 1, :] = p_coords[:, 1, :] / (H - 1.)
        p_coords_n = (p_coords_n - 0.5) * 2.
        u_proj_mask = ((p_coords_n[:, 0, :] > 1) + (p_coords_n[:, 0, :] < -1))
        p_coords_n[:, 0, :][u_proj_mask] = 2
        v_proj_mask = ((p_coords_n[:, 1, :] > 1) + (p_coords_n[:, 1, :] < -1))
        p_coords_n[:, 1, :][v_proj_mask] = 2

        p_coords_n = p_coords_n.reshape(B, 2, H, W).permute(0, 2, 3, 1)
        return p_coords_n
    else:
        return p_coords



def _distance_2_depth(distance_map, K_inv):
    B, H, W = distance_map.shape[0], distance_map.shape[2], distance_map.shape[3]

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid_y, grid_x = torch.tensor(grid_y, dtype=torch.float32), torch.tensor(grid_x, dtype=torch.float32)
    q = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.ones_like(grid_x.reshape(-1))), dim=0).unsqueeze(
        0).expand(B, 3, H * W)
    q=q
    pc = torch.bmm(K_inv, q)

    denom = torch.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + 1)  # [B, N]
    depth_map = distance_map.reshape(B, -1) / denom
    depth_map = depth_map.reshape(B, 1, H, W)

    return depth_map


def Rotx(t):
    """
        Rotation about the x-axis.
        np.array([[1,  0,  0], [0,  c, -s], [0,  s,  c]])

        -- input t shape B x 1
        -- return B x 3 x 3
    """
    B = t.shape[0]
    Rx = torch.zeros((B, 9, 1), dtype=torch.float)

    c = torch.cos(t)
    s = torch.sin(t)
    ones = torch.ones(B)

    Rx[:, 0, 0] = ones
    Rx[:, 4, 0] = c
    Rx[:, 5, 0] = -s
    Rx[:, 7, 0] = s
    Rx[:, 8, 0] = c

    Rx = Rx.reshape(B, 3, 3)

    return Rx


def Roty(t):
    """
        Rotation about the x-axis.
        np.array([[c,  0,  s], [0,  1,  0], [-s, 0,  c]])

        -- input t shape B x 1
        -- return B x 3 x 3
    """
    B = t.shape[0]
    Ry = torch.zeros((B, 9, 1), dtype=torch.float)

    c = torch.cos(t)
    s = torch.sin(t)
    ones = torch.ones(B)

    Ry[:, 0, 0] = c
    Ry[:, 2, 0] = s
    Ry[:, 4, 0] = ones
    Ry[:, 6, 0] = -s
    Ry[:, 8, 0] = c

    Ry = Ry.reshape(B, 3, 3)

    return Ry


def Rotz(t):
    """
        Rotation about the z-axis.
        np.array([[c, -s,  0], [s,  c,  0], [0,  0,  1]])

        -- input t shape B x 1
        -- return B x 3 x 3
    """
    B = t.shape[0]
    Rz = torch.zeros((B, 9, 1), dtype=torch.float)

    c = torch.cos(t)
    s = torch.sin(t)
    ones = torch.ones(B)

    Rz[:, 0, 0] = c
    Rz[:, 1, 0] = -s
    Rz[:, 3, 0] = s
    Rz[:, 4, 0] = c
    Rz[:, 8, 0] = ones

    Rz = Rz.reshape(B, 3, 3)

    return Rz

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    eps=1e-5
    min_disp = 1 / (max_depth+eps)
    max_disp = 1 / (min_depth+eps)
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / (scaled_disp+eps)
    return scaled_disp, depth



def warp_depth_with_pose(tgt_depth,pose_perturbed,K,K_inv):
    B, H, W = int(tgt_depth.shape[0]/2), tgt_depth.shape[2], tgt_depth.shape[3]
    old_depth=tgt_depth[:B,]
    pc = image_to_pointcloud_GPU(old_depth, K_inv, homogeneous_coord=True)
    dist_map = _compute_distance_map(pc, H, W)
    pc_perturbed_depth, pc_perturbed_image = _transform_pc_with_poses_GPU(pc,pose_perturbed)

    pixel_coords_perturbed = pointcloud_to_pixel_coords_1_GPU(pc_perturbed_image, K, B, H, W)
    # image_warped = F.grid_sample(image, pixel_coords_perturbed, padding_mode="border")

    dist_warped = F.grid_sample(dist_map, pixel_coords_perturbed,align_corners=True)
    depth_warped = _distance_2_depth_GPU(dist_warped,K_inv)  # no translation in our case, convert depth to distance to get rid of grid artifacts
    depth_warped[depth_warped > 1e3] = 0.

    return depth_warped,tgt_depth[B:,]



def image_to_pointcloud_GPU(depth, K_inv, homogeneous_coord=False):
    assert depth.dim() == 4
    assert depth.size(1) == 1

    B, H, W = depth.shape[0], depth.shape[2], depth.shape[3]
    depth_v = depth.reshape(B, 1, -1)

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid_y, grid_x = torch.tensor(grid_y, dtype=torch.float32), torch.tensor(grid_x, dtype=torch.float32)
    q = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.ones_like(grid_x.reshape(-1))), dim=0).unsqueeze(
        0).expand(B, 3, H * W)
    q = q.cuda()
    pc = torch.bmm(K_inv, q) * depth_v
    if homogeneous_coord:
        pc = torch.cat((pc, torch.ones((B, 1, depth_v.shape[-1]), dtype=pc.dtype).cuda()), dim=1)
    return pc



def _distance_2_depth_GPU(distance_map, K_inv):
    B, H, W = distance_map.shape[0], distance_map.shape[2], distance_map.shape[3]

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid_y, grid_x = torch.tensor(grid_y, dtype=torch.float32), torch.tensor(grid_x, dtype=torch.float32)
    q = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.ones_like(grid_x.reshape(-1))), dim=0).unsqueeze(
        0).expand(B, 3, H * W)
    q=q.cuda()
    pc = torch.bmm(K_inv, q)

    denom = torch.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + 1)  # [B, N]
    depth_map = distance_map.reshape(B, -1) / (denom+1e-8)
    depth_map = depth_map.reshape(B, 1, H, W)

    return depth_map



def _transform_pc_with_poses_GPU(pc, pose2):
    B = pose2.shape[0]

    R_1 = torch.eye(3).unsqueeze(0).expand(B, 3, 3).cuda()
    t_1 = torch.zeros(3).unsqueeze(-1).expand(B, 3, 1).cuda()

    R_2 = torch.bmm(Rotz(pose2[:, 5]), torch.bmm(Roty(pose2[:, 4]), Rotx(pose2[:, 3]))).cuda()
    t_2 = pose2[:, :3].reshape(-1, 3, 1).cuda()

    R0 = torch.eye(3).unsqueeze(0).expand(B, 3, 3).cuda()
    R0[:, 0, 0] = -1
    R_1 = torch.bmm(R_1, R0)
    R_2 = torch.bmm(R_2, R0)

    cam_coord = pc[:, :3, :]

    cam_coord_depth = R_2.transpose(1, 2) @ R_1 @ cam_coord + R_2.transpose(1, 2) @ (t_2 - t_1)
    cam_coord_rgb = R_1.transpose(1, 2) @ R_2 @ cam_coord + R_1.transpose(1, 2) @ (t_1 - t_2)

    return cam_coord_depth, cam_coord_rgb



def _transform_pc_with_poses_GPU(pc, pose2):
    B = pose2.shape[0]

    R_1 = torch.eye(3).unsqueeze(0).expand(B, 3, 3).cuda()
    t_1 = torch.zeros(3).unsqueeze(-1).expand(B, 3, 1).cuda()

    R_2 = torch.bmm(Rotz(pose2[:, 5]), torch.bmm(Roty(pose2[:, 4]), Rotx(pose2[:, 3]))).cuda()
    t_2 = pose2[:, :3].reshape(-1, 3, 1).cuda()

    R0 = torch.eye(3).unsqueeze(0).expand(B, 3, 3).cuda()
    R0[:, 0, 0] = -1
    R_1 = torch.bmm(R_1, R0)
    R_2 = torch.bmm(R_2, R0)

    cam_coord = pc[:, :3, :]

    cam_coord_depth = R_2.transpose(1, 2) @ R_1 @ cam_coord + R_2.transpose(1, 2) @ (t_2 - t_1)
    cam_coord_rgb = R_1.transpose(1, 2) @ R_2 @ cam_coord + R_1.transpose(1, 2) @ (t_1 - t_2)

    return cam_coord_depth, cam_coord_rgb




def pointcloud_to_pixel_coords_1_GPU(pc, K,B,H,W, normalization=True, eps=1e-8):
    # B, H, W = image.shape[0], image.shape[2], image.shape[3]
    pc = pc[:, :3, :]
    pc = pc / (pc[:, -1, :].unsqueeze(1) + eps)
    p_coords = torch.bmm(K, pc)
    p_coords = p_coords[:, :2, :]
    if normalization:
        p_coords_n = torch.zeros_like(p_coords, dtype=torch.float32)
        p_coords_n[:, 0, :] = p_coords[:, 0, :] / (W - 1.)
        p_coords_n[:, 1, :] = p_coords[:, 1, :] / (H - 1.)
        p_coords_n = (p_coords_n - 0.5) * 2.
        u_proj_mask = ((p_coords_n[:, 0, :] > 1) + (p_coords_n[:, 0, :] < -1))
        p_coords_n[:, 0, :][u_proj_mask] = 2
        v_proj_mask = ((p_coords_n[:, 1, :] > 1) + (p_coords_n[:, 1, :] < -1))
        p_coords_n[:, 1, :][v_proj_mask] = 2

        p_coords_n = p_coords_n.reshape(B, 2, H, W).permute(0, 2, 3, 1)
        return p_coords_n
    else:
        return p_coords
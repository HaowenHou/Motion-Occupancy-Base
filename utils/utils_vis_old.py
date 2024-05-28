from collections import OrderedDict
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.axes3d import Axes3D

from utils.quaternion import cont6d_to_matrix, cs_to_quat_y2x, matrix_to_axis_angle, matrix_to_quat, qmul, qrot, quaternion_to_matrix, radian_to_quat
from utils.utils_smpl import smpl_fk

matplotlib.use('Agg')
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

def init_axes3d(fig, radius=4, elev=25, azim=45):
    ax = fig.add_axes(Axes3D(fig, proj_type='ortho'))

    ax.set_xlim([-radius / 2, radius / 2])
    ax.set_ylim([-radius / 2, radius / 2])
    ax.set_zlim([0, radius])

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide YZ Plane
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XZ Plane
    ax.zaxis.line.set_visible(False)
    ax.set_zticks([0.0])
    ax.grid(False)
    ax.set_xticks([0.0])
    ax.set_yticks([0.0])

    # elev = 90
    # azim = 90
    ax.view_init(elev=elev, azim=azim)
    return ax

@torch.no_grad()
def visualize_motion_all_info(tensor_dict, save_path, template_joints, past_kf, future_kf):
    """
    Takes a tensor dict which contains all the frames and visualize it by mp4.
    Plots all the information including traj, vel and dir. This can be modified in function visualize_single_frame.
    """

    device = tensor_dict['joint_rot'].device
    joint_rotations_6d = tensor_dict['joint_rot']
    root_rot_vel = tensor_dict['root_rot_vel']
    root_vel = tensor_dict['root_vel']
    N_FRAMES = len(joint_rotations_6d)

    figsize=(15, 15)
    radius=4
    fig = plt.figure(figsize=figsize)
    ax = init_axes3d(fig, radius=radius)

    # Vis.
    # Absolute joints.
    absolute_joints = tensor_dict['joint_pos'] # [F, 22, 3]
    # SMPL joints and velocities.
    joint_rot_vec = matrix_to_axis_angle(cont6d_to_matrix(tensor_dict['joint_rot'])) # [F, 22, 3] TODO cont6d to quat
    joint_pos = smpl_fk(template_joints, joint_rot_vec, SMPL_PARENTS) # [F, 22, 3]
    # joints += tensor_dict['root_traj'][..., past_kf:past_kf+1, :] # [F, 22, 3] Set height.
    joint_vel = joint_pos + tensor_dict['joint_vel'] # [F, 22, 3]
    joint_vel_vis = torch.stack([joint_pos, joint_vel], dim=-2) # [F, 22, 2, 3]
    # Root and limb traj and velocities.
    root_limb_traj = torch.cat([tensor_dict['root_traj'][..., None, :], tensor_dict['limb_traj']], dim=-2) # [F-1, 2KF+1, 5, 3]
    joint_traj_vel = torch.cat([tensor_dict['root_vel'][..., None, :], tensor_dict['limb_vel']], dim=-2) # [F-1, 2KF+1, 5, 3]
    root_limb_vel = root_limb_traj[:, :-1] + joint_traj_vel # [F-1, 2KF, 5, 3]
    root_limb_vel_vis = torch.stack([root_limb_traj[:, :-1], root_limb_vel], dim=-2) # [F-1, 2KF, 5, 2, 3]
    # Root directions.
    root_dir = tensor_dict['root_dir'] # [F-1, 2KF+1, 2]
    root_dir = root_dir / 10.0
    root_dir = torch.stack([root_dir[..., 0], root_dir[..., 1], torch.zeros_like(root_dir[..., 0])], dim=-1) # [F-1, 2KF+1, 3]
    traj_z0 = tensor_dict['root_traj'].clone() # [F-1, 2KF+1, 3]
    traj_z0[..., 2] = 0.0
    root_dir_end = traj_z0 + root_dir # [F-1, 2KF+1, 3]
    root_dir_vis = torch.stack([traj_z0, root_dir_end], dim=-2) # [F-1, 2KF+1, 2, 3]
    # Rotate and translate.
    all_dir = tensor_dict['all_dir'] # [F, 2]
    all_dir_quat = cs_to_quat_y2x(all_dir) # [F, 4]
    all_traj = tensor_dict['all_traj'] # [F, 3]

    def rotate_translate(tensor, quat, translation, add_height=False):
        """Input: tensor of size [..., 3]"""
        # return tensor
        rotated = qrot(quat.expand(*tensor.size()[:-1], 4), tensor)
        if add_height:
            translation = translation.expand(tensor.size())
        else:
            translation = torch.cat([translation[..., :2], torch.zeros_like(translation[..., :1])], dim=-1)
        return rotated + translation
    joint_pos = rotate_translate(joint_pos, all_dir_quat[:, None, :], all_traj[:, None, :], add_height=True)
    joint_vel_vis = rotate_translate(joint_vel_vis, all_dir_quat[:, None, None, :], all_traj[:, None, None, :], add_height=True)
    root_limb_traj = rotate_translate(root_limb_traj, all_dir_quat[:, None, None, :], all_traj[:, None, None, :])
    root_limb_vel_vis = rotate_translate(root_limb_vel_vis, all_dir_quat[:, None, None, None, :], all_traj[:, None, None, None, :])
    root_dir_vis = rotate_translate(root_dir_vis, all_dir_quat[:, None, None, :], all_traj[:, None, None, :])
    absolute_joints = rotate_translate(absolute_joints, all_dir_quat[:, None, :], all_traj[:, None, :])

    # Convert to numpy.
    joint_pos = joint_pos.cpu().numpy() # [22, 3]
    joint_vel_vis = joint_vel_vis.cpu().numpy() # [22, 2, 3]
    root_limb_traj = root_limb_traj.cpu().numpy() # [2KF+1, 5, 3]
    root_limb_vel_vis = root_limb_vel_vis.cpu().numpy() # [2KF+1, 5, 2, 3]
    root_dir_vis = root_dir_vis.cpu().numpy() # [2KF+1, 2, 3]
    absolute_joints = absolute_joints.cpu().numpy() # [22, 3]
    def update(pivot, ax):
        for line in ax.lines:
            line.remove()
        # Skeleton.
        for i in range(1, len(SMPL_PARENTS)):
            linewidth = 2.0
            line = joint_pos[pivot, (i, SMPL_PARENTS[i]), :]
            ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='blue')
        # Joint velocities.
        for i in range(22):
            linewidth = 2.0
            line = joint_vel_vis[pivot, i]
            ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='red')
        # Trajectories.
        for i in range(5):
            linewidth = 1.0 if i == 0 else 0.3
            line = root_limb_traj[pivot, :, i]
            ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='darkblue')
        # Trajectory velocities.
        for i in range(past_kf+future_kf):
            for j in range(5):
                linewidth = 1.0
                line = root_limb_vel_vis[pivot, i, j]
                ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='darkred')
        # Ego directions.
        for i in range(past_kf+future_kf+1):
            linewidth = 0.5
            line = root_dir_vis[pivot, i]
            ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='red')
        # Absolute joints.
        for i in range(1, len(SMPL_PARENTS)):
            linewidth = 1.0
            line = absolute_joints[pivot, (i, SMPL_PARENTS[i]), :]
            ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='green')
    
    FPS=10
    ani = FuncAnimation(fig, partial(update, ax=ax), frames=N_FRAMES-1, interval=1000 / FPS, repeat=False)
    ani.save(save_path, fps=FPS)
    plt.close()



    

@torch.no_grad()
def validate_data_by_visulization(tensor_dict: OrderedDict, KEY_FRAMES: int, template_joints: torch.Tensor, smpl_parents: torch.Tensor):
    """
    Takes a tensor dict.
    Visualize only joint rotations or joint positions. And root velocity & rotation speed.
    """
    device = tensor_dict['joint_rotations'].device
    joint_rotations_6d = tensor_dict['joint_rotations']
    ego_rotation_velocities_in_window = tensor_dict['ego_rotation_velocities']
    center_velocities = tensor_dict['ego_velocities']
    joint_positions = tensor_dict['joint_positions']
    FRAMES = len(joint_rotations_6d) + 1

    current_center = torch.tensor([.0, .0, .0], device=device)
    current_theta = torch.tensor([.0], device=device)
    smpl_motion = []
    absolute_motion = []
    for i in range(FRAMES-1):
        # Current local rotation.
        current_local_rot_quat = radian_to_quat(current_theta)

        # bm
        joint_rotations_vec = matrix_to_axis_angle(cont6d_to_matrix(joint_rotations_6d[i, 1:])) # [F-1, 21, 3]
        root_rotation_quat = matrix_to_quat(cont6d_to_matrix(joint_rotations_6d[i, :1]))
        root_rotation_quat = qmul(current_local_rot_quat[None], root_rotation_quat)
        root_rotation_vec = matrix_to_axis_angle(quaternion_to_matrix(root_rotation_quat))
        joints = smpl_fk(template_joints, torch.cat([root_rotation_vec, joint_rotations_vec], dim=0)[None], smpl_parents)
        
        smpl_positions = joints + current_center[None]
        smpl_motion.append(smpl_positions)

        # positions
        rotated_positions = qrot(current_local_rot_quat[None].expand(22, 4), joint_positions[i])
        final_positions = rotated_positions + current_center[None]
        absolute_motion.append(final_positions)

        # Update theta and center.
        current_center += qrot(current_local_rot_quat, center_velocities[i, KEY_FRAMES])
        current_theta += torch.atan2(ego_rotation_velocities_in_window[i, KEY_FRAMES, 1], ego_rotation_velocities_in_window[i, KEY_FRAMES, 0])
    smpl_motion = torch.cat(smpl_motion, dim=0)
    absolute_motion = torch.stack(absolute_motion, dim=0)

    kin_list = smpl_parents.tolist()
    print('Generating mp4...')
    plot_smpl_motion(absolute_motion.cpu().numpy(), 'absolute_motion.mp4', kin_list, fps=10)
    plot_smpl_motion(smpl_motion.cpu().numpy(), 'smpl_motion.mp4', kin_list, fps=10)
    
"""Above functions take tensor dict as input."""

def plot_multiple_smpl_skeletons(save_path, kinematic_tree, joints, figsize=(10, 10), radius=4):
    matplotlib.use('Agg')

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        ax.grid(b=False)

    # (joints_num, 3)
    data = joints
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    ax.lines = []
    ax.collections = []
    # ax.view_init(elev=120, azim=-90)
    # ax.dist = 7.5

    for frame_idx in range(data.shape[0]):
        for i in range(1, len(kinematic_tree)):
            #             print(color)
            linewidth = 2.0
            line = data[frame_idx, (i, kinematic_tree[i]), :]
            ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='blue')

    # plt.axis('off')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    plt.savefig(save_path)
    plt.close()

def plot_smpl_skeleton(save_path, kinematic_tree, joints, figsize=(10, 10), radius=4):
    matplotlib.use('Agg')

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        ax.grid(b=False)

    # (joints_num, 3)
    data = joints.copy().reshape(-1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    ax.lines = []
    ax.collections = []
    # ax.view_init(elev=120, azim=-90)
    # ax.dist = 7.5

    for i in range(1, len(kinematic_tree)):
        #             print(color)
        linewidth = 2.0
        line = data[(i, kinematic_tree[i]), :]
        ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='blue')

    # plt.axis('off')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    plt.savefig(save_path)
    plt.close()

def plot_humanml3d_skeleton(save_path, kinematic_tree, joints, figsize=(10, 10), radius=4):
    matplotlib.use('Agg')

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        ax.grid(b=False)

    # (joints_num, 3)
    data = joints.copy().reshape(-1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    ax.lines = []
    ax.collections = []
    # ax.view_init(elev=120, azim=-90)
    # ax.dist = 7.5

    for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
        #             print(color)
        linewidth = 2.0
        ax.plot3D(data[chain, 0], data[chain, 1], data[chain, 2], linewidth=linewidth, color='red')

    # plt.axis('off')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    plt.savefig(save_path)
    plt.close()

def plot_trajectory(data, save_path, figsize=(10, 10), radius=4):
    """Input: list of [N, 3]"""
    matplotlib.use('Agg')

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)

    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([0, radius])
    ax.grid(b=False)

    linewidth = 2.0
    for item in data:
        ax.plot3D(item[:, 0], item[:, 1], item[:, 2], linewidth=linewidth, color='red')

    plt.savefig(save_path)
    plt.close()

def plot_smpl_motion(data, save_path, kinematic_list, fps, figsize=(15, 15), radius=4):
    fig = plt.figure(figsize=figsize)
    ax = init_axes3d(fig, radius=radius)

    frame_number = data.shape[0]
    trajec = data[:, 0, [0, 1]]

    def update(index):
        for line in ax.lines:
            line.remove()

        if index > 1:
            ax.plot3D(trajec[:index, 0], trajec[:index, 1],
                      np.zeros_like(trajec[:index, 0]), linewidth=1.0,
                      color='blue')

        for i in range(1, len(kinematic_list)):
            linewidth = 2.0
            line = data[index, (kinematic_list[i], i), :]
            ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=linewidth, color='red')

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()
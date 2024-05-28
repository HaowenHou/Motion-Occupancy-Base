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


def create_axes3d(fig, rect, radius=4, elev=25, azim=45):
    ax = Axes3D(fig, rect=rect, proj_type='ortho')

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

    ax.view_init(elev=elev, azim=azim)
    return ax

def create_fig():
    figsize=(15, 15)
    fig = plt.figure(figsize=figsize)
    # rect: [left, bottom, width, height]
    ax1 = create_axes3d(fig, rect=[0, 0, 0.3, 1])
    ax2 = create_axes3d(fig, rect=[0.35, 0, 0.3, 1])
    ax3 = create_axes3d(fig, rect=[0.7, 0, 0.3, 1])
    fig.add_axes(ax1)
    fig.add_axes(ax2)
    fig.add_axes(ax3)
    return fig

@torch.no_grad()
def visualize_amass_3views(tensor_dict, save_path, template_joints, past_kf, future_kf):
    """
    Takes a tensor dict which contains all the frames and visualize it by mp4.
    Plots all the information including traj, vel and dir. This can be modified in function visualize_single_frame.
    """

    device = tensor_dict['joint_rot'].device
    joint_rotations_6d = tensor_dict['joint_rot']
    root_rot_vel = tensor_dict['root_rot_vel']
    root_vel = tensor_dict['root_vel']
    N_FRAMES = len(joint_rotations_6d)

    figsize=(30, 10)
    radius=4
    fig = plt.figure(figsize=figsize)
    ax1 = create_axes3d(fig, rect=[0, 0, 0.33, 1], elev=25, azim=45) # 3d view
    ax2 = create_axes3d(fig, rect=[0.33, 0, 0.33, 1], elev=90, azim=-90) # top view
    ax3 = create_axes3d(fig, rect=[0.67, 0, 0.33, 1], elev=0, azim=90) # front view
    axs = [ax1, ax2, ax3]
    for ax in axs:
        fig.add_axes(ax)

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
    def update(pivot, axs):
        """axs: list of axes."""
        def update_ax(ax):
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
        for ax in axs:
            update_ax(ax)
    
    FPS=10
    ani = FuncAnimation(fig, partial(update, axs=axs), frames=N_FRAMES-1, interval=1000 / FPS, repeat=False)
    ani.save(save_path, fps=FPS)
    plt.close()
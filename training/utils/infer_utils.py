import os
import pickle
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import numpy as np
from dataset import LIMBS
from utils.quaternion import aa_to_quat, cont6d_to_quat, cos_sin_delta, fdir_to_quat, qbetween, qinv, qmul, qrot, quat_to_aa, rad_to_quat
from utils.smpl import smpl_forward
from utils.data_misc import change_system
from utils.visualize import clear_ax, create_axes3d, plot_single_pose, plot_smpl_motion


def calc_orient_in_global(orient, crt_rad):
    g2l_quat = rad_to_quat(crt_rad)
    orient = quat_to_aa(qmul(g2l_quat, aa_to_quat(orient)))
    return orient

def get_pose(poses, pos=None, rad=None, bm=None, return_verts=False):
    if rad is None:
        orient = poses[..., :3]
    else:
        orient = calc_orient_in_global(poses[..., :3], rad)
    smpl_result = smpl_forward(bm, orient=orient, bpose=poses[..., 3:72], rm_offset=True, return_verts=return_verts)
    joints = smpl_result['joints']
    if pos is not None:
        joints = joints + pos[:, None, :]
    if not return_verts:
        return joints
    else:
        verts = smpl_result['vertices']
        if pos is not None:
            verts = verts + pos[:, None, :]
        return joints, verts

def load_g_occu(mid, config): # on cpu.
    voxel, unit, llb = pickle.load(open(os.path.join(config.ASSETS.OCCUG_DIR, f'{mid:08}.pkl'), 'rb'))
    voxel = torch.from_numpy(voxel).float()
    llb = torch.from_numpy(llb).float()
    return voxel, unit, llb

def load_g_occu_ref(mid, config):
    ref   = torch.from_numpy(np.load(os.path.join(config.ASSETS.OCCUG_REF_DIR, f'{mid:08}.npy')))
    shape = ref.shape
    return ref, shape

# For checking.
def get_abs_limbs(crt_pos, crt_rad, rel_limb_pos):
    crt_pos = crt_pos.clone()
    crt_pos[..., 2] = 0.0
    g2l_quat = rad_to_quat(crt_rad)
    rotated = qrot(g2l_quat[:, None].expand(len(rel_limb_pos), rel_limb_pos.shape[1], -1), rel_limb_pos) # [1, 5, 3]
    abs_limb_pos = rotated + crt_pos[:, None] # [1, 5, 3]
    return abs_limb_pos

def draw_seq(save_path, draw_dict):
    if 'occug' in draw_dict:
        occug = draw_dict['occug']
        unit = draw_dict['unit']
        llb = draw_dict['llb'].cpu().numpy()
        def array2voxel(voxel_array):
            x, y, z          = np.where(voxel_array == 1)
            index_voxel      = np.vstack((x, y, z))
            grid_index_array = index_voxel.T
            return grid_index_array
        occu_ary = array2voxel((occug[:,:,:18]).cpu().numpy())
        occu_ary = occu_ary * unit + 0.5*unit + llb

    pose_seq = draw_dict['seq'].cpu().numpy()
    if 'start' in draw_dict:
        st_pose = draw_dict['start'].cpu().numpy()
    if 'end' in draw_dict:
        end_pose = draw_dict['end'].cpu().numpy()
    if 'tgt_seq' in draw_dict:
        tgt_seq = draw_dict['tgt_seq'].cpu().numpy()
    if 'ori_v' in draw_dict:
        scale = 3.0
        ori_v = draw_dict['ori_v']
        dv = draw_dict['dv']
        final_v = ori_v + dv
        ori_v_len = (100*torch.linalg.vector_norm(ori_v, dim=-1)).tolist()
        dv_len = (100*torch.linalg.vector_norm(dv, dim=-1)).tolist()
        final_v_len = (100*torch.linalg.vector_norm(final_v, dim=-1)).tolist()
        ori_v = ori_v.cpu().numpy() * scale
        dv = dv.cpu().numpy() * scale
        final_v = final_v.cpu().numpy() * scale

    FPS = 10
    fig = plt.figure(figsize=(15, 15))
    if 'occug' in draw_dict:
        ax = create_axes3d(fig, radius=4, azim=-80, elev=75)
        color = occu_ary[:, 2]
        ax.scatter(occu_ary[:, 0], occu_ary[:, 1], occu_ary[:, 2], s=0.8, c=color, cmap='jet')
    else:
        ax = create_axes3d(fig, radius=4)

    global update_lines
    update_lines = []
    def update(index):
        global update_lines
        clear_ax(ax, update_lines)
        if 'start' in draw_dict:
            update_lines.extend(plot_single_pose(st_pose, ax, 'green'))
        if 'end' in draw_dict:
            update_lines.extend(plot_single_pose(end_pose, ax, 'red'))
        update_lines.extend(plot_single_pose(pose_seq[index], ax, 'blue'))
        if 'tgt_seq' in draw_dict:
            ax.scatter(tgt_seq[index, :, 0], tgt_seq[index, :, 1], tgt_seq[index, :, 2], c='black')
        if 'ori_v' in draw_dict:
            root = pose_seq[index][0]
            ori_v1 = ori_v[index]
            dv1 = dv[index]
            final_v1 = final_v[index]
            line = np.stack([root, root+ori_v1], axis=0)
            update_lines.extend(ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=2.0, color='red'))
            line = np.stack([root, root+dv1], axis=0)
            update_lines.extend(ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=2.0, color='orange'))
            line = np.stack([root, root+final_v1], axis=0)
            update_lines.extend(ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=2.0, color='green'))
            vel_text = f"ori_v: {ori_v_len[index]:.1f}\ndv: {dv_len[index]:.1f}\nfinal_v: {final_v_len[index]:.1f}"
            ax.text2D(0.05, 0.9, vel_text, transform=ax.transAxes, color='black', fontsize=18)
            # arrows.append(ax.quiver(root[0], root[1], root[2], ori_v[0], ori_v[1], ori_v[2], color='red'))
            # arrows.append(ax.quiver(root[0], root[1], root[2], dv[0], dv[1], dv[2], color='blue'))
            # arrows.append(ax.quiver(root[0], root[1], root[2], final_v[0], final_v[1], final_v[2], color='green'))
        if 'hand' in draw_dict:
            hand = draw_dict['hand'].cpu().numpy() # [F, 2, 3]
            for i in range(1, len(hand)):
                line = hand[(i-1, i), 0]
                update_lines.extend(ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=2.0, color='green'))
                line = hand[(i-1, i), 1]
                update_lines.extend(ax.plot3D(line[:, 0], line[:, 1], line[:, 2], linewidth=2.0, color='orange'))
                
    ani_len = len(pose_seq)
    ani = FuncAnimation(fig, update, frames=ani_len, interval=1000 / FPS, repeat=False)
    ani.save(save_path, fps=FPS)
    plt.close()

def get_pos_dir(j_abs, fdir, idx):
    crt_pos = j_abs[:, idx, 0] # [bs, 3]
    crt_fdir = fdir[:, idx] # [bs, 3]
    return crt_pos, crt_fdir

def get_jevel(dataset, mid, st_fid, crt_jego):
    data = dataset.data_dict[mid][st_fid-1:st_fid]
    _, prev_jego, _, _, _ = dataset.parse_npy_data(data)
    return crt_jego - prev_jego

def get_g_joints_evel(j_6d, j_ego, past_kf): # Aborted.
    in_j6d = j_6d[:, past_kf]
    in_jego = j_ego[:, past_kf]
    in_jevel = in_jego - j_ego[:, past_kf-1]
    return in_j6d, in_jego, in_jevel

def get_g_joints(j_6d, j_ego, j_abs, fdir, past_kf, nxt_vel=False):
    in_j6d = j_6d[:, past_kf]
    in_jego = j_ego[:, past_kf]
    jabs = j_abs[:, past_kf-1:past_kf+1] if not nxt_vel else j_abs[:, past_kf:past_kf+2]
    j_vel = jabs[:, 1] - jabs[:, 0]
    j_fdir = fdir[:, past_kf]
    j_vel = qrot(qinv(fdir_to_quat(j_fdir[:, None])).expand(-1, j_vel.shape[1], -1), j_vel)
    return in_j6d, in_jego, j_vel

def get_g_past_traj(traj, fdir, past_kf, crt_idx):
    tmp_traj = traj[:, crt_idx-past_kf:crt_idx+1]
    gt_crt_pos = tmp_traj[:, -1]
    gt_crt_fdir = fdir[:, crt_idx]
    gt_crt_quat = fdir_to_quat(gt_crt_fdir)
    past_crt_traj = change_system(tmp_traj, gt_crt_pos, gt_crt_quat, keep_h=True)
    past_vel = past_crt_traj[:, 1:] - past_crt_traj[:, :-1]
    past_traj = past_crt_traj[:, :-1]
    return past_traj, past_vel
    
def get_g_past_limb_traj(j_abs, fdir, past_kf, crt_idx):
    tmp_traj = j_abs[:, crt_idx-past_kf:crt_idx+1, LIMBS] # [bs, past_kf+1, 4, 3]
    gt_crt_pos = j_abs[:, crt_idx, 0] # [bs, 3]
    gt_crt_fdir = fdir[:, crt_idx]
    gt_crt_quat = fdir_to_quat(gt_crt_fdir)
    past_crt_traj = change_system(tmp_traj, gt_crt_pos, gt_crt_quat, keep_h=True)
    past_vel = past_crt_traj[:, 1:] - past_crt_traj[:, :-1]
    return past_crt_traj, past_vel

def get_g_past_fdir(fdir, past_kf, crt_idx):
    past_crt_fdir = fdir[:, crt_idx-past_kf:crt_idx+1]
    past_fdir_vel = cos_sin_delta(past_crt_fdir[:, :-1], past_crt_fdir[:, 1:])
    past_fdir = past_crt_fdir[:, :-1]
    past_fdir = qrot(qinv(fdir_to_quat(past_crt_fdir[:, -1:])).expand(-1, past_kf, -1), past_fdir)
    return past_fdir, past_fdir_vel
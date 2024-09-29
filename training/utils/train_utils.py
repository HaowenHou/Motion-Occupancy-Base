from collections import OrderedDict
import torch
from dataset import LIMBS
from utils.quaternion import cont6d_to_aa, cos_sin_delta, cs_to_vec, dir_to_fdir, dir_to_quat, fdir_to_quat, qinv, qrot
from utils.smpl import smpl_forward
from utils.data_misc import change_system


def pad_fcontact_norm(input_mean, input_std):
    '''Do not normalize the contact lable.'''
    input_mean = torch.cat([
        input_mean[..., :264], torch.zeros_like(input_mean[..., :2]), input_mean[..., 264:]], dim=-1)
    input_std = torch.cat([
        input_std[..., :264], torch.ones_like(input_std[..., :2]), input_std[..., 264:]], dim=-1)
    return input_mean, input_std

def get_foot_contact(joints):
    feet_id = LIMBS[:2]
    feet_joints = joints[:, feet_id, 2]
    fcontact = torch.empty_like(feet_joints)
    fcontact[feet_joints < 0.05] = 1.0
    fcontact[(feet_joints >= 0.05) & (feet_joints < 0.1)] = 0.5
    fcontact[feet_joints >= 0.1] = 0.0
    return fcontact

def get_foot_contact_lin(joints):
    feet_id = LIMBS[:2]
    feet_joints = joints[:, feet_id, 2]
    fcontact = torch.empty_like(feet_joints)
    fcontact[feet_joints < 0.0] = 1.0
    mask = (feet_joints >= 0.0) & (feet_joints < 0.1)
    fcontact[mask] = 1.0 - 10.0 * feet_joints[mask]
    fcontact[feet_joints >= 0.1] = 0.0
    return fcontact

def get_pos_dir(j_abs, fdir, past_kf, r):
    '''
        Get root location and forward for current frame in global coordinate
    '''
    crt_pos = j_abs[:, past_kf+r, 0] # [bs, 3]
    crt_fdir = fdir[:, past_kf+r] # [bs, 3]
    return crt_pos, crt_fdir

def get_g_joints_evel(j_6d, j_ego, r, past_kf): # Aborted.
    in_j6d = j_6d[:, past_kf+r]
    in_jego = j_ego[:, past_kf+r]
    in_jevel = in_jego - j_ego[:, past_kf+r-1]
    return in_j6d, in_jego, in_jevel

def get_g_joints(j_6d, j_ego, j_abs, fdir, r, past_kf, nxt_vel=False):
    '''
        Get joint rotation, joint location and joint velocity for current frame in egocentric coordinate
    '''
    in_j6d  = j_6d[:, past_kf+r]
    in_jego = j_ego[:, past_kf+r]
    jabs    = j_abs[:, past_kf+r-1:past_kf+r+1] if not nxt_vel else j_abs[:, past_kf+r:past_kf+r+2]
    j_vel   = jabs[:, 1] - jabs[:, 0]
    j_fdir  = fdir[:, past_kf+r]
    j_vel   = qrot(qinv(fdir_to_quat(j_fdir[:, None])).expand(-1, j_vel.shape[1], -1), j_vel)
    return in_j6d, in_jego, j_vel

def get_gt_joints_evel(j_6d, j_ego, r, past_kf): # Aborted.
    gt_j6d = j_6d[:, past_kf+r+1]
    gt_jego = j_ego[:, past_kf+r+1]
    gt_jevel = gt_jego - j_ego[:, past_kf+r]
    return gt_j6d, gt_jego, gt_jevel

def get_gt_joints(j_6d, j_ego, j_abs, fdir, r, past_kf, nxt_vel=False):
    gt_j6d = j_6d[:, past_kf+r+1]
    gt_jego = j_ego[:, past_kf+r+1]
    jabs = j_abs[:, past_kf+r:past_kf+r+2] if not nxt_vel else j_abs[:, past_kf+r+1:past_kf+r+3]
    j_vel = jabs[:, 1] - jabs[:, 0]
    j_fdir = fdir[:, past_kf+r+1]
    j_vel = qrot(qinv(fdir_to_quat(j_fdir[:, None])).expand(-1, j_vel.shape[1], -1), j_vel)
    return gt_j6d, gt_jego, j_vel

def get_g_past_traj(traj, fdir, r, past_kf):
    '''
        Get historical root trajectory and velocity in egocentric coordinate
    '''
    tmp_traj = traj[:, r:r+past_kf+1]
    gt_crt_pos = tmp_traj[:, -1]
    gt_crt_fdir = fdir[:, r+past_kf]
    gt_crt_quat = fdir_to_quat(gt_crt_fdir)
    past_crt_traj = change_system(tmp_traj, gt_crt_pos, gt_crt_quat, keep_h=True)
    past_vel = past_crt_traj[:, 1:] - past_crt_traj[:, :-1]
    past_traj = past_crt_traj[:, :-1]
    return past_traj, past_vel
    
def get_g_past_limb_traj(j_abs, fdir, r, past_kf):
    '''
        Get historical joint trajectory and velocity in egocentric coordinate
    '''
    tmp_traj = j_abs[:, r:r+past_kf+1, LIMBS] # [bs, past_kf+1, 4, 3]
    gt_crt_pos = j_abs[:, r+past_kf, 0] # [bs, 3]
    gt_crt_fdir = fdir[:, r+past_kf]
    gt_crt_quat = fdir_to_quat(gt_crt_fdir)
    past_crt_traj = change_system(tmp_traj, gt_crt_pos, gt_crt_quat, keep_h=True)
    past_vel = past_crt_traj[:, 1:] - past_crt_traj[:, :-1]
    return past_crt_traj, past_vel

def get_g_past_fdir(fdir, r, past_kf):
    '''
        Get historical forward direction and angular velocity in egocentric coordinate
    '''
    past_crt_fdir = fdir[:, r:r+past_kf+1]
    past_fdir_vel = cos_sin_delta(past_crt_fdir[:, :-1], past_crt_fdir[:, 1:])
    past_fdir = past_crt_fdir[:, :-1]
    past_fdir = qrot(qinv(fdir_to_quat(past_crt_fdir[:, -1:])).expand(-1, past_kf, -1), past_fdir)
    return past_fdir, past_fdir_vel

def get_ss_past_traj(past_traj, past_vel, crt_height, nxt_traj, fdir_vel): # Aborted.
    crt = torch.zeros_like(nxt_traj)
    crt[:, 2:3] = crt_height
    fdir_vel_quat = dir_to_quat(fdir_vel)
    tmp_traj = torch.cat([past_traj[:, 1:], crt[:, None], nxt_traj[:, None]], dim=1)
    past_crt_traj = change_system(tmp_traj, nxt_traj, fdir_vel_quat, keep_h=True)
    past_traj = past_crt_traj[:, :-1]
    past_vel = past_crt_traj[:, 1:] - past_crt_traj[:, :-1]
    return past_traj, past_vel
    
def get_ss_past_traj(past_traj, past_vel, crt_height, vel, nxt_traj, fdir_vel):
    crt = torch.zeros_like(nxt_traj)
    crt[:, 2:3] = crt_height
    fdir_vel_quat = dir_to_quat(fdir_vel)
    tmp_traj = torch.cat([past_traj[:, 1:], crt[:, None], nxt_traj[:, None]], dim=1)
    past_crt_traj = change_system(tmp_traj, nxt_traj, fdir_vel_quat, keep_h=True)
    past_traj = past_crt_traj[:, :-1]
    vels = torch.cat([past_vel[:, 1:], vel[:, None]], dim=1)
    vels_cano = qrot(qinv(fdir_vel_quat[:, None]).expand(-1, vels.shape[1], -1), vels)
    return past_traj, vels_cano

def get_ss_past_limb_traj(past_limb_traj, crt_ss_jegor, nxt_jegor, nxt_traj, fdir_vel): # Aborted.
    crt = crt_ss_jegor[:, None, LIMBS]
    past = torch.cat([past_limb_traj[:, 1:], crt], dim=1)
    past = change_system(past, nxt_traj, dir_to_quat(fdir_vel), keep_h=True)
    nxt = nxt_jegor[:, None, LIMBS]
    past_crt_traj = torch.cat([past, nxt], dim=1)
    past_traj = past_crt_traj[:, :-1]
    past_vel = past_crt_traj[:, 1:] - past_crt_traj[:, :-1]
    return past_traj, past_vel

def get_ss_past_limb_traj(past_traj, nxt_traj, nxt_limb_traj, past_vel, nxt_vel, fdir_vel):
    limb_traj = torch.cat([past_traj[:, 1:], nxt_limb_traj[:, None]], dim=1)
    limb_traj = change_system(limb_traj, nxt_traj, dir_to_quat(fdir_vel), keep_h=True)
    limb_vel = torch.cat([past_vel[:, 1:], nxt_vel[:, None]], dim=1)
    ori_shape = limb_vel.shape
    limb_vel = limb_vel.view(len(limb_vel), -1, 3)
    limb_vel = qrot(qinv(dir_to_quat(fdir_vel[:, None])).expand(-1, limb_vel.shape[1], -1), limb_vel)
    limb_vel = limb_vel.view(ori_shape)
    return limb_traj, limb_vel

def get_ss_past_vel(past_fdir, past_fdir_vel, fdir_vel):
    ydir = torch.zeros_like(fdir_vel)
    ydir[:, 1] = 1.0
    past_crt_fdir = torch.cat([past_fdir[:, 1:], ydir[:, None], dir_to_fdir(fdir_vel[:, None])], dim=1)
    past_fdir_vel = cos_sin_delta(past_crt_fdir[:, :-1], past_crt_fdir[:, 1:])
    past_fdir = past_crt_fdir[:, :-1]
    past_fdir = qrot(qinv(fdir_to_quat(past_crt_fdir[:, -1:])).expand(-1, past_fdir.shape[1], -1), past_fdir)
    return past_fdir, past_fdir_vel

def get_gt_fut_traj(traj, fdir, r, past_kf, future_kf):
    crt_fut_traj = change_system(traj[:, past_kf+r:past_kf+r+future_kf+1], 
                                traj[:, past_kf+r],
                                fdir_to_quat(fdir[:, past_kf+r]), keep_h=True)
    fut_traj = crt_fut_traj[:, 1:]
    fut_vel = crt_fut_traj[:, 1:] - crt_fut_traj[:, :-1]
    return fut_traj, fut_vel

def get_gt_fut_limb_traj(j_abs, fdir, r, past_kf, future_kf):
    crt_traj = j_abs[:, past_kf+r, 0] # [bs, 3]
    limb_traj = j_abs[:, past_kf+r:past_kf+r+future_kf+1, LIMBS] # [bs, future_kf+1, 4, 3]
    crt_fut_traj = change_system(limb_traj, crt_traj,
                                fdir_to_quat(fdir[:, past_kf+r]), keep_h=True)
    fut_traj = crt_fut_traj[:, 1:]
    fut_vel = crt_fut_traj[:, 1:] - crt_fut_traj[:, :-1]
    return fut_traj, fut_vel

def get_gt_fut_fdir(fdir, r, past_kf, future_kf):
    crt_fut_fdir = fdir[:, past_kf+r: past_kf+r+future_kf+1]
    fut_fdir = crt_fut_fdir[:, 1:]
    fut_fdir = qrot(qinv(fdir_to_quat(crt_fut_fdir[:, :1])).expand(-1, future_kf, -1), fut_fdir)
    fut_fdir_vel = cos_sin_delta(crt_fut_fdir[:, :-1], crt_fut_fdir[:, 1:])
    return fut_fdir, fut_fdir_vel

def get_ssgt_fut_traj(traj, crt_pos, crt_fdir, past_kf, future_kf, r):
    tmp_traj = torch.cat([crt_pos[:, None], traj[:, past_kf+r+1:past_kf+r+future_kf+1]], dim=1)
    crt_fut_traj = change_system(tmp_traj, crt_pos, fdir_to_quat(crt_fdir), keep_h=True)
    fut_traj = crt_fut_traj[:, 1:]
    fut_vel = crt_fut_traj[:, 1:] - crt_fut_traj[:, :-1]
    return fut_traj, fut_vel

def get_ssgt_fut_limb_traj(crt_ss_jegor, j_abs, crt_pos, crt_fdir, past_kf, future_kf, r):
    j_abs = j_abs[:, past_kf+r+1:past_kf+r+future_kf+1, LIMBS] # [bs, future_kf, 4, 3]
    j_fut = change_system(j_abs, crt_pos, fdir_to_quat(crt_fdir), keep_h=True) # [bs, future_kf, 4, 3]
    traj = torch.cat([crt_ss_jegor[:, None, LIMBS], j_fut], dim=1) # [bs, fkf+1, 4, 3]
    fut_vel = traj[:, 1:] - traj[:, :-1]
    fut_traj = traj[:, 1:]
    return fut_traj, fut_vel

def get_ssgt_fut_limb_traj(crt_limbs, j_abs, crt_pos, crt_fdir, past_kf, future_kf, r):
    j_abs = j_abs[:, past_kf+r+1:past_kf+r+future_kf+1, LIMBS] # [bs, future_kf, 4, 3]
    j_fut = change_system(j_abs, crt_pos, fdir_to_quat(crt_fdir), keep_h=True) # [bs, future_kf, 4, 3]
    traj = torch.cat([crt_limbs[:, None], j_fut], dim=1) # [bs, fkf+1, 4, 3]
    fut_vel = traj[:, 1:] - traj[:, :-1]
    fut_traj = traj[:, 1:]
    return fut_traj, fut_vel

def get_ssgt_fut_fdir(fdir, crt_fdir, past_kf, future_kf, r):
    fut_fdir = fdir[:, past_kf+r+1: past_kf+r+future_kf+1]
    crt_fut_fdir = torch.cat([crt_fdir[:, None], fut_fdir], dim=1)
    fut_fdir = qrot(qinv(fdir_to_quat(crt_fdir[:, None])).expand(-1, future_kf, -1), fut_fdir)
    fut_fdir_vel = cos_sin_delta(crt_fut_fdir[:, :-1], crt_fut_fdir[:, 1:])
    return fut_fdir, fut_fdir_vel

def update_pos(crt_pos, crt_fdir, nxt_traj):
    quat = fdir_to_quat(crt_fdir)
    trans = qrot(quat, nxt_traj)
    crt_pos = crt_pos.clone()
    crt_pos[..., 2] = 0.0
    crt_pos = crt_pos + trans
    return crt_pos

def update_fdir(crt_fdir, fdir_vel):
    quat = dir_to_quat(fdir_vel)
    crt_fdir = qrot(quat, crt_fdir)
    return crt_fdir

def get_g_tgt(tgt_limb_abs, traj, fdir, past_kf, r):
    g_pos = traj[:, past_kf+r]
    g_fdir = fdir[:, past_kf+r]
    tgt = change_system(tgt_limb_abs, g_pos, fdir_to_quat(g_fdir), keep_h=True)
    return tgt

def get_ss_tgt(tgt_limb_abs, crt_pos, crt_fdir):
    '''
        Get target joint location in egocentric coordinate
    '''
    tgt = change_system(tgt_limb_abs, crt_pos, fdir_to_quat(crt_fdir), keep_h=True)
    return tgt

def calc_jegor(cont6d, height, bm):
    joints = smpl_forward(bm, poses=cont6d_to_aa(cont6d).flatten(1), rm_offset=True)['joints'] # [bs, 22, 3]
    joints[..., 2:3] += height[:, None]
    return joints

def get_in_vec(in_dict):
    '''
        Aggregate input dict into input vector
    '''
    tensor_lst = [in_dict['j6d'], in_dict['jego'], in_dict['jevel']]
    if 'fcontact' in in_dict:
        tensor_lst.append(in_dict['fcontact'])
    tensor_lst.extend([
        in_dict['crt_pos'][..., 2:3], in_dict['past_traj'], in_dict['past_vel'],
        in_dict['past_fdir'][..., :2], in_dict['past_fdir_vel'][..., :2]])
    if 'past_limb_traj' in in_dict:
        tensor_lst.extend([in_dict['past_limb_traj'], in_dict['past_limb_vel']])
    if 'tgt_limbs' in in_dict:
        tensor_lst.append(in_dict['tgt_limbs'])
    if 'occu_l' in in_dict:
        tensor_lst.append(in_dict['occu_l'])
    if 'close' in in_dict:
        tensor_lst.append(in_dict['close'])
    if 'hand' in in_dict:
        tensor_lst.append(in_dict['hand'])
    in_vec = torch.cat([t.flatten(1).detach() for t in tensor_lst], dim=-1)
    return in_vec

def get_gt_vec(gt_dict):
    tensor_lst = [gt_dict['j6d'], gt_dict['jego'], gt_dict['jevel'], 
                  gt_dict['fut_traj'], gt_dict['fut_vel'],
                  gt_dict['fut_fdir'][..., :2], gt_dict['fut_fdir_vel'][..., :2]]
    if 'fut_limb_traj' in gt_dict:
        tensor_lst.extend([gt_dict['fut_limb_traj'], gt_dict['fut_limb_vel']])
    gt_vec = torch.cat([t.flatten(1).detach() for t in tensor_lst], dim=-1)
    return gt_vec

# For check.
def output_from_gt(j_6d, j_ego, gt_fut_traj, gt_fut_vel, gt_fut_fdir_vel, r, past_kf):
    nxt_j6d = j_6d[:, past_kf+r+1]
    nxt_jego = j_ego[:, past_kf+r+1]
    nxt_jevel = nxt_jego - j_ego[:, past_kf+r]
    nxt_traj = gt_fut_traj[:, 0]
    vel = gt_fut_vel[:, 0]
    fdir_vel = gt_fut_fdir_vel[:, 0]
    return nxt_j6d, nxt_jego, nxt_jevel, nxt_traj, vel, fdir_vel

def get_from_outputs(pred_vec, future_kf):
    j6d = pred_vec[:, :132].view(-1, 22, 6)
    jego = pred_vec[:, 132:198].view(-1, 22, 3)
    jevel = pred_vec[:, 198:264].view(-1, 22, 3)
    win = pred_vec[:, 264:]
    # [3, 3, 2, 2], [4*3, 4*3]
    nxt_traj = win[:, :3]
    vel = win[:, 3*future_kf:3*future_kf+3]
    fdir_vel = cs_to_vec(win[:, 8*future_kf:8*future_kf+2])
    return j6d, jego, jevel, nxt_traj, vel, fdir_vel

def get_limb_from_outputs(pred_vec, future_kf):
    limbs = pred_vec[:, 264+10*future_kf:]
    nxt_limb_traj = limbs[:, :4*3].view(-1, 4, 3)
    limb_vel = limbs[:, 4*3*future_kf:4*3*future_kf+4*3].view(-1, 4, 3)
    return nxt_limb_traj, limb_vel

def set_io_dims(config):
    PAST_KF = config.TRAIN.PAST_KF
    FUTURE_KF = config.TRAIN.FUTURE_KF
    j_dim = 264
    if config.TRAIN.get('USE_FCONTACT', False):
        j_dim += 2
    config.MODEL.JOINT_INDIM = j_dim
    in_dim = j_dim + 1 + PAST_KF*(3+3+2+2) # joints, height, past window.
    if config.TRAIN.USE_LIMB_TRAJ:
        in_dim += 4*3 + PAST_KF*(4*3+4*3) # current limbs, past window.
    if config.TRAIN.USE_TARGET:
        in_dim += 5*3
        if config.TRAIN.get('CLOSE_SW', False) and config.TRAIN.get('CLOSE_LABEL', False):
            in_dim += 1
    if config.TRAIN.USE_VOX:
        if config.TRAIN.get('USE_BPS', True):
            in_dim += int(config.ASSETS.BASIS_PATH.split('.')[0].split('_')[1])
        else:
            gsize = config.TRAIN.GRID_SIZE
            in_dim += gsize[0] * gsize[1] * gsize[2]
    if config.TRAIN.get('USE_HAND', False):
        in_dim += (config.TRAIN.HAND_PAST_KF + config.TRAIN.HAND_FUTURE_KF + 1) * 2 * 3
    out_dim = 264 + FUTURE_KF*(3+3+2+2) # joints, future window.
    if config.TRAIN.USE_LIMB_TRAJ:
        out_dim += FUTURE_KF*(4*3+4*3) # future limbs.
    config.MODEL.IN_DIM = in_dim
    config.MODEL.OUT_DIM = out_dim
    config.MODEL.TRAJ_DIM = 1+PAST_KF*(3+3+2+2) + PAST_KF*(4*3+4*3)+4*3 # 47
    return in_dim, out_dim

def dist_loss(vec, future_kf):
    j1 = vec[:, :132]
    j2 = vec[:, 132:264]
    win = vec[:, 264:]
    traj = win[:, :6*future_kf]
    fdir = win[:, 6*future_kf:(6+4)*future_kf]
    l1_lst = [j1, fdir]
    l2_lst = [j2, traj]
    if win.shape[1] > 10*future_kf:
        limb_traj = win[:, 10*future_kf:]
        l2_lst.append(limb_traj)
    l1 = torch.cat(l1_lst, dim=1)
    l2 = torch.cat(l2_lst, dim=1)
    return l1, l2

def calc_loss(pred, gt, past_kf, l1_ratio=1.0, l2_ratio=1.0):
    gt_l1, gt_l2 = dist_loss(gt, past_kf)
    pred_l1, pred_l2 = dist_loss(pred, past_kf)
    l1_loss = l1_ratio * torch.nn.functional.l1_loss(pred_l1, gt_l1)
    l2_loss = l2_ratio * torch.nn.functional.mse_loss(pred_l2, gt_l2)
    return l1_loss, l2_loss

def calc_grid_loss(abs_pose, occu_g, llb, occu_g_ref, occu_shape, unit, device):
    bs = len(occu_g)
    abs_pose = abs_pose - llb[:, None]
    coord = abs_pose // unit
    coord = coord.long()
    coord = torch.clamp(coord, min=0)
    x_mask = torch.logical_and(coord[..., 0] >= 0, coord[..., 0] < occu_shape[:, None, 0])
    y_mask = torch.logical_and(coord[..., 1] >= 0, coord[..., 1] < occu_shape[:, None, 1])
    z_mask = torch.logical_and(coord[..., 2] >= 0, coord[..., 2] < occu_shape[:, None, 2])
    mask   = torch.logical_and(torch.logical_and(x_mask, y_mask), z_mask)
    bidx, gidx = torch.where(mask)
    rmask = occu_g[bidx, coord[bidx, gidx, 0], coord[bidx, gidx, 1], coord[bidx, gidx, 2]] > 0
    bidx  = bidx[rmask]
    gidx  = gidx[rmask]
    ref   = occu_g_ref[bidx, coord[bidx, gidx, 0], coord[bidx, gidx, 1], coord[bidx, gidx, 2]]
    loss  = torch.linalg.norm(abs_pose[bidx, gidx] - ref, dim=-1).sum() / (bs * abs_pose.shape[1])
    return loss

def vec_to_ctrl(config, in_vec):
    j_dim = config.MODEL.JOINT_INDIM
    ctrl_dict = OrderedDict({'joints': in_vec[:, :j_dim]})
    past_kf = config.TRAIN.PAST_KF
    bs = in_vec.shape[0]
    traj_dim = config.MODEL.TRAJ_DIM
    ctrl_dict['traj'] = in_vec[:, j_dim:j_dim+traj_dim]
    if config.STAGE == 'TRAIN' and config.TRAIN.get('DROP_PTRAJ', 0.0) > 0.0:
        mask = torch.rand(bs, device=in_vec.device) < config.TRAIN.DROP_PTRAJ
        ctrl_dict['traj'] = ctrl_dict['traj'].masked_fill(mask[:, None], 0.0)
    base_dims = j_dim + traj_dim
    if config.TRAIN.USE_TARGET:
        ctrl_dict['tgt'] = in_vec[:, base_dims:base_dims+15]
        if config.STAGE == 'TRAIN' and (not config.TRAIN.CALC_NORM) and config.TRAIN.get('DROP_TGT', 0.0) > 0.0:
            mask = torch.rand(bs, device=in_vec.device) < config.TRAIN.DROP_TGT
            ctrl_dict['tgt'] = ctrl_dict['tgt'].masked_fill(mask[:, None], 0.0)
        if config.STAGE == 'INFER' and config.INFER.get('DROP_TGT', 0.0) > 0.0:
            mask = torch.rand(bs, device=in_vec.device) < config.INFER.DROP_TGT
            ctrl_dict['tgt'] = ctrl_dict['tgt'].masked_fill(mask[:, None], 0.0)
        if config.STAGE == 'TRAIN' and config.TRAIN.get('TGT_ROOT_ONLY', False):
            ctrl_dict['tgt'][:, 3:] = 0.0
        if config.STAGE == 'INFER' and config.INFER.get('TGT_ROOT_ONLY', False):
            ctrl_dict['tgt'][:, 3:] = 0.0
        base_dims += 15
    if config.TRAIN.USE_VOX:
        if config.TRAIN.get('USE_BPS'):
            gsize_dim = int(config.ASSETS.BASIS_PATH.split('.')[0].split('_')[1])
        else:
            gsize = config.TRAIN.GRID_SIZE
            gsize_dim = gsize[0] * gsize[1] * gsize[2]
        ctrl_dict['vox'] = in_vec[:, base_dims:base_dims+gsize_dim]
        base_dims += gsize_dim
    if config.TRAIN.get('CLOSE_SW', False) and config.TRAIN.get('CLOSE_LABEL', False):
        ctrl_dict['tgt'] = torch.cat([ctrl_dict['tgt'], in_vec[:, base_dims:base_dims+1]], dim=-1)
        base_dims += 1
    if config.TRAIN.get('USE_HAND', False):
        hand_dim = (config.TRAIN.HAND_PAST_KF + config.TRAIN.HAND_FUTURE_KF + 1) * 2 * 3
        ctrl_dict['hand'] = in_vec[:, base_dims:base_dims+hand_dim]
        base_dims += hand_dim
    assert base_dims == config.MODEL.IN_DIM, (base_dims, config.MODEL.IN_DIM)
    return ctrl_dict
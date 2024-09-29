import numpy as np
import torch
import tqdm
from utils.infer_utils import get_pose
from utils.metric_utils import calculate_distances
from utils.occu import get_grid, query_bps_batched, query_occu, query_bps, get_bps, query_occu_batched, query_occu_batched_field, project, get_delta_v, query_bps_batched_field
from utils.train_utils import calc_jegor, get_foot_contact, get_from_outputs, get_in_vec, get_limb_from_outputs, get_ss_past_limb_traj, get_ss_past_traj, get_ss_past_vel, get_ss_tgt, pad_fcontact_norm, set_io_dims, update_fdir, update_pos, vec_to_ctrl
from utils.infer_utils import *

from utils.quaternion import *
from dataset import ELIMBS, LIMBS, MphaseDataset, collate_fn, normalize, denormalize
import smplx1 as smplx

@torch.no_grad()
def infer_dynamic(bm, network, config, norm, device,
                  init_pos, init_fdir, tgt_limb_abs=None,
                  occug_lst=None, llb=None, unit=None, length=None):
    if isinstance(llb, np.ndarray):
        llb = torch.from_numpy(llb).float().to(device)
    input_mean, input_std, output_mean, output_std = norm
    if config.TRAIN.get('USE_FIELD', False):
        vel_mean = output_mean[264:267].to(device) # get mean of vel
        vel_std = output_std[264:267].to(device) # get std of vel
        vel_norm = (vel_mean, vel_std)
    network.eval()
    past_kf = config.TRAIN.PAST_KF
    future_kf = config.TRAIN.FUTURE_KF
    bs = 1

    USE_EVEL = config.TRAIN.USE_EVEL
    USE_NORM = config.TRAIN.USE_NORM
    USE_FCONTACT = config.TRAIN.get('USE_FCONTACT', False)
    USE_LIMB_TRAJ = config.TRAIN.USE_LIMB_TRAJ
    USE_TGT = config.TRAIN.USE_TARGET
    USE_VOX = config.TRAIN.USE_VOX
    USE_BPS = config.TRAIN.get('USE_BPS', False)
    USE_SMPL_UPDATE = config.TRAIN.get('USE_SMPL_UPDATE', False) and bm is not None
    if config.TRAIN.USE_VOX:
        if USE_BPS: 
            bps = torch.from_numpy(np.load(config.ASSETS.BASIS_PATH)).float().to(device)
            bps = get_bps(bps, config.TRAIN.GRID_UNIT, config.TRAIN.GRID_SIZE)
        else:
            grid_size = config.TRAIN.GRID_SIZE
            grid, oidx, occu_l = get_grid(unit=config.TRAIN.GRID_UNIT, size=grid_size, device=device)
            grid = grid[None].expand(bs, -1, -1).contiguous()
            oidx = oidx[None].expand(bs, -1, -1).contiguous()
            occu_l = occu_l[None].expand(bs, -1, -1, -1).contiguous()

    first_pose  = torch.load(config.INFER.FIRST_PATH)
    if USE_TGT:
        tgt_limb_abs = tgt_limb_abs.to(device)
    else:
        tgt_limb_abs = None
    crt_j6d     = first_pose[:22 * 6].view(1, 22, 6).expand(bs, -1, -1).to(device) # bs, 22, 6
    crt_jego    = first_pose[22 * 6:].view(1, 22, 3).expand(bs, -1, -1).to(device) # bs, 22, 3
    height      = crt_jego[:, 0, 2]
    limb_height = crt_jego[:, LIMBS, 2]
    crt_jevel   = torch.zeros(bs, 22, 3, device=device) # bs, 22, 3
    # crt_fdir    = torch.zeros(bs, 3, device=device)  # bs, 1, 3
    # crt_fdir[..., 1] = -1.0
    crt_fdir = init_fdir.expand(bs, -1).to(device)
    past_traj   = torch.zeros(bs, past_kf, 3, device=device) # bs, kf, 3
    past_traj[..., 2] = height[:, None]
    past_vel    = torch.zeros(bs, past_kf, 3, device=device) # bs, kf, 3
    past_fdir   = torch.zeros(bs, past_kf, 3, device=device) # bs, kf, 3
    past_fdir[..., 1] = 1.0
    past_fdir_vel = torch.zeros(bs, past_kf, 3, device=device) # bs, kf, 3
    past_fdir_vel[..., 0] = 1.0
    if USE_LIMB_TRAJ:
        past_limb_traj = torch.zeros(bs, past_kf + 1, 4, 3, device=device) # bs, kf+1, 3
        past_limb_traj[..., 2] = limb_height[:, None]
        past_limb_vel = torch.zeros(bs, past_kf, 4, 3, device=device) # bs, kf, 3
    # crt_pos = torch.zeros(bs, 3, device=device)
    crt_pos = init_pos.expand(bs, -1).to(device)
    crt_pos[:, 2] = height

    results = {
        'nxt_j6d': [crt_j6d.clone().detach()[:, None]],
        'nxt_pos': [crt_pos.clone().detach()[:, None]], 
        'nxt_fdir': [crt_fdir.clone().detach()[:, None]], 
    }
    # if not USE_BPS:
    #     results['occul'] = []
    if USE_TGT:
        results['tgt'] = []

    # Main loop.
    for i in tqdm.trange(length):
        if USE_TGT:
            results['tgt'].append(tgt_limb_abs.clone().cpu()[:, None])

        # Prepare inputs.
        in_dict = {'j6d': crt_j6d, 'jego': crt_jego, 'jevel': crt_jevel}
        in_dict.update({'crt_pos': crt_pos})
        in_dict.update({'past_traj': past_traj, 'past_vel': past_vel, 'past_fdir': past_fdir, 'past_fdir_vel': past_fdir_vel})
        if USE_FCONTACT:
            fcontact = get_foot_contact(crt_jego)
            in_dict['fcontact'] = fcontact
        if USE_LIMB_TRAJ:
            in_dict.update({'past_limb_traj': past_limb_traj, 'past_limb_vel': past_limb_vel})
        if USE_TGT:
            tgt_limbs = get_ss_tgt(tgt_limb_abs, crt_pos, crt_fdir)
            in_dict.update({'tgt_limbs': tgt_limbs})
                
        if USE_VOX:
            if USE_BPS:
                if config.TRAIN.get('USE_FIELD', False):
                    occu_l_lst, d_vecs = query_bps_on_mesh_field(crt_pos, fdir_to_rad(crt_fdir), bps, config.TRAIN.GRID_UNIT, device, scene_mesh, scene, sdf_dict)
                else:
                    occu_l_lst = query_bps_on_mesh(crt_pos, fdir_to_rad(crt_fdir), bps, config.TRAIN.GRID_UNIT, device, scene_mesh, sdf_dict)
            else:                        
                if config.TRAIN.get('USE_FIELD', False):
                    query_result = query_occu_batched_field(occug_lst[i].unsqueeze(0), llb, crt_pos, fdir_to_rad(crt_fdir), unit, grid_size, device, 
                        grid=grid, oidx=oidx, occu_l=occu_l,
                        jego=crt_jego, config=config, tgt_limb_abs=tgt_limb_abs)
                    # query_result = query_occu_batched_field_mesh(crt_pos, fdir_to_rad(crt_fdir),
                    #     unit=config.TRAIN.GRID_UNIT, grid_size=config.TRAIN.GRID_SIZE, device=device, grid=grid, oidx=oidx, occul=occu_l,
                    #     sdf_dict=sdf_dict, fill=0.31 + 0.3 * np.random.rand(), limit_min=limit_min, limit_max=limit_max,
                    #     jego=crt_jego, config=config, tgt_limb_abs=tgt_limb_abs)
                    occu_l_lst, d_vecs = query_result['occul'], query_result['d_vecs']
                    if 'close' in query_result and config.TRAIN.get('CLOSE_LABEL', False):
                        close = query_result['close']
                    # results['occul'].append(occul_grid.clone().cpu().detach()[:, None])
                    # v = crt_jevel[:, 0] if i == 0 else nxt_traj # [bs, 3]
                    # delta_v = get_delta_v(d_vecs, v) # [bs, 3] before multiplied by alpha
                    # vel_std = output_std[264:266].to(device) # get std of vel
                    # delta_v[:, :2] = delta_v[:, :2] / vel_std
                else:
                    occu_l_lst = query_occu_batched(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir), config.TRAIN.GRID_UNIT, None, device, grid, oidx, occu_l)
            occu_l_lst = occu_l_lst.flatten(1).to(device)
            assert torch.isnan(occu_l_lst).sum() == 0
            in_dict['occu_l'] = occu_l_lst
        if USE_TGT and 'close' in locals():
            in_dict['close'] = close
        in_vec = get_in_vec(in_dict)
        
        # Infer.
        if USE_NORM:
            ndims = input_mean.shape[0]
            in_vec[:, :ndims] = normalize(in_vec[:, :ndims], input_mean, input_std)
        if config.TRAIN.SEP_CTRLS:
            ctrl_dict = vec_to_ctrl(config, in_vec)
            if config.TRAIN.get('USE_FIELD', False):
                vel = past_vel.view(-1, 3)
                pred_dict = network(ctrl_dict, d_vecs, vel_norm, vel, return_vel=True)
                pred_vec = pred_dict['pred']
            else:
                pred_vec = network(ctrl_dict)
        if USE_NORM:
            pred_vec = denormalize(pred_vec, output_mean, output_std)
        nxt_j6d, nxt_jego, nxt_jevel, nxt_traj, vel, fdir_vel = get_from_outputs(pred_vec, future_kf)
        if USE_LIMB_TRAJ:
            nxt_limbs, nxt_limb_vel = get_limb_from_outputs(pred_vec, future_kf)

        # Update.
        # Update current info. (nxt -> crt)
        nxt_pos  = update_pos(crt_pos, crt_fdir, nxt_traj)
        nxt_fdir = update_fdir(crt_fdir, fdir_vel)
        results['nxt_j6d'].append(nxt_j6d.clone().detach()[:, None])
        results['nxt_pos'].append(nxt_pos.clone().detach()[:, None])
        results['nxt_fdir'].append(nxt_fdir.clone().detach()[:, None])
        if USE_SMPL_UPDATE:
            nxt_jego = get_pose(cont6d_to_aa(nxt_j6d).flatten(1), bm=bm)
        past_traj, past_vel = get_ss_past_traj(past_traj, past_vel, crt_pos[:, 2:3], vel, nxt_traj, fdir_vel)
        if USE_LIMB_TRAJ:
            past_limb_traj, past_limb_vel = \
                get_ss_past_limb_traj(past_limb_traj, nxt_traj, nxt_limbs, past_limb_vel, nxt_limb_vel, fdir_vel)
        past_fdir, past_fdir_vel = get_ss_past_vel(past_fdir, past_fdir_vel, fdir_vel)
        crt_j6d = nxt_j6d
        crt_jego = nxt_jego
        crt_jevel = nxt_jevel
        crt_pos = nxt_pos
        crt_fdir = nxt_fdir
        
    if 'tgt' in results:
        results['tgt'].append(results['tgt'][-1])
    if 'occul' in results:
        results['occul'].append(torch.zeros_like(results['occul'][-1]))
    for key in results.keys():
        results[key] = torch.cat(results[key], dim=1)
    if USE_TGT and config.INFER.get('TGT_ROOT_ONLY', False):
        results['tgt'] = results['tgt'][:, :, 0:1]
    
    return results

def distance_point_line_segment(point, line_start, line_end):
    """
    Calculate the shortest distance from a point to a line segment.
    """
    # Convert points to numpy arrays for easier manipulation
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    # Vector from line_start to line_end
    line_vec = line_end - line_start
    # Vector from line_start to point
    point_vec = point - line_start

    # Calculate squared length of the line segment
    line_len_sq = np.dot(line_vec, line_vec)

    # Project point_vec onto line_vec
    projection = np.dot(point_vec, line_vec) / line_len_sq

    # Check if the projection lies within the line segment
    if projection < 0.0:
        # Closest to line_start
        # closest_point = line_start
        result = 9999.9
    elif projection > 1.0:
        # Closest to line_end
        # closest_point = line_end
        result = 9999.9
    else:
        # Closest point lies within the segment
        closest_point = line_start + projection * line_vec
        result = np.linalg.norm(point - closest_point)

    # Distance from the point to the closest point on the line segment
    return result

# Function to get line segment points based on rotation angle
def get_line_segment_points(angle, length):
    x1 = length * np.cos(angle)
    y1 = length * np.sin(angle)
    return (x1, y1, 0), (0, 0, 0)

def draw_door_seg(rad, grid_t, llb, x_dim, y_dim, UNIT, LINE_LENGTH, MAX_DISTANCE):
    line_start, line_end = get_line_segment_points(rad, LINE_LENGTH)
    for x in range(x_dim):
        for y in range(y_dim):
            # Convert grid indices to actual coordinates
            point = (x * UNIT + llb[0], y * UNIT + llb[1], 0)
            # Calculate distance to the line segment
            distance = distance_point_line_segment(point, line_start, line_end)
            # Update grid value if distance is less than MAX_DISTANCE
            if distance < MAX_DISTANCE:
                grid_t[x, y, :] = 1
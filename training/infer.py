import argparse
from functools import partial
# from multiprocessing import Pool, set_start_method
import os
import pickle
import random
import numpy as np
from omegaconf import OmegaConf
import torch
from models import get_model
from utils.data_misc import trunc_norm
from utils.infer_utils import get_pose
from utils.metric_utils import calculate_distances
from utils.occu import get_grid, query_bps_batched, query_occu, query_bps, get_bps, query_occu_batched, query_occu_batched_field, project, get_delta_v, query_bps_batched_field
from utils.train_utils import calc_jegor, get_foot_contact, get_from_outputs, get_in_vec, get_limb_from_outputs, get_ss_past_limb_traj, get_ss_past_traj, get_ss_past_vel, get_ss_tgt, pad_fcontact_norm, set_io_dims, update_fdir, update_pos, vec_to_ctrl
from utils.infer_utils import *

from utils.quaternion import *
from utils.utils import backup_code, backup_config_file, create_logger, get_config_path
from dataset import ELIMBS, LIMBS, MphaseDataset, collate_fn, normalize, denormalize
import smplx1 as smplx

@torch.no_grad()
def infer(data, dataset, bm, network, config, norm):
    input_mean, input_std, output_mean, output_std = norm
    if config.TRAIN.get('USE_FIELD', False):
        vel_mean = output_mean[264:267].to(device) # get mean of vel
        vel_std = output_std[264:267].to(device) # get std of vel
        vel_norm = (vel_mean, vel_std)
    network.eval()
    past_kf = config.TRAIN.PAST_KF
    future_kf = config.TRAIN.FUTURE_KF
    bs = len(data['j_6d'])
    j6d  = data['j_6d'].to(device, non_blocking=True)
    jego = data['j_ego'].to(device, non_blocking=True)
    jabs = data['j_abs'].to(device, non_blocking=True)
    traj  = data['traj'].to(device, non_blocking=True)
    fdir  = data['fdir'].to(device, non_blocking=True)

    USE_EVEL = config.TRAIN.USE_EVEL
    USE_NORM = config.TRAIN.USE_NORM
    USE_FCONTACT = config.TRAIN.get('USE_FCONTACT', False)
    USE_LIMB_TRAJ = config.TRAIN.USE_LIMB_TRAJ
    USE_TGT = config.TRAIN.USE_TARGET
    USE_VOX = config.TRAIN.USE_VOX
    USE_BPS = config.TRAIN.get('USE_BPS', False)
    USE_SMPL_UPDATE = config.TRAIN.get('USE_SMPL_UPDATE', False) and bm is not None
    if config.TRAIN.USE_VOX:
        if config.TRAIN.BATCH_VOX:
            if config.TRAIN.get('PRECREATE_GRID', False):
                vox_device = device if config.TRAIN.VOX_ON_GPU else 'cpu'
                grid_size = config.TRAIN.GRID_SIZE
                grid, oidx, occu_l = get_grid(unit=config.TRAIN.GRID_UNIT, size=grid_size, device=vox_device)
                bs = len(j6d)
                grid = grid[None].expand(bs, -1, -1).contiguous()
                oidx = oidx[None].expand(bs, -1, -1).contiguous()
                occu_l = occu_l[None].expand(bs, -1, -1, -1).contiguous()
            # vox_device = device if config.TRAIN.VOX_ON_GPU else 'cpu'
            occu_g = data['vox'].to(vox_device)
            llb    = data['llb'].to(vox_device)
            if USE_BPS: 
                occu_g_ref = data['ref'].to(vox_device)
                occu_shape = data['shape'].to(vox_device)
                bps = torch.from_numpy(np.load(config.ASSETS.BASIS_PATH)).float().to(vox_device)
                bps = get_bps(bps, config.TRAIN.GRID_UNIT, config.TRAIN.GRID_SIZE)
        else:
            mid = data['mid'].tolist()
            llb = data['llb'].to('cpu')
            occu_g = [dataset.occu_g_dict[i][0].to('cpu') for i in mid]
    if USE_TGT:
        tgt_limb_abs = data['jabs_tgt'][:, ELIMBS].to(device, non_blocking=True)
    else:
        tgt_limb_abs = None

    if USE_EVEL:
        crt_j6d, crt_jego, crt_jevel = get_g_joints_evel(j6d, jego, past_kf)
    else:
        crt_j6d, crt_jego, crt_jevel = get_g_joints(j6d, jego, jabs, fdir, past_kf, nxt_vel=config.TRAIN.USE_NXT_EVEL)
    crt_pos, crt_fdir = get_pos_dir(jabs, fdir, idx=past_kf)
    past_traj, past_vel = get_g_past_traj(traj, fdir, past_kf, crt_idx=past_kf)
    past_fdir, past_fdir_vel = get_g_past_fdir(fdir, past_kf, crt_idx=past_kf)
    if USE_LIMB_TRAJ:
        past_limb_traj, past_limb_vel = get_g_past_limb_traj(jabs, fdir, past_kf, crt_idx=past_kf)

    if config.INFER.get('SPECIFY_FIRST', False):
        first_pose = torch.load(config.INFER.FIRST_PATH)
        crt_j6d = first_pose[:22*6].view(1, 22, 6).expand(bs, -1, -1).to(device)
        crt_jego = first_pose[22*6:].view(1, 22, 3).expand(bs, -1, -1).to(device)
        height = crt_jego[:, 0, 2]
        limb_height = crt_jego[:, LIMBS, 2]
        crt_jevel = torch.zeros_like(crt_jevel)
        past_traj = torch.zeros_like(past_traj)
        past_traj[..., 2] = height[:, None]
        past_vel  = torch.zeros_like(past_vel)
        past_fdir = torch.zeros_like(past_fdir)
        past_fdir[..., 1] = 1.0
        past_fdir_vel = torch.zeros_like(past_fdir_vel)
        past_fdir_vel[..., 0] = 1.0
        if USE_LIMB_TRAJ:
            past_limb_traj = torch.zeros_like(past_limb_traj)
            past_limb_traj[..., 2] = limb_height[:, None]
            past_limb_vel = torch.zeros_like(past_limb_vel)
        crt_pos[:, 2] = height

    pose_seq = [get_pose(cont6d_to_aa(crt_j6d).flatten(1), crt_pos, fdir_to_rad(crt_fdir), bm).cpu()]
    if USE_TGT:
        rel_pos_seq = [get_abs_limbs(crt_pos, fdir_to_rad(crt_fdir), rel_limb_pos=get_ss_tgt(tgt_limb_abs, crt_pos, crt_fdir)).cpu()]
    if config.TRAIN.get('USE_FIELD', False):
        ori_v_lst = []
        dv_lst = []

    # Main loop.
    for i in range(config.INFER.INFER_LEN):
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
                    occu_l_lst, d_vecs = query_bps_batched_field(occu_g, llb, occu_g_ref, occu_shape, crt_pos, fdir_to_rad(crt_fdir), bps, unit=config.TRAIN.GRID_UNIT, device=vox_device)
                else:
                    occu_l_lst = query_bps_batched(occu_g, llb, occu_g_ref, occu_shape, crt_pos, fdir_to_rad(crt_fdir), bps, unit=config.TRAIN.GRID_UNIT, device=vox_device)
            else:
                if config.TRAIN.get('USE_FIELD', False):
                    query_result = query_occu_batched_field(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir), 
                        unit=config.TRAIN.GRID_UNIT, grid_size=config.TRAIN.GRID_SIZE, device=vox_device, grid=grid, oidx=oidx, occu_l=occu_l,
                        jego=crt_jego, config=config, tgt_limb_abs=tgt_limb_abs)
                    occu_l_lst, d_vecs = query_result['occul'], query_result['d_vecs']
                    if 'close' in query_result and config.TRAIN.get('CLOSE_LABEL', False):
                        close = query_result['close']
                else:
                    occu_l_lst = query_occu_batched(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir), config.TRAIN.GRID_UNIT, None, vox_device, grid, oidx, occu_l)
                # v = crt_jevel[:, 0] if i == 0 else nxt_traj # [bs, 3]
                # delta_v = get_delta_v(d_vecs, v) # [bs, 3] before multiplied by alpha
                # vel_std = output_std[264:266].to(device) # get std of vel
                # delta_v[:, :2] = delta_v[:, :2] / vel_std
            occu_l_lst = occu_l_lst.flatten(1).to(device)
            # assert torch.isnan(occu_l_lst).sum() == 0
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
                pred_vec, ori_v, dv = pred_dict['pred'], pred_dict['vel'], pred_dict['dv']
                ori_v = qrot(fdir_to_quat(crt_fdir), ori_v)
                dv = qrot(fdir_to_quat(crt_fdir), dv)
                ori_v[..., 2] = 0.0
                dv[..., 2] = 0.0
                ori_v_lst.append(ori_v.cpu())
                dv_lst.append(dv.cpu())
            else:
                pred_vec = network(ctrl_dict)
        if USE_NORM:
            pred_vec = denormalize(pred_vec, output_mean, output_std)
        nxt_j6d, nxt_jego, nxt_jevel, nxt_traj, vel, fdir_vel = get_from_outputs(pred_vec, future_kf)
        if USE_LIMB_TRAJ:
            nxt_limbs, nxt_limb_vel = get_limb_from_outputs(pred_vec, future_kf)

        # Update.
        # Update current info. (nxt -> crt)
        nxt_pos = update_pos(crt_pos, crt_fdir, nxt_traj)
        nxt_fdir = update_fdir(crt_fdir, fdir_vel)
        abs_pose = get_pose(cont6d_to_aa(nxt_j6d).flatten(1), nxt_pos, fdir_to_rad(nxt_fdir), bm)
        if USE_SMPL_UPDATE:
            nxt_jego = get_pose(cont6d_to_aa(nxt_j6d).flatten(1), bm=bm)
        pose_seq.append(abs_pose.clone().cpu())
        if USE_TGT:
            rel_pos_seq.append(get_abs_limbs(crt_pos, fdir_to_rad(crt_fdir), tgt_limbs).cpu())
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
            
    pose_seq = torch.stack(pose_seq, dim=1) # [bs, frames, 22, 3]
    starts = pose_seq[:, 0]
    rel_pos_seq = torch.stack(rel_pos_seq, dim=1) # [bs, frames, 5, 3]
    if config.INFER.get('ANI_EARLY_STOP', False):
        DIST_THRES = config.INFER.get('DIST_THRES', 4.0)
        INC_LASTS = config.INFER.get('INC_LASTS', 2)
        dists, closest_ids = calculate_distances(pose_seq[:, :, ELIMBS], tgt_limb_abs.cpu(), INC_LASTS, DIST_THRES)
        poses = []
        rel_limbs = []
        for i in range(len(pose_seq)):
            poses.append(pose_seq[i, :closest_ids[i]+1])
            rel_limbs.append(rel_pos_seq[i, :closest_ids[i]+1])
        pose_seq = poses
        rel_limbs = rel_pos_seq

    draw_dict = {'seq': pose_seq, 'start': starts} # [bs, 22, 3]
    draw_dict['mid_snip'] = data['mid_snip']
    if USE_TGT:
        draw_dict['tgt_seq'] = rel_pos_seq
        draw_dict['end'] = data['jabs_tgt'] # [bs, 22, 3]
    if USE_VOX:
        draw_dict['occug'] = [dataset.occu_g_dict[i][0] for i in data['mid_snip'][:, 0].tolist()]
        draw_dict['llb'] = llb
        draw_dict['unit'] = config.TRAIN.GRID_UNIT
    if config.TRAIN.get('USE_FIELD', False):
        ori_v_lst.append(torch.zeros_like(ori_v_lst[0]))
        dv_lst.append(torch.zeros_like(dv_lst[0]))
        ori_v_lst = torch.stack(ori_v_lst, dim=1)
        dv_lst = torch.stack(dv_lst, dim=1)
        draw_dict['ori_v'] = ori_v_lst
        draw_dict['dv'] = dv_lst
    return draw_dict

def draw_batch(draw_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(draw_dict['seq'])):
        single_draw_dict = {
            'seq': draw_dict['seq'][i],
            'start': draw_dict['start'][i],
            'end': draw_dict['end'][i],
            'tgt_seq': draw_dict['tgt_seq'][i],
            'occug': draw_dict['occug'][i],
            'llb': draw_dict['llb'][i],
            'unit': draw_dict['unit'],
        }
        if 'ori_v' in draw_dict:
            single_draw_dict['ori_v'] = draw_dict['ori_v'][i]
            single_draw_dict['dv'] = draw_dict['dv'][i]
        mid, st_fid, end_fid, idx = draw_dict['mid_snip'][i]
        sinp_len = len(single_draw_dict['seq'])
        save_name = f'vox_len{sinp_len}_no{idx}_{mid}_{st_fid}_{end_fid}.mp4'
        draw_seq(os.path.join(save_dir, save_name), single_draw_dict)

if __name__ == '__main__':
    config_path = get_config_path()
    config = OmegaConf.load(config_path)
    config.STAGE = 'INFER'
    if config.INFER.get('IGNORE_WARNINGS', False):
        import warnings
        warnings.filterwarnings("ignore")
    if type(config.TRAIN.GRID_SIZE) is int:
        config.TRAIN.GRID_SIZE = [config.TRAIN.GRID_SIZE] * 3
    config.DEVICE_STR = f"cuda:{config.DEVICE}" if torch.cuda.is_available() else "cpu"
    device = torch.device(config.DEVICE_STR)
    logger = create_logger(config, to_file=False)
    logger.info(f'Animation save dir: {config.INFER.ANI_SAVE_DIR}')
    # backup_config_file(config, config_path)
    # backup_code(config, config_path)
    PAST_KF = config.TRAIN.PAST_KF
    FUTURE_KF = config.TRAIN.FUTURE_KF
    IN_DIM, OUT_DIM = set_io_dims(config)

    logger.info("Initializing network...")
    if config.TRAIN.USE_NORM:
        mean_std = torch.load(os.path.join(config.ASSETS.SPLIT_DIR, config.ASSETS.MEAN_STD_NAME))
        input_mean, input_std, output_mean, output_std = [v.to(device) for k, v in list(mean_std.items())]
        if config.TRAIN.get('USE_FCONTACT', False):
            input_mean, input_std = pad_fcontact_norm(input_mean, input_std)
        input_mean, input_std, output_mean, output_std = trunc_norm(config, input_mean, input_std, output_mean, output_std)

    network = get_model(config)
    load_res = network.load_state_dict(torch.load(config.ASSETS.CHECKPOINT, map_location=torch.device('cpu')), strict=False)
    logger.info(f'Loaded checkpoint: {load_res}')
    network.to(device)
    network.eval()
    torch.set_grad_enabled(False)

    dataset = MphaseDataset(config, config.INFER.SPLIT)
    if config.INFER.get('SPECIFY_CASES', False):
        case_config = OmegaConf.load(config.INFER.CASE_CONFIG)
        split_dirname = os.path.basename(config.ASSETS.SPLIT_DIR)
        data_ids = eval(f'case_config.CASES.{split_dirname}')
    else:
        interval = config.INFER.get('SAVE_ID_INTERVAL', 500)
        data_ids = [i for i in range(0, len(dataset), interval)]
    data = collate_fn([dataset[i] for i in data_ids])
    logger.info(f'Loaded {len(data_ids)} snippets to be processed.')
    logger.info(f'The snippet ids are: {data_ids}')

    bm = smplx.create(config.ASSETS.SMPL_DIR, model_type='smpl', gender='male', num_betas=16).to(device)

    norm = (input_mean, input_std, output_mean, output_std)
    draw_dict = infer(data, dataset, bm, network, config, norm)
    ANI_SAVE_DIR = config.INFER.ANI_SAVE_DIR
    draw_batch(draw_dict, ANI_SAVE_DIR)

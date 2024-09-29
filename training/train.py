import argparse
import random
import shutil
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from torch import optim
import os
from models import get_model
from utils.occu import get_grid, query_occu, query_occu_batched, query_bps_batched, get_bps, query_occu_batched_field, project, get_delta_v, query_bps_batched_field
from utils.quaternion import fdir_to_rad
import smplx1 as smplx
from utils.smpl import smpl_forward
from utils.train_utils import *
from utils.infer_utils import get_pose
from utils.data_misc import change_system, save_norm, trunc_norm
from utils.utils import backup_code, backup_config_file, create_logger, get_config_path, get_save_schedule, set_seed
from dataset import ELIMBS, LIMBS, MphaseDataset, denormalize, normalize, collate_fn

def train_one_epoch(config, network, optimizer, scheduler, dataloader, epoch_num, bm=None, **kwargs):
    if config.TRAIN.USE_NORM:
        input_mean, input_std, output_mean, output_std = kwargs.get('norm', (None, None, None, None))
        if config.TRAIN.get('USE_FIELD', False):
            vel_mean = output_mean[264:267].to(device) # get mean of vel
            vel_std = output_std[264:267].to(device) # get std of vel
            vel_norm = (vel_mean, vel_std)
    ROLLOUT_FRAMES = config.TRAIN.ROLLOUT_FRAMES
    USE_FCONTACT = config.TRAIN.get('USE_FCONTACT', False) and not config.TRAIN.CALC_NORM
    USE_LIMB_TRAJ = config.TRAIN.USE_LIMB_TRAJ
    USE_TGT = config.TRAIN.USE_TARGET
    USE_VOX = config.TRAIN.USE_VOX
    USE_BPS = config.TRAIN.get('USE_BPS', False)
    USE_NORM = config.TRAIN.USE_NORM
    USE_EVEL = config.TRAIN.USE_EVEL
    USE_SMPL_UPDATE = config.TRAIN.get('USE_SMPL_UPDATE', False) and bm is not None
    USE_GRID_LOSS   = config.TRAIN.get('USE_GRID_LOSS', False) and bm is not None
    USE_FLOSS = config.TRAIN.get('USE_FIELD', False) and config.TRAIN.get('USE_FLOSS', False)
    USE_VLOSS = config.TRAIN.get('USE_FIELD', False) and config.TRAIN.get('USE_VLOSS', False)
    USE_SS = config.TRAIN.get('USE_SS', True) # TODO rename
    USE_HAND = config.TRAIN.get('USE_HAND', False)
    past_kf = config.TRAIN.PAST_KF
    future_kf = config.TRAIN.FUTURE_KF
    network.train()
    total_loss_lst = []
    l1_loss_lst = []
    l2_loss_lst = []
    if USE_FLOSS:
        floss_lst = []
    if USE_VLOSS:
        vloss_lst = []
    if USE_GRID_LOSS:
        grid_loss_lst = []
    it_len = len(dataloader)
    bs = config.TRAIN.BATCH_SIZE
    CALC_NORM = config.TRAIN.CALC_NORM
    if USE_BPS:
        bps = torch.from_numpy(np.load(config.ASSETS.BASIS_PATH)).float().to(device)
        bps = get_bps(bps, config.TRAIN.GRID_UNIT, config.TRAIN.GRID_SIZE)
    if CALC_NORM:
        in_lst = []
        gt_lst = []
    if config.TRAIN.BATCH_VOX:
        vox_device = device if config.TRAIN.VOX_ON_GPU else 'cpu'
        if config.TRAIN.get('PRECREATE_GRID', False):
            grid_size = config.TRAIN.GRID_SIZE
            grid, oidx, occu_l = get_grid(unit=config.TRAIN.GRID_UNIT, size=grid_size, device=vox_device)
            grid = grid[None].expand(bs, -1, -1).contiguous()
            oidx = oidx[None].expand(bs, -1, -1).contiguous()
            occu_l = occu_l[None].expand(bs, -1, -1, -1).contiguous()

    for it, data in enumerate(dataloader):
        j_6d  = data['j_6d'].to(device, non_blocking=True)   # xp: SMPL-X parameters, in rot6d
        j_ego = data['j_ego'].to(device, non_blocking=True)  # xp: joint location in egocentric coordinate
        j_abs = data['j_abs'].to(device, non_blocking=True)  # xp: joint location in global coordinate
        traj  = data['traj'].to(device, non_blocking=True)   # xp: historical root trajectory -- global? ego?
        fdir  = data['fdir'].to(device, non_blocking=True)   # xp: historical forward direction 
        if USE_HAND:
            hand_ego = data['hand_ego'].to(device, non_blocking=True)
            hand_abs = data['hand_abs'].to(device, non_blocking=True)
        if config.TRAIN.USE_TARGET:
            tgt_limb_abs = data['jabs_tgt'][:, ELIMBS].to(device, non_blocking=True) # xp: target joint location in global coordinate
        else:
            tgt_limb_abs = None
            # tgt_fdir = data['tgt_fdir'].to(device, non_blocking=True)
        if config.TRAIN.USE_VOX:
            if config.TRAIN.BATCH_VOX:
                # vox_device = device if config.TRAIN.VOX_ON_GPU else 'cpu'
                occu_g = data['vox'].to(vox_device)  # xp: global occupancy in global coordinate
                llb    = data['llb'].to(vox_device)  # xp: global occupancy offset
                if USE_BPS or USE_GRID_LOSS: 
                    occu_g_ref = data['ref'].to(vox_device)   # xp: global closest voxel reference
                    occu_shape = data['shape'].to(vox_device) # xp: global occupancy volume shape
            else:
                mid = data['mid'].tolist()
                llb = data['llb'].to('cpu')
                occu_g = [train_dataset.occu_g_dict[i][0].to('cpu') for i in mid]

        r_ids = torch.rand(bs) < sample_schedule[epoch_num] if not config.TRAIN.DEBUG else torch.tensor([True, False])
        for r in range(ROLLOUT_FRAMES):
            optimizer.zero_grad()

            with torch.no_grad():
                # Generate inputs.
                crt_pos, crt_fdir = get_pos_dir(j_abs, fdir, past_kf, r) # root location and forward for current frame in global coordinate
                if USE_EVEL:
                    in_j6d, in_jego, in_jevel = get_g_joints_evel(j_6d, j_ego, r, past_kf) 
                else:
                    in_j6d, in_jego, in_jevel = get_g_joints(j_6d, j_ego, j_abs, fdir, r, past_kf, nxt_vel=config.TRAIN.USE_NXT_EVEL) # joint rotation, location, and velocity for current frame in egocentric coordinate
                past_traj, past_vel = get_g_past_traj(traj, fdir, r, past_kf) # historical root trajectory and velocity
                past_fdir, past_fdir_vel = get_g_past_fdir(fdir, r, past_kf)  # historical forward direction and angular velocity in egocentric coordinate
                if USE_LIMB_TRAJ:
                    crt_limbs = in_jego[:, LIMBS].clone()
                    past_limb_traj, past_limb_vel = get_g_past_limb_traj(j_abs, fdir, r, past_kf) # historical joint trajectory and velocity in egocentric coordinate
                if r > 0 and not CALC_NORM and USE_SS: # Sched Samp
                    in_j6d[r_ids]        = crt_ss_j6d[r_ids]
                    in_jego[r_ids]       = crt_ss_jego[r_ids]
                    in_jevel[r_ids]      = crt_ss_jevel[r_ids]
                    past_traj[r_ids]     = ss_past_traj[r_ids]
                    past_vel[r_ids]      = ss_past_vel[r_ids]
                    past_fdir[r_ids]     = ss_past_fdir[r_ids]
                    past_fdir_vel[r_ids] = ss_past_fdir_vel[r_ids]
                    if USE_LIMB_TRAJ:
                        crt_limbs[r_ids]      = crt_ss_limbs[r_ids]
                        past_limb_traj[r_ids] = ss_past_limb_traj[r_ids]
                        past_limb_vel[r_ids]  = ss_past_limb_vel[r_ids]
                    crt_pos[r_ids]  = crt_ss_pos[r_ids]
                    crt_fdir[r_ids] = crt_ss_fdir[r_ids]
                if USE_TGT:
                    tgt_limbs = get_ss_tgt(tgt_limb_abs, crt_pos, crt_fdir) # target joint location in egocentric coordinate
                if USE_VOX:
                    if USE_BPS:
                        if config.TRAIN.get('USE_FIELD', False):
                            occu_l_lst, d_vecs = query_bps_batched_field(occu_g, llb, occu_g_ref, occu_shape, crt_pos, fdir_to_rad(crt_fdir), bps, unit=config.TRAIN.GRID_UNIT, device=vox_device)
                        else:
                            occu_l_lst = query_bps_batched(occu_g, llb, occu_g_ref, occu_shape, crt_pos, fdir_to_rad(crt_fdir), bps, unit=config.TRAIN.GRID_UNIT, device=vox_device)
                    else:
                        if config.TRAIN.get('USE_FIELD', False):
                            if config.TRAIN.get('DROP_VOX', False):
                                query_result = query_occu_batched_field(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir),
                                    unit=config.TRAIN.GRID_UNIT, grid_size=config.TRAIN.GRID_SIZE, device=vox_device, grid=grid, oidx=oidx, occu_l=occu_l,
                                    jego=in_jego, config=config, tgt_limb_abs=tgt_limb_abs)
                                occu_l_lst, d_vecs = query_result['occul'], query_result['d_vecs']
                                if 'close' in query_result and config.TRAIN.get('CLOSE_LABEL', False):
                                    close = query_result['close']
                            else:
                                occu_l_lst, d_vecs = query_occu_batched_field(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir),
                                                    unit=config.TRAIN.GRID_UNIT, grid_size=config.TRAIN.GRID_SIZE, device=vox_device, grid=grid, oidx=oidx, occu_l=occu_l)
                            # v = in_jevel[:, 0] # [bs, 3]
                            # if r > 0:
                            #     v[r_ids] = nxt_traj[r_ids]
                            # delta_v = get_delta_v(d_vecs, v) # [bs, 3] before multiplied by alpha
                            # delta_v[:, :2] = delta_v[:, :2] / vel_std
                        else:
                            occu_l_lst = query_occu_batched(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir), config.TRAIN.GRID_UNIT, None, vox_device, grid, oidx, occu_l)
                        occu_l_lst = occu_l_lst.flatten(1).to(device)
                        # assert torch.isnan(occu_l_lst).sum() == 0
                if USE_HAND:
                    hand_future_kf = config.TRAIN.HAND_FUTURE_KF
                    in_hand = hand_abs[:, r:r+hand_future_kf+1] # [bs, past_kf+1, 4, 3]
                    crt_quat = fdir_to_quat(crt_fdir)
                    in_hand = change_system(in_hand, crt_pos, crt_quat, keep_h=True)
                in_dict = {'j6d': in_j6d, 'jego': in_jego, 'jevel': in_jevel}
                in_dict.update({'crt_pos': crt_pos})
                in_dict.update({'past_traj': past_traj, 'past_vel': past_vel, 'past_fdir': past_fdir, 'past_fdir_vel': past_fdir_vel})
                if USE_FCONTACT:
                    if config.TRAIN.get('LINEAR_FCONTACT', False):
                        fcontact = get_foot_contact_lin(in_jego)
                    else:
                        fcontact = get_foot_contact(in_jego)
                    in_dict['fcontact'] = fcontact
                if USE_TGT:
                    in_dict.update({'tgt_limbs': tgt_limbs})
                    if 'close' in locals():
                        in_dict['close'] = close
                if USE_LIMB_TRAJ:
                    in_dict.update({'past_limb_traj': past_limb_traj, 'past_limb_vel': past_limb_vel})
                if USE_VOX:
                    in_dict.update({'occu_l': occu_l_lst})
                if USE_HAND:
                    in_dict.update({'hand': in_hand})
                in_vec = get_in_vec(in_dict)

                # Generate gt.
                if USE_EVEL:
                    gt_j6d, gt_jego, gt_jevel = get_gt_joints_evel(j_6d, j_ego, r, past_kf)
                else:
                    gt_j6d, gt_jego, gt_jevel = get_gt_joints(j_6d, j_ego, j_abs, fdir, r, past_kf, nxt_vel=config.TRAIN.USE_NXT_EVEL)
                gt_fut_traj, gt_fut_vel = get_gt_fut_traj(traj, fdir, r, past_kf, future_kf)
                gt_fut_fdir, gt_fut_fdir_vel = get_gt_fut_fdir(fdir, r, past_kf, future_kf)
                if USE_LIMB_TRAJ:
                    gt_fut_limb_traj, gt_fut_limb_vel = get_gt_fut_limb_traj(j_abs, fdir, r, past_kf, future_kf)
                if r > 0 and not CALC_NORM and USE_SS:
                    if config.TRAIN.SS_TRAJ:
                        ssgt_fut_traj, ssgt_fut_vel = get_ssgt_fut_traj(traj, crt_pos, crt_fdir, past_kf, future_kf, r)
                        ssgt_fut_fdir, ssgt_fut_fdir_vel = get_ssgt_fut_fdir(fdir, crt_fdir, past_kf, future_kf, r)
                        gt_fut_traj[r_ids], gt_fut_vel[r_ids] = ssgt_fut_traj[r_ids], ssgt_fut_vel[r_ids]
                        gt_fut_fdir[r_ids], gt_fut_fdir_vel[r_ids] = ssgt_fut_fdir[r_ids], ssgt_fut_fdir_vel[r_ids]
                    if USE_LIMB_TRAJ and config.TRAIN.SS_LIMB_TRAJ:
                        ssgt_fut_limb_traj, ssgt_fut_limb_vel = get_ssgt_fut_limb_traj(crt_limbs, j_abs, crt_pos, crt_fdir, past_kf, future_kf, r)
                        gt_fut_limb_traj[r_ids], gt_fut_limb_vel[r_ids] = ssgt_fut_limb_traj[r_ids], ssgt_fut_limb_vel[r_ids]
                gt_dict = {'j6d': gt_j6d, 'jego': gt_jego, 'jevel': gt_jevel}
                gt_dict.update({'fut_traj': gt_fut_traj, 'fut_vel': gt_fut_vel, 'fut_fdir': gt_fut_fdir, 'fut_fdir_vel': gt_fut_fdir_vel})
                if USE_LIMB_TRAJ:
                    gt_dict.update({'fut_limb_traj': gt_fut_limb_traj, 'fut_limb_vel': gt_fut_limb_vel})
                gt_vec = get_gt_vec(gt_dict)
                if CALC_NORM:
                    in_lst.append(in_vec.cpu())
                    gt_lst.append(gt_vec.cpu())
                    continue

            # Infer.
            if config.TRAIN.DEBUG:
                pred_vec = gt_vec.clone()
                # nxt_j6d, nxt_jego, nxt_jevel, nxt_traj, vel, fdir_vel = output_from_gt(j_6d, j_ego, gt_fut_traj, gt_fut_vel, gt_fut_fdir_vel, r, past_kf)
            else:
                if USE_NORM:
                    ndims = input_mean.shape[0]
                    in_vec[:, :ndims] = normalize(in_vec[:, :ndims], input_mean, input_std)
                    gt_vec = normalize(gt_vec, output_mean, output_std)
                if config.TRAIN.SEP_CTRLS:
                    ctrl_dict = vec_to_ctrl(config, in_vec)
                    if config.TRAIN.get('USE_FIELD', False):
                        vel = past_vel.view(-1, 3)
                        pred_dict = network(ctrl_dict, d_vecs, vel_norm, vel)
                        pred_vec = pred_dict['pred']
                        if USE_FLOSS:
                            floss = pred_dict['floss']
                        if USE_VLOSS:
                            vloss = pred_dict['vloss']
                    else:
                        pred_vec = network(ctrl_dict)
                else:
                    pred_vec = network(in_vec)
                l1_loss, l2_loss = calc_loss(pred_vec, gt_vec, future_kf, 
                                                   l1_ratio=config.TRAIN.L1_RATIO, l2_ratio=config.TRAIN.L2_RATIO)
                loss = l1_loss + l2_loss
                if USE_FLOSS:
                    floss = config.TRAIN.FLOSS_RATIO * floss
                    loss += floss
                if USE_VLOSS:
                    vloss = config.TRAIN.VLOSS_RATIO * vloss
                    loss += vloss
                if USE_NORM:
                    pred_vec = denormalize(pred_vec, output_mean, output_std)

            # Update current info. (nxt -> crt, crt -> past)
            with torch.no_grad():
                nxt_j6d, nxt_jego, nxt_jevel, nxt_traj, vel, fdir_vel = get_from_outputs(pred_vec, future_kf)
                nxt_pos  = update_pos(crt_pos, crt_fdir, nxt_traj)
                nxt_fdir = update_fdir(crt_fdir, fdir_vel)
                assert (nxt_fdir[..., 2] == 0).all() and (crt_fdir[..., 2] == 0).all()
                if USE_LIMB_TRAJ:
                    nxt_limbs, nxt_limb_vel = get_limb_from_outputs(pred_vec, future_kf)
                if USE_SMPL_UPDATE:
                    nxt_jego = get_pose(cont6d_to_aa(nxt_j6d).flatten(1), bm=bm)
                if USE_GRID_LOSS:
                    abs_pose = get_pose(cont6d_to_aa(nxt_j6d).flatten(1), nxt_pos, fdir_to_rad(nxt_fdir), bm) # in global coordinate, B,J,3
                    grid_loss = calc_grid_loss(abs_pose, occu_g, llb, occu_g_ref, occu_shape, unit=config.TRAIN.GRID_UNIT, device=device)
                    grid_loss = grid_loss * config.TRAIN.get('LG_RATIO', 1.0)
                    loss += grid_loss
                if r != ROLLOUT_FRAMES - 1: # SS
                    ss_past_traj, ss_past_vel = get_ss_past_traj(past_traj, past_vel, crt_pos[:, 2:3], vel, nxt_traj, fdir_vel)
                    ss_past_fdir, ss_past_fdir_vel = get_ss_past_vel(past_fdir, past_fdir_vel, fdir_vel)
                    if USE_LIMB_TRAJ:
                        ss_past_limb_traj, ss_past_limb_vel = \
                            get_ss_past_limb_traj(past_limb_traj, nxt_traj, nxt_limbs, past_limb_vel, nxt_limb_vel, fdir_vel)
                        crt_ss_limbs = ss_past_limb_traj[:, -1].clone()
                    crt_ss_j6d   = nxt_j6d
                    crt_ss_jego  = nxt_jego
                    crt_ss_jevel = nxt_jevel
                    crt_ss_pos   = nxt_pos
                    crt_ss_fdir  = nxt_fdir

            if config.TRAIN.DEBUG:
                continue
            l1_loss_lst.append(l1_loss.item())
            l2_loss_lst.append(l2_loss.item())
            if USE_FLOSS:
                floss_lst.append(floss.item())
            if USE_VLOSS:
                vloss_lst.append(vloss.item())
            if USE_GRID_LOSS:
                grid_loss_lst.append(grid_loss.item())
            total_loss_lst.append(loss.item())
            loss.backward()
            optimizer.step()
        if config.TRAIN.DEBUG:
            continue
        if config.TRAIN.SHOW_ITER and (it+1) % (config.TRAIN.LOG_FREQ) == 0 and it != 0:
            if CALC_NORM:
                logger.info(f"Epoch: {epoch_num+1}, Iteration: {it+1}/{min(int(config.TRAIN.CALC_NORM_RATIO * it_len), config.TRAIN.CALC_NORM_ITER):}, Total length: {it_len}")
            else:
                info = f"Epoch: {epoch_num+1}, Iteration: {it+1}/{it_len}, Average loss: {sum(total_loss_lst) / len(total_loss_lst):.4f}, " \
                       f"L1: {sum(l1_loss_lst) / len(l1_loss_lst):.4f}, L2: {sum(l2_loss_lst) / len(l2_loss_lst):.4f}"
                if USE_FLOSS:
                    info += f", Floss: {sum(floss_lst) / len(floss_lst):.4f}"
                if USE_VLOSS:
                    info += f", Vloss: {sum(vloss_lst) / len(vloss_lst):.4f}"
                if USE_GRID_LOSS:
                    info += f", Grid loss: {sum(grid_loss_lst) / len(grid_loss_lst):.4f}"
                logger.info(info)
        if CALC_NORM:
            if it + 1 == min(int(config.TRAIN.CALC_NORM_RATIO * it_len), config.TRAIN.CALC_NORM_ITER):
                break
        if not CALC_NORM and it + 1 == config.TRAIN.ITER_PER_EP:
            break

    if CALC_NORM:
        ins = torch.cat(in_lst, dim=0)
        gts = torch.cat(gt_lst, dim=0)
        input_mean, input_std = ins.mean(dim=0), ins.std(dim=0)
        output_mean, output_std = gts.mean(dim=0), gts.std(dim=0)
        return input_mean.cpu(), input_std.cpu(), output_mean.cpu(), output_std.cpu()
    scheduler.step()
    avg_loss = {
        'l1': sum(l1_loss_lst) / len(l1_loss_lst),
        'l2': sum(l2_loss_lst) / len(l2_loss_lst),
        'total': sum(total_loss_lst) / len(total_loss_lst),
    }
    if USE_FLOSS:
        avg_loss['floss'] = sum(floss_lst) / len(floss_lst)
    if USE_VLOSS:
        avg_loss['vloss'] = sum(vloss_lst) / len(vloss_lst)
    if USE_GRID_LOSS:
        avg_loss['grid_loss'] = sum(grid_loss_lst) / len(grid_loss_lst)
    return avg_loss

def train(config, network, optimizer, scheduler, train_loader, bm=None, start_ep=0, **kwargs):
    save_schedule = get_save_schedule(config.TRAIN.RESTART_PERIOD, config.TRAIN.RESTART_MULT)
    logger.info(f"Length of train loader: {len(train_loader)}")
    for epoch in range(start_ep, config.TRAIN.EPOCHS):
        if config.TRAIN.CALC_NORM:
            with torch.no_grad():
                input_mean, input_std, output_mean, output_std = train_one_epoch(config, network, optimizer, scheduler, train_loader, epoch, bm)
                save_norm(config, input_mean, input_std, output_mean, output_std)
            return
        norm = kwargs.get('norm', (None, None, None, None))
        avg_loss = train_one_epoch(config, network, optimizer, scheduler, train_loader, epoch, bm, norm=norm)
        info = f"Epoch: {epoch+1}, Average loss: {avg_loss['total']:.4f}, " \
               f"L1: {avg_loss['l1']:.4f}, L2: {avg_loss['l2']:.4f}"
        if config.TRAIN.get('USE_FIELD', False) and config.TRAIN.get('USE_FLOSS', False):
            info += f", floss: {avg_loss['floss']:.4f}"
        if config.TRAIN.get('USE_FIELD', False) and config.TRAIN.get('USE_VLOSS', False):
            info += f", vloss: {avg_loss['vloss']:.4f}"
        if config.TRAIN.get('USE_FIELD', False) and config.TRAIN.get('USE_GRID_LOSS', False):
            info += f", grid_loss: {avg_loss['grid_loss']:.4f}"
        # if config.TRAIN.get('USE_FIELD', False):
        #     info += f", alpha: {avg_loss['alpha']:.8f}"
        logger.info(info)

        if config.TRAIN.get('SAVE_TRAIN_CKPT', True):
            checkpoint = {
                'epoch': epoch, # from 0
                'model_state': network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
                'loss': avg_loss
            }
            # network.to(device=config.DEVICE_STR)
            if config.DEVICE_STR.startswith('cuda'):
                checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state(device=config.DEVICE_STR)
            # if config.TRAIN.NUM_WORKERS > 0:
            #     checkpoint['worker_rng_states'] = torch.utils.data.get_worker_info().seed
            ckpt_save_dir = os.path.join(config.ASSETS.RESULT_DIR, config.RUN_NAME)
            ckpt_save_path = os.path.join(ckpt_save_dir, 'checkpoint.pth')
            if os.path.exists(ckpt_save_path):
                shutil.copyfile(ckpt_save_path, os.path.join(ckpt_save_dir, 'checkpoint_prev.pth'))
            torch.save(checkpoint, ckpt_save_path)
        
        loss = avg_loss['total']
        if epoch + 1 in save_schedule:
            logger.info("Saving checkpoint...")
            torch.save(network.state_dict(), 
                    os.path.join(config.ASSETS.RESULT_DIR, config.RUN_NAME, f'epoch_{epoch+1}.pt'))
            logger.info(f"Checkpoint saved at epoch {epoch+1} with loss {loss:.4f}")

def create_optimizer(config, network):
    optimizer = optim.Adam(
        network.parameters(), lr=config.TRAIN.LEARNING_RATE, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.TRAIN.RESTART_PERIOD, T_mult=config.TRAIN.RESTART_MULT, eta_min=config.TRAIN.LEARNING_RATE_MIN)
    return optimizer, scheduler

if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # torch.autograd.set_detect_anomaly(True)
    config_path = get_config_path()
    config = OmegaConf.load(config_path)
    config.STAGE = 'TRAIN'
    if config.TRAIN.DEBUG:
        config.TRAIN.BATCH_SIZE = 2
    if config.TRAIN.CALC_NORM:
        config.TRAIN.USE_NORM = False
        config.TRAIN.USE_VOX = False
        config.TRAIN.USE_FIELD = False
        config.TRAIN.DROP_TGT = 0.0
        config.TRAIN.SHOW_ITER = True
        config.TRAIN.LOG_FREQ = 500
    if config.TRAIN.IGNORE_WARNINGS:
        import warnings
        warnings.filterwarnings("ignore")
    if type(config.TRAIN.GRID_SIZE) is int:
        config.TRAIN.GRID_SIZE = [config.TRAIN.GRID_SIZE] * 3
    
    config.DEVICE_STR = f"cuda:{config.DEVICE}" if torch.cuda.is_available() else "cpu"
    # config.DEVICE_STR = 'cpu'
    device = torch.device(config.DEVICE_STR)
    
    set_seed(config.TRAIN.SEED)
    PAST_KF = config.TRAIN.PAST_KF
    FUTURE_KF = config.TRAIN.FUTURE_KF
    IN_DIM, OUT_DIM = set_io_dims(config)

    logger = create_logger(config)
    backup_config_file(config, config_path)
    backup_code(config, config_path)

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Train name: {config.TRAIN.TRAIN_NAME}")
    if config.DESCRIPTION != "PH":
        logger.info(f"Description: {config.DESCRIPTION}")
    logger.info(f"Using device: {config.DEVICE_STR}")
    logger.info("Initializing dataset...")
    if config.TRAIN.USE_NORM:
        try:
            mean_std = torch.load(os.path.join(config.ASSETS.SPLIT_DIR1, config.ASSETS.MEAN_STD_NAME))
        except:
            mean_std = torch.load(os.path.join(config.ASSETS.SPLIT_DIR, config.ASSETS.MEAN_STD_NAME))
        input_mean, input_std, output_mean, output_std = [v.to(device) for k, v in list(mean_std.items())]
        if config.TRAIN.get('USE_FCONTACT', False):
            input_mean, input_std = pad_fcontact_norm(input_mean, input_std)
        input_mean, input_std, output_mean, output_std = trunc_norm(config, input_mean, input_std, output_mean, output_std)

    train_dataset = MphaseDataset(config, split='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True if not config.TRAIN.DEBUG else False,
        drop_last=True,
        num_workers=0 if config.TRAIN.TO_GPU_FIRST or config.TRAIN.DEBUG else config.TRAIN.NUM_WORKERS,
        persistent_workers=False if config.TRAIN.TO_GPU_FIRST or config.TRAIN.DEBUG else True,
        # pin_memory=False if config.TRAIN.TO_GPU_FIRST else True,
        prefetch_factor=None if config.TRAIN.TO_GPU_FIRST else config.TRAIN.PREFETCH,
        collate_fn=collate_fn,
    )
    # train_dataset.__getitem__(1000)

    # test_dataset = MphaseDataset(config, split='test', mean_std=mean_std, normalize_data=True)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=config.TEST.BATCH_SIZE,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=0 if config.TRAIN.TO_GPU_FIRST else config.TEST.NUM_WORKERS,
    #     pin_memory=False if config.TRAIN.TO_GPU_FIRST else True,
    #     persistent_workers=False if config.TRAIN.TO_GPU_FIRST else True
    # )

    logger.info("Initializing network...")
    
    network = get_model(config)
    network.to(device)

    MODEL_DIR=config.ASSETS.SMPL_DIR
    model_type = 'smplx' if config.TRAIN.get('USE_HAND', False) else 'smpl'
    bm = smplx.create(MODEL_DIR, model_type=model_type, gender='male', num_betas=16).to(device)

    logger.info("Initializing optimizer...")
    optimizer, scheduler = create_optimizer(config, network)

    sample_schedule = torch.cat([
        torch.zeros(config.TRAIN.TEACHER_EPOCHS),
        torch.linspace(0.0, config.TRAIN.STUDENT_RATIO, config.TRAIN.RAMPING_EPOCHS),
        torch.ones(config.TRAIN.STUDENT_EPOCHS),
    ])

    if config.TRAIN.get('RESUME', False):
        checkpoint = torch.load(config.TRAIN.RESUME_CKPT, map_location='cpu')
        network.load_state_dict(checkpoint['model_state'])
        # for state in checkpoint['optimizer_state']['state'].values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        torch.set_rng_state(checkpoint['rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        # if config.TRAIN.NUM_WORKERS > 0:
        #     torch.utils.data.set_worker_seed(checkpoint['worker_rng_states'])
        if config.DEVICE_STR.startswith('cuda'):
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'], device=config.DEVICE_STR)
        logger.info(f"Resuming from epoch {start_epoch+1}")
        logger.info(f"Loss: {loss}")
    else:
        start_epoch = 0

    logger.info("Start training...")
    norm = (input_mean, input_std, output_mean, output_std) if config.TRAIN.USE_NORM else (None, None, None, None)
    train(config=config, network=network,
        optimizer=optimizer, scheduler=scheduler, train_loader=train_loader, bm=bm, norm=norm, start_ep=start_epoch)
    logger.info("Training finished.")



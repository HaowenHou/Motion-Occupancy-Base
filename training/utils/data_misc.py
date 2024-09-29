from collections import OrderedDict
import os
import torch

from utils.quaternion import qinv, qrot

# To transform the 3d points from a sys to another.
# E.g. From global to ego.
# Translate then rotate.
# Inputs: new origin under current system (new sys oritin), 
#         rotation quat (from current sys to the new).
def change_system(pts, origin, rot_q, keep_h=False):
    """Input shapes:
    pts: [BS, ..., 3]
    origin: [BS, 3]
    rot: [BS, 4]"""
    in_shape = pts.shape
    pts = pts.reshape(in_shape[0], -1, 3)
    # Translate
    offset = origin[:, None, :]
    if keep_h:
        offset = offset.clone()
        offset[:, :, 2] = 0
    pts = pts - offset
    # Rotate
    rot_q = qinv(rot_q)
    quat = rot_q[:, None, :].expand(-1, pts.shape[1], -1)
    pts = qrot(quat, pts)
    pts = pts.reshape(in_shape)
    return pts

def trunc_norm(config, input_mean, input_std, output_mean, output_std):
    PAST_KF = config.TRAIN.PAST_KF
    FUTURE_KF = config.TRAIN.FUTURE_KF
    if not config.TRAIN.USE_LIMB_TRAJ:
        tdims = (PAST_KF+1)*4*3 + PAST_KF*4*3
        input_mean = torch.cat([input_mean[:-15-tdims], input_mean[-15:]])
        input_std = torch.cat([input_std[:-15-tdims], input_std[-15:]])
        tdims = FUTURE_KF*4*6
        output_mean = output_mean[:-tdims]
        output_std = output_std[:-tdims]
    if not config.TRAIN.USE_TARGET and not config.TRAIN.get('USE_HAND', False):
        input_mean = input_mean[:-15]
        input_std = input_std[:-15]
    return input_mean, input_std, output_mean, output_std

def save_norm(config, input_mean, input_std, output_mean, output_std):
    to_save = OrderedDict([
        ('input_mean', input_mean),
        ('input_std', input_std),
        ('output_mean', output_mean),
        ('output_std', output_std)
    ])
    torch.save(to_save, os.path.join(config.ASSETS.SPLIT_DIR, config.ASSETS.MEAN_STD_NAME))
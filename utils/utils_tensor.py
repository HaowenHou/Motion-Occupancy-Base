from collections import OrderedDict
import torch


@torch.no_grad()
def flatten_tensor_dict(tensor_dict: OrderedDict):
    """
    # 22*6 + 22*3 + 22*3 + 2KF+1 * (3+3+2+2+12+12)=34 = 910
    """
    joints = torch.cat([v.flatten(1) for k, v in tensor_dict.items() if k in ['joint_rot', 'joint_pos', 'joint_vel']], dim=1) # [F, 22, 12]
    traj_dir = torch.cat([v.flatten(2) for k, v in tensor_dict.items() if k in ['root_traj', 'limb_traj', 'root_dir']], dim=2) # [F, 2KF+1, 17]
    vel = torch.cat([v.flatten(2) for k, v in tensor_dict.items() if k in ['root_vel', 'limb_vel', 'root_rot_vel']], dim=2) # [F, 2KF, 17]
    flatten = torch.cat([joints, traj_dir.flatten(1), vel.flatten(1), tensor_dict['all_traj'], tensor_dict['all_dir']], dim=1) # [F, 22*12+(2KF+1)*17+2KF*17+3]
    return flatten

def parse_input(input_vec, past_kf=10, future_kf=10):
    parsed_dict = OrderedDict()
    parsed_dict['joint_rot'] = input_vec[:, :132]
    parsed_dict['joint_pos'] = input_vec[:, 132:198]
    parsed_dict['joint_vel'] = input_vec[:, 198:264]
    parsed_dict['all_traj'] = input_vec[:, -5:-2]
    parsed_dict['all_dir'] = input_vec[:, -2:]
    traj_dir_win = input_vec[:, 264:264+(past_kf+future_kf+1)*17].view(-1, past_kf+future_kf+1, 17)
    parsed_dict['root_traj'] = traj_dir_win[..., :3]
    parsed_dict['limb_traj'] = traj_dir_win[..., 3:15]
    parsed_dict['root_dir'] = traj_dir_win[..., 15:17]
    vel_win = input_vec[:, 264+(past_kf+future_kf+1)*17:264+(past_kf+future_kf+1)*17+(past_kf+future_kf)*17].view(-1, past_kf+future_kf, 17)
    parsed_dict['root_vel'] = vel_win[..., :3]
    parsed_dict['limb_vel'] = vel_win[..., 3:15]
    parsed_dict['root_rot_vel'] = vel_win[..., 15:17]
    return parsed_dict

def parse_vec(input_vec, past_kf=10, future_kf=10, has_all=False):
    traj_size = past_kf+future_kf+1 if past_kf else future_kf
    parsed_dict = OrderedDict()
    parsed_dict['joint_rot'] = input_vec[..., :132]
    parsed_dict['joint_pos'] = input_vec[..., 132:198]
    parsed_dict['joint_vel'] = input_vec[..., 198:264]
    if has_all:
        parsed_dict['all_traj'] = input_vec[..., -5:-2]
        parsed_dict['all_dir'] = input_vec[..., -2:]
    traj_dir_win = input_vec[..., 264:264+traj_size*17].view(-1, traj_size, 17)
    parsed_dict['root_traj'] = traj_dir_win[..., :3]
    parsed_dict['limb_traj'] = traj_dir_win[..., 3:15]
    parsed_dict['root_dir'] = traj_dir_win[..., 15:17]
    vel_win = input_vec[..., 264+traj_size*17:264+traj_size*17+(past_kf+future_kf)*17].view(-1, past_kf+future_kf, 17)
    parsed_dict['root_vel'] = vel_win[..., :3]
    parsed_dict['limb_vel'] = vel_win[..., 3:15]
    parsed_dict['root_rot_vel'] = vel_win[..., 15:17]

    parsed_dict['joint'] = input_vec[..., :264]
    parsed_dict['traj_dir_win'] = traj_dir_win # [N, PKF+FKF+1, 17]
    parsed_dict['vel_win'] = vel_win # [N, PKF+FKF, 17]
    return parsed_dict

@torch.no_grad()
def save_tensor_dict(save_path, tensor_dict):
    tensor_dict_cpu = OrderedDict()
    for k, v in tensor_dict.items():
        tensor_dict_cpu[k] = v.cpu()
    torch.save(tensor_dict_cpu, save_path)

@torch.no_grad()
def tensor_dict_to(tensor_dict, device, dtype):
    return OrderedDict([(k, v.to(device=device, dtype=dtype)) for k, v in tensor_dict.items()])
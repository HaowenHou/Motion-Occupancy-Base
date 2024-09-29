from collections import OrderedDict
from functools import partial
from multiprocessing import Manager, get_context
import math
import numpy as np
import os
import pickle
import time
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils.quaternion import aa_to_quat, cont6d_to_matrix, matrix_to_axis_angle, matrix_to_quat, qbetween, qinv, qmul, qrot, quat_to_aa, quat_to_6d, quaternion_to_matrix, radian_to_quat
from body_model.body_model import BodyModel
from utils.utils_smpl import get_template_joints, smpl_fk
from utils.utils import error_handler, init_queue


class ProcessDataFilter:
    def __init__(self, config):
        self.DEVICE = config.DEVICE
        self.DTYPE = eval(f'torch.{config.DTYPE}')
        self.BATCH_SIZE = config.BATCH_SIZE
        self.FPS = config.FPS
        self.ROLLOUT = config.ROLLOUT
        self.PAST_KF = config.PAST_KF
        self.FUTURE_KF = config.FUTURE_KF
        self.AMASS_DIR = config.AMASS_DIR
        self.MALE_BM_PATH = config.MALE_BM_PATH
        self.FEMALE_BM_PATH = config.FEMALE_BM_PATH

    def load_smpl_body_model(self):
        num_betas = 10 # number of body parameters
        self.male_bm = BodyModel(bm_fname=self.MALE_BM_PATH, num_betas=num_betas, dtype=self.DTYPE).to(self.DEVICE)
        self.female_bm = BodyModel(bm_fname=self.FEMALE_BM_PATH, num_betas=num_betas, dtype=self.DTYPE).to(self.DEVICE)
        self.template_joints, _ = get_template_joints(self.male_bm)
        self.parents = self.male_bm.kintree_table[0, :22]

    @torch.no_grad()
    def process_smpl(self, data_path):
        """
        Takes AMASS data in SMPL.
        Returns in right-handed coordinate system.
        Positions are already summed up with trajectory.
        The root joint is set to the origin in smpl.
        [frames, 22, 3]
        """
        if data_path.endswith('.pkl'):
            data = pickle.load(open(data_path, 'rb')) # CIRCLE
        else:
            data = np.load(data_path)
        if 'trans' not in data:
            return None
        frame_number = data['trans'].shape[0]
        ori_fps = int(data.get('mocap_framerate', 30)) # CIRCLE: 30fps
        downsample_rate = ori_fps // self.FPS

        # Get bone length scale to scale the trajectory.
        betas = torch.tensor(data['betas'][:10][np.newaxis], dtype=self.DTYPE, device=self.DEVICE) # controls the body shape
        bm = self.male_bm if data['gender'] == 'male' else self.female_bm
        shaped_rest_pose = bm(betas=betas).Jtr[0]
        get_leg_length = lambda pose: torch.linalg.vector_norm(pose[4] - pose[1]) + torch.linalg.vector_norm(pose[7] - pose[4])
        template_leg_length = get_leg_length(self.template_joints)
        shaped_leg_length = get_leg_length(shaped_rest_pose)
        length_scale = template_leg_length / shaped_leg_length

        root_trajectory = torch.tensor(data['trans'], dtype=self.DTYPE, device=self.DEVICE)
        root_trajectory = length_scale * root_trajectory

        poses = torch.tensor(data['poses'][:, :66], dtype=self.DTYPE, device=self.DEVICE) # controls the body
        
        pose_seq = []
        valid_frame_ids = list(range(0, frame_number, downsample_rate))
        if len(valid_frame_ids) <= 1:
            return None
        for i in range(math.ceil(len(valid_frame_ids) / self.BATCH_SIZE)):
            batch_ids = valid_frame_ids[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
            rotations = poses[batch_ids, :66].view(-1, 22, 3)
            joints = smpl_fk(self.template_joints, rotations, self.parents)
            # joints = bm(pose_body=poses[batch_ids, 3:], betas=betas, root_orient=poses[batch_ids, :3]).Jtr[:, :22]
            pose_seq.append(joints)
        pose_seq = torch.cat(pose_seq, dim=0) # [F, 22, 3]
        root_trajectory = root_trajectory[valid_frame_ids] # [F, 3]
        pose_seq = pose_seq + root_trajectory[:, None, :].expand_as(pose_seq)

        # Put on the floor.
        lowest_height = pose_seq[..., 2].min()
        root_trajectory[..., 2] -= lowest_height
        pose_seq[..., 2] -= lowest_height

        return {
            'smpl_rot': poses[valid_frame_ids].reshape(-1, 22, 3), # [F, 22, 3]
            'joint_pos': pose_seq, # [F, 22, 3]
            'root_traj': root_trajectory # [F, 22, 3]
        }

    @torch.no_grad()
    def generate_data(self, smpl_dict):
        smpl_dict = {k: v.to(device=self.DEVICE, dtype=self.DTYPE) for k, v in smpl_dict.items()}
        smpl_rot = smpl_dict['smpl_rot']
        joint_pos = smpl_dict['joint_pos']
        N_FRAMES = len(joint_pos)
        root_traj = smpl_dict['root_traj']
        target_rot_quat = aa_to_quat(smpl_rot[:, 0, :]) # [F, 4]

        # Root direction.
        l_hip, r_hip, l_sdr, r_sdr = (1, 2, 16, 17)
        right_dir = (joint_pos[:, r_hip] - joint_pos[:, l_hip]) + (joint_pos[:, r_sdr] - joint_pos[:, l_sdr])
        root_dir = torch.stack([-right_dir[:, 1], right_dir[:, 0], torch.zeros_like(right_dir[:, 0])], dim=-1) # [F, 3]
        root_dir = F.normalize(root_dir, dim=-1)
        # Make the skeleton face Y+.
        local_to_global_quat = qbetween(root_dir, torch.tensor([[0.0, 1.0, 0.0]], dtype=self.DTYPE, device=self.DEVICE)) # [F, 4]
        local_to_target_quat = qmul(local_to_global_quat, target_rot_quat)
        local_to_target_vec = quat_to_aa(local_to_target_quat)
        smpl_rot[:, 0] = local_to_target_vec
        # # Joint velocity.
        # joint_vel = qrot(local_to_global_quat[:-1, None, :].expand(-1, 22, -1), joint_pos[1:] - joint_pos[:-1]) # [F-1, 22, 3]
        # joint_vel = torch.cat([joint_vel, joint_vel[-1:]], dim=0) # [F, 22, 3]
        # Root rotation velocity.
        root_dir_vel_y = qrot(local_to_global_quat[:-1], root_dir[1:]) # [F-1, 3]
        root_rot_vel_cs = torch.stack([root_dir_vel_y[..., 1], -root_dir_vel_y[..., 0]], dim=-1) # [F-1, 2]
        root_rot_vel_cs = torch.cat([root_rot_vel_cs, torch.tensor([[1.0, 0.0]], dtype=self.DTYPE, device=self.DEVICE)], dim=0) # [F, 2]

        def generate_id_matrix(seq_len, extend_past, extend_future, omit_last=False):
            """
            Generate an id matrix for shifting window.
            Before clamped: [F, 2*KF+1]  ([[-KF, -KF+1, ...], [-KF+1, -KF+2, ...], ...])
            Clamped.
            """
            valid_frame_ids = torch.arange(seq_len)
            frame_offset = torch.arange(start=-extend_past, end=extend_future+1)
            frame_ids = valid_frame_ids[:, None] + frame_offset
            frame_ids = torch.clamp(frame_ids, min=0, max=seq_len-1 if not omit_last else seq_len-2)
            return frame_ids

        # Calculate facing directions in each egocentric window.
        frame_ids = generate_id_matrix(seq_len=N_FRAMES, extend_past=self.PAST_KF, extend_future=self.FUTURE_KF) # [F, 2KF+1]
        root_dir_win = root_dir[frame_ids] # [F, 2KF+1, 3]
        root_dir_win = qrot(local_to_global_quat[:, None, :].expand(-1, self.PAST_KF+self.FUTURE_KF+1, -1), root_dir_win) # [F, 2KF+1, 3]
        root_dir_win_cs = root_dir_win[..., [0, 1]] # [F, 2KF+1, 2]

        # Ego/root/limb trajectories and velocities in each egocentric window.
        # root, l_foot, r_foot, l_hand, r_hand = (0, 10, 11, 20, 21)
        root_and_limbs = (0, 10, 11, 20, 21)
        joint_traj = joint_pos[:, root_and_limbs, :] # [F, n_limbs+1, 3]
        frame_ids = generate_id_matrix(seq_len=N_FRAMES, extend_past=self.PAST_KF, extend_future=self.FUTURE_KF)
        joint_traj_win = joint_traj[frame_ids] # [F, 2KF+1, n_limbs+1, 3]
        local_to_global_quat_tmp = local_to_global_quat[:, None, None, :].expand(joint_traj_win.size()[:-1] + (4,))
        joint_traj_win = qrot(local_to_global_quat_tmp, joint_traj_win) # [F, 2KF+1, n_limbs+1, 3]
        joint_traj_win[..., [0, 1]] -= joint_traj_win[:, self.PAST_KF:self.PAST_KF+1, 0:1, [0, 1]].clone() # [F, 2KF+1, n_limbs+1, 3]
        joint_vel_win = joint_traj_win[:, 1:] - joint_traj_win[:, :-1] # [F, 2KF, n_limbs+1, 3]
        # Root rotation velocities in window.
        frame_ids = generate_id_matrix(seq_len=N_FRAMES, extend_past=self.PAST_KF, extend_future=self.FUTURE_KF-1, omit_last=True)
        root_rot_vel_win_cs = root_rot_vel_cs[frame_ids] # [F, 2KF, 2]
        # Root-centric (for XY plane) joint positions.
        joint_pos = joint_pos.clone() # [F, 22, 3]
        joint_pos[..., [0, 1]] -= root_traj[:, None, [0, 1]]
        joint_pos = qrot(local_to_global_quat[:, None, :].expand(N_FRAMES, 22, 4), joint_pos)
        # Joint velocity.
        joint_vel = qrot(local_to_global_quat[:-1, None, :].expand(-1, 22, -1), joint_pos[1:] - joint_pos[:-1]) # [F-1, 22, 3]
        joint_vel = torch.cat([joint_vel, joint_vel[-1:]], dim=0) # [F, 22, 3]
        # Set the first root direction to Y+.
        root_traj = root_traj.clone()
        root_traj[..., [0, 1]] -= root_traj[0, [0, 1]].clone()
        root_traj = qrot(local_to_global_quat[0:1].expand(N_FRAMES, -1), root_traj) # [F, 3]
        root_dir = qrot(local_to_global_quat[0:1].expand(N_FRAMES, -1), root_dir) # [F, 3]

        """
        # f-1 * (2KF+1 * 32 + 22 * 12) = f-1 * 872
        ~1M frames in HumanML3D
        # data size = 1M * 872 * 4B = 3.5GB
        """
        tensor_dict = OrderedDict({
            'joint_rot': quat_to_6d(aa_to_quat(smpl_rot.view(-1, 3))).view(N_FRAMES, 22, 6), # [F, 22, 6]
            'joint_pos': joint_pos, # [F, 22, 3] Feet on floor.
            'joint_vel': joint_vel, # [F, 22, 3]

            'root_traj': joint_traj_win[..., 0, :], # [F, 2KF+1, 3] Feet on floor.
            'limb_traj': joint_traj_win[..., 1:, :], # [F, 2KF+1, 4, 3] Feet on floor.
            'root_dir': root_dir_win_cs, # [F, 2KF+1, 2]

            'root_vel': joint_vel_win[..., 0, :], # [F, 2KF, 3]
            'limb_vel': joint_vel_win[..., 1:, :], # [F, 2KF, 4, 3]
            'root_rot_vel': root_rot_vel_win_cs, # [F, 2KF, 2]

            'all_traj': root_traj, # [F, 3]
            'all_dir': root_dir[..., :-1] # [F, 2]
        })
        return tensor_dict
    
    def get_tensor_dict(self, path):
        data_path = os.path.join(self.AMASS_DIR, path + '.npz')
        smpl = self.process_smpl(data_path)
        if not smpl:
            return
        tensor_dict = self.generate_data(smpl)
        return tensor_dict

    def filter_data(self, tensor_dict):
        WIN_MIN = max(self.ROLLOUT, 1)
        WIN_MAX = 30 # 10 fps
        snippets = []
        n_frames = len(tensor_dict['joint_rot'])
        for start_fid in range(n_frames-self.ROLLOUT):
            for end_fid in range(start_fid+WIN_MIN-1, min(n_frames, start_fid+WIN_MAX)):
                # Experimental: Data can be filtered here
                snippets.append((start_fid, end_fid))
        return snippets

    def run(self, mid, path, queue=None):
        data_path = os.path.join(self.DATASET_DIR, path + '.pkl')
        smpl = self.process_smpl(data_path)
        if not smpl:
            return
        tensor_dict = self.generate_data(smpl)
        snippets = self.filter_data(tensor_dict)
        if queue:
            queue.put(1)
        return snippets


if __name__ == '__main__':
    config = OmegaConf.load("process_data_config.yml")
    INDEX_PATH = config.INDEX_CSV_PATH

    with open(INDEX_PATH, 'r') as f:
        lines = f.readlines()
    mid_path_dict = {}
    for i in range(len(lines)):
        line = lines[i]
        line = line.strip().split(',')
        mid_path_dict[int(line[0])] = line[1]

    NUM_PROCESSES = 1 # Set >1 to enable multi-processing for speedup.
    process_data = ProcessDataFilter(config)
    process_data.load_smpl_body_model()
    mid_snippets_dict = {}

    if NUM_PROCESSES == 1:
        pbar = tqdm(total=len(mid_path_dict))
        for mid, path in mid_path_dict.items():
            snippets = process_data.run(mid, path)
            if snippets:
                mid_snippets_dict[mid] = snippets
            pbar.update(1)
    else:
        manager = Manager()
        queue = manager.Queue()
        total_tasks = len(mid_path_dict)
        partial_visualize = partial(process_data.visualize, queue=queue)
        
        with get_context("spawn").Pool(initializer=init_queue, initargs=(queue,), processes=NUM_PROCESSES) as pool:
            result_iterator = pool.starmap_async(partial_visualize, mid_path_dict.items(), error_callback=error_handler)
            pbar = tqdm(total=total_tasks)
            completed_tasks = 0
            while completed_tasks < total_tasks:
                while not queue.empty():
                    queue.get()
                    completed_tasks += 1
                    pbar.update(1)
                time.sleep(0.1)
            pbar.close()

    # Save dict as pickle.
    MID_SNIP_DICT_DIR = config.MID_SNIP_DICT_DIR
    os.makedirs(MID_SNIP_DICT_DIR, exist_ok=True)
    DICT_NAME = 'mid_snip_dict.pkl'
    with open(os.path.join(MID_SNIP_DICT_DIR, DICT_NAME), 'wb') as f:
        pickle.dump(mid_snippets_dict, f)

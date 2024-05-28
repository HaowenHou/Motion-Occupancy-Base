from collections import OrderedDict
import math
import os
import pickle
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from utils.quaternion import aa_to_quat, cont6d_to_matrix, matrix_to_axis_angle, matrix_to_quat, qbetween, qinv, qmul, qrot, quat_to_aa, quat_to_6d, quaternion_to_matrix, radian_to_quat
from body_model.body_model import BodyModel
from utils.utils_smpl import get_template_joints, smpl_fk

from tqdm import tqdm
from utils.utils_tensor import flatten_tensor_dict

class ProcessDataBase:
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
    
    def get_root_dir(self, joint_pos):
        l_hip, r_hip, l_sdr, r_sdr = (1, 2, 16, 17)
        right_dir = (joint_pos[:, r_hip] - joint_pos[:, l_hip]) + (joint_pos[:, r_sdr] - joint_pos[:, l_sdr])
        root_dir = torch.stack([-right_dir[:, 1], right_dir[:, 0], torch.zeros_like(right_dir[:, 0])], dim=-1) # [F, 3]
        root_dir = F.normalize(root_dir, dim=-1)
        l2g_quat = qbetween(root_dir, torch.tensor([[0.0, 1.0, 0.0]], dtype=self.DTYPE, device=self.DEVICE)) # [F, 4]
        return root_dir, l2g_quat
    
    def cano_seq(self, joint_pos, l2g_quat):
        joint_pos = joint_pos.clone() # [F, 22, 3]
        joint_pos[..., [0, 1]] -= joint_pos[0:1, 0:1, [0, 1]].clone()
        joint_pos = qrot(l2g_quat[0:1, None, :].expand(len(joint_pos), 22, 4), joint_pos)
        return joint_pos
    
    def cano_every(self, joint_pos, l2g_quat):
        joint_pos = joint_pos.clone() # [F, 22, 3]
        joint_pos[..., [0, 1]] -= joint_pos[:, 0:1, [0, 1]].clone()
        joint_pos = qrot(l2g_quat[:, None, :].expand(-1, 22, 4), joint_pos)
        return joint_pos

    def cano_traj(self, root_traj, l2g_quat):
        root_traj = root_traj.clone()
        root_traj[..., [0, 1]] -= root_traj[0:1, [0, 1]].clone()
        root_traj = qrot(l2g_quat[0:1].expand(len(root_traj), 4), root_traj)
        l2g_quat = qmul(qinv(l2g_quat[0:1].expand(len(l2g_quat), -1)), l2g_quat)
        ydir = torch.tensor([[0.0, 1.0, 0.0]], dtype=self.DTYPE, device=self.DEVICE)
        fdir = qrot(qinv(l2g_quat), ydir.expand(len(l2g_quat), 3))
        return root_traj, fdir
    
    def cano_aa(self, j_aa, l2g_quat):
        g2tgt_quat = aa_to_quat(j_aa[:, 0, :]) # [F, 4]
        l2tgt_quat = qmul(l2g_quat, g2tgt_quat)
        l2tgt_aa = quat_to_aa(l2tgt_quat)
        j_aa[:, 0] = l2tgt_aa
        return j_aa

    @torch.no_grad()
    def generate_data(self, smpl_dict):
        smpl_dict = {k: v.to(device=self.DEVICE, dtype=self.DTYPE) for k, v in smpl_dict.items()}
        # smpl_rot = smpl_dict['smpl_rot']
        joint_pos = smpl_dict['joint_pos']
        N_FRAMES = len(joint_pos)
        root_traj = smpl_dict['root_traj']
        
        root_dir, l2g_quat = self.get_root_dir(joint_pos)
        # assert torch.allclose(root_traj[:, None, [0, 1]], joint_pos[:, 0:1, [0, 1]])
        j_abs = self.cano_seq(joint_pos, l2g_quat)
        j_ego = self.cano_every(joint_pos, l2g_quat)
        traj, fdir = self.cano_traj(root_traj, l2g_quat)

        j_aa = smpl_dict['smpl_rot']
        j_aa = self.cano_aa(j_aa, l2g_quat)
        j_6d = quat_to_6d(aa_to_quat(j_aa))
        
        return j_6d, j_ego, j_abs, traj, fdir
    
    def get_data(self, path):
        data_path = os.path.join(self.AMASS_DIR, path + '.pkl')
        smpl = self.process_smpl(data_path)
        if not smpl:
            return
        data = self.generate_data(smpl)
        return data
    

class ProcessDataGenerate(ProcessDataBase):
    def __init__(self, config):
        super().__init__(config)
        self.NPY_SAVE_DIR = config.NPY_SAVE_DIR
        os.makedirs(self.NPY_SAVE_DIR, exist_ok=True)

    def save_npy(self, mid, path):
        data = self.get_data(path)
        if data is None:
            return
        # dims = [132, 66, 66, 3, 3]
        j_6d, j_ego, j_abs, traj, fdir = data
        to_save = torch.cat([j_6d.flatten(1), j_ego.flatten(1), j_abs.flatten(1), 
                          traj, fdir], dim=-1)
        npy_save_path = os.path.join(self.NPY_SAVE_DIR, f'{mid:08}.npy')
        to_save = to_save.cpu().float().numpy().astype(np.float32)
        np.save(npy_save_path, to_save)


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

    process_data = ProcessDataGenerate(config)
    process_data.load_smpl_body_model()

    pbar = tqdm(total=len(mid_path_dict))
    for mid, path in mid_path_dict.items():
        process_data.save_npy(mid, path)
        pbar.update(1)
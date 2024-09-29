from collections import OrderedDict
import pickle
import joblib
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import logging
import torch.nn.functional as F

from tqdm import tqdm

ELIMBS = (0, 10, 11, 20, 21) # root, l_foot, r_foot, l_hand, r_hand
LIMBS = ELIMBS[1:]

def normalize(data, mean, std):
    epsilon = 1e-7
    mask = std.abs() < epsilon
    normed = (data - mean) / std
    normed[..., mask] = 0.0
    return normed

def denormalize(data, mean, std):
    return data * std + mean

class MphaseDataset(Dataset):
    def __init__(self, config: OmegaConf, split: str = 'train'):
        super().__init__()
        self.CONFIG       = config
        self.ROLLOUT      = eval(f'config.{split.upper()}.ROLLOUT_FRAMES')
        self.SPLIT        = split
        self.USE_VOXEL    = config.TRAIN.USE_VOX
        self.USE_BPS      = config.TRAIN.get('USE_BPS', False) | config.TRAIN.get('USE_GRID_LOSS', False)
        self.PRE_LOAD_VOX = config.TRAIN.get('PRE_LOAD_VOX', True)
        self.USE_TGT      = config.TRAIN.USE_TARGET

        # Load snippet array.
        snip_dict_path = os.path.join(config.ASSETS.SPLIT_DIR, config.ASSETS.SNIP_DICT_NAME)
        with open(snip_dict_path, 'rb') as f:
            mid_snip_dict = pickle.load(f)
        mid_list = []
        split_file = os.path.join(config.ASSETS.SPLIT_DIR, split.lower() + '.txt')
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                mid_list.append(int(line.strip()))
        # mid_snip_dict = {idx: mid_snip_dict[idx] for idx in id_list}
        lst = [(mid, start_fid, end_fid) for mid in mid_list for start_fid, end_fid in mid_snip_dict[mid]]
        snip_ary = np.array(lst)
        logging.info(f'Loaded {len(snip_ary)} snippets from {split} split.')
        if config.STAGE == 'TRAIN':
            snip_ary = snip_ary[snip_ary[:, 2] - snip_ary[:, 1] >= self.ROLLOUT - 1]
        elif config.STAGE == 'INFER':
            snip_ary = snip_ary[snip_ary[:, 2] - snip_ary[:, 1] == config.INFER.SNIP_LEN]
        logging.info(f'After filtering, {len(snip_ary)} snippets left.')
        self.snip_ary = snip_ary
        self.mid_list = mid_list

        # Load data.
        self.data_dict = {}
        pbar = tqdm(total=len(self.mid_list), desc='Loading data')
        for mid in self.mid_list:
            file_path = os.path.join(config.ASSETS.NPY_DIR, f'{mid:08}.npy')
            data_np = np.load(file_path)
            data = torch.from_numpy(data_np)
            if self.CONFIG.TRAIN.TO_GPU_FIRST:
                data = data.to(self.CONFIG.DEVICE_STR, non_blocking=True)
            self.data_dict[mid] = data
            pbar.update(1)
        pbar.close()

        if config.ASSETS.get('SPLIT_DIR1', False):
            max_mid0 = max(mid_list)
            # Load snippet array.
            snip_dict_path = os.path.join(config.ASSETS.SPLIT_DIR1, config.ASSETS.SNIP_DICT_NAME)
            with open(snip_dict_path, 'rb') as f:
                mid_snip_dict = pickle.load(f)
            mid_list = []
            split_file = os.path.join(config.ASSETS.SPLIT_DIR1, split.lower() + '.txt')
            with open(split_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    mid_list.append(int(line.strip()))
            # mid_snip_dict = {idx: mid_snip_dict[idx] for idx in id_list}
            lst = [(mid, start_fid, end_fid) for mid in mid_list for start_fid, end_fid in mid_snip_dict[mid]]
            snip_ary = np.array(lst)
            logging.info(f'Loaded {len(snip_ary)} snippets from {split} split.')
            if config.STAGE == 'TRAIN':
                snip_ary = snip_ary[snip_ary[:, 2] - snip_ary[:, 1] >= self.ROLLOUT - 1]
            elif config.STAGE == 'INFER':
                snip_ary = snip_ary[snip_ary[:, 2] - snip_ary[:, 1] == config.INFER.SNIP_LEN]
            logging.info(f'After filtering, {len(snip_ary)} snippets left.')

            # load another data dict
            pbar = tqdm(total=len(mid_list), desc='Loading data')
            for mid in mid_list:
                file_path = os.path.join(config.ASSETS.NPY_DIR1, f'{mid:08}.npy')
                data_np = np.load(file_path)
                data = torch.from_numpy(data_np)
                if self.CONFIG.TRAIN.TO_GPU_FIRST:
                    data = data.to(self.CONFIG.DEVICE_STR, non_blocking=True)
                self.data_dict[mid+max_mid0+1] = data
                pbar.update(1)

        self.PAST_KF = config.TRAIN.PAST_KF
        self.FUTURE_KF = config.TRAIN.FUTURE_KF

        # Voxel.
        if self.USE_VOXEL:
            if self.PRE_LOAD_VOX:
                max_shape = torch.zeros(3)
                # avg_shape = torch.zeros(3)
                self.occu_g_dict  = {}
                self.occu_g_shape = {}
                pbar = tqdm(total=len(self.mid_list), desc='Loading global occu')
                for mid in self.mid_list:
                    file_path = os.path.join(config.ASSETS.OCCUG_DIR, f'{mid:08}.pkl')
                    occu_g, unit, llb = pickle.load(open(file_path, 'rb'))
                    occu_g = torch.from_numpy(occu_g).bool() # 1 is wall.
                    # update max shape
                    self.occu_g_shape[mid] = torch.tensor(occu_g.shape)
                    max_shape = torch.max(max_shape, self.occu_g_shape[mid])
                    # avg_shape += torch.tensor(occu_g.shape)
                    llb = torch.from_numpy(llb) # 1 is wall.
                    self.occu_g_dict[mid] = (occu_g, llb)
                    pbar.update(1)
                pbar.close()

                if config.ASSETS.get('SPLIT_DIR1', False):
                    pbar = tqdm(total=len(mid_list), desc='Loading global occu')
                    for mid in mid_list:
                        file_path = os.path.join(config.ASSETS.OCCUG_DIR1, f'{mid:08}.pkl')
                        occu_g, unit, llb = pickle.load(open(file_path, 'rb'))
                        occu_g = torch.from_numpy(occu_g).bool() # 1 is wall.
                        # update max shape
                        self.occu_g_shape[mid+max_mid0+1] = torch.tensor(occu_g.shape)
                        max_shape = torch.max(max_shape, self.occu_g_shape[mid+max_mid0+1])
                        # avg_shape += torch.tensor(occu_g.shape)
                        llb = torch.from_numpy(llb) # 1 is wall.
                        self.occu_g_dict[mid+max_mid0+1] = (occu_g, llb)
                        pbar.update(1)
                    pbar.close()

                # # print(max_shape)
                # # print(avg_shape / len(self.mid_list))
                # if config.TRAIN.BATCH_VOX:
                #     # Pad all occug to max shape with 1.
                #     pbar = tqdm(total=len(self.mid_list), desc='Padding global occu')
                #     for mid in self.mid_list:
                #         occu_g, llb = self.occu_g_dict[mid]
                #         occu_g = self.pad_tensor(occu_g, max_shape, 1)
                #         self.occu_g_dict[mid] = (occu_g, llb)
                #         pbar.update(1)
                if self.USE_BPS:
                    self.occu_g_ref = {}
                    pbar = tqdm(total=len(self.mid_list), desc='Loading global occu reference')
                    for mid in self.mid_list:
                        ref = np.load(os.path.join(config.ASSETS.OCCUG_REF_DIR, f'{mid:08}.npy'))
                        self.occu_g_ref[mid] = torch.from_numpy(ref)
                        # self.occu_g_ref[mid] = F.pad(torch.from_numpy(ref), (0, 0, 0, int(max_shape[2] - self.occu_g_shape[mid][2]), 0, int(max_shape[1] - self.occu_g_shape[mid][1]), 0, int(max_shape[0] - self.occu_g_shape[mid][0])))
                        pbar.update(1)
                    pbar.close()

        if config.ASSETS.get('SPLIT_DIR1', False):
            snip_ary[:, 0] += max_mid0 + 1
            mid_list = [mid + max_mid0 + 1 for mid in mid_list]
            self.snip_ary = np.concatenate([self.snip_ary, snip_ary], axis=0)
            self.mid_list = np.concatenate([self.mid_list, mid_list], axis=0)
    
    def pad_tensor(self, tensor, max_shape, pad_val):
        # Calculate padding for each dimension
        pad_x = int(max_shape[0] - tensor.shape[0])
        pad_y = int(max_shape[1] - tensor.shape[1])
        pad_z = int(max_shape[2] - tensor.shape[2])
        # Apply padding. The order is reverse: (left, right, top, bottom, front, back)
        padded_tensor = F.pad(tensor, (0, pad_z, 0, pad_y, 0, pad_x), value=pad_val)
        return padded_tensor
            
    def __len__(self):
        return len(self.snip_ary)
    
    @staticmethod
    def parse_npy_data(data):
        dims = [132, 66, 66, 3, 3]
        dims = [sum(dims[:i]) for i in range(len(dims)+1)]
        j_6d = data[..., dims[0]:dims[1]].view(-1, 22, 6)
        j_ego = data[..., dims[1]:dims[2]].view(-1, 22, 3)
        j_abs = data[..., dims[2]:dims[3]].view(-1, 22, 3)
        traj = data[..., dims[3]:dims[4]].view(-1, 3)
        fdir = data[..., dims[4]:dims[5]].view(-1, 3)
        return j_6d, j_ego, j_abs, traj, fdir
    
    def get_info(self, mid, st_fid):
        st = st_fid-self.PAST_KF
        en = st_fid+self.ROLLOUT+self.FUTURE_KF+1
        data = self.data_dict[mid][max(0, st):en]
        if st < 0:
            data = pad_first_dim(data, -st+len(data), pos='before')
        if en > len(self.data_dict[mid]):
            data = pad_first_dim(data, en-len(self.data_dict[mid])+len(data), pos='after')
        j_6d, j_ego, j_abs, traj, fdir = self.parse_npy_data(data)
        return j_6d, j_ego, j_abs, traj, fdir
    
    def get_target(self, mid, end_fid):
        data = self.data_dict[mid][end_fid:end_fid+1]
        _, _, j_abs, _, fdir = self.parse_npy_data(data)
        return j_abs[0], fdir[0]
    
    def __getitem__(self, idx):
        mid, st_fid, end_fid = self.snip_ary[idx]

        # Info in range [st-past_kf, st+rollout+future_kf]
        # j:joints. fdir: forward direction.
        j_6d, j_ego, j_abs, traj, fdir = self.get_info(mid, st_fid)
        
        data = {
            'j_6d': j_6d, # [F, 22, 6]
            'j_ego': j_ego, # [F, 22, 3]
            'j_abs': j_abs, # [F, 22, 3]
            'traj': traj, # [F, 3]
            'fdir': fdir, # [F, 3]
            'mid_snip': (mid, st_fid, end_fid, idx),
        }

        if self.USE_TGT:
            j_tgt, tgt_fdir = self.get_target(mid, end_fid)
            data.update({
                'jabs_tgt': j_tgt, # [1, 22, 3]
                'tgt_fdir': tgt_fdir, # [1, 3]
            })

        if self.USE_VOXEL:
            if self.PRE_LOAD_VOX:
                # data['voxel'] = vox_data
                occu_g, llb = self.occu_g_dict[mid]
                data['llb'] = llb[0]
                data['shape'] = self.occu_g_shape[mid]
                if self.CONFIG.TRAIN.BATCH_VOX:
                    data['vox'] = occu_g.float()
                # else:
                #     data['mid'] = mid
                if self.USE_BPS:
                    data['ref'] = self.occu_g_ref[mid]
            else:
                occu_g, unit, llb = pickle.load(open(os.path.join(self.CONFIG.ASSETS.OCCUG_DIR, f'{mid:08}.pkl'), 'rb'))
                data['vox'] = occu_g
                data['llb'] = llb[0]
        return data

def pad_first_dim(tensor, tgt_len, pos='after'):
    if len(tensor) == tgt_len:
        return tensor
    assert len(tensor) < tgt_len
    pad_len = tgt_len - len(tensor)
    if pos == 'after':
        last = tensor[-1:]
        pad_mat = torch.cat([last] * pad_len, dim=0)
        return torch.cat([tensor, pad_mat], dim=0)
    if pos == 'before':
        first = tensor[:1]
        pad_mat = torch.cat([first] * pad_len, dim=0)
        return torch.cat([pad_mat, tensor], dim=0)
    raise ValueError(f'Unknown pos {pos}')

def collate_fn(data):
    batch = {}
    b = len(data)
    for key in data[0].keys():
        if key in ['j_6d', 'j_ego', 'j_abs', 'traj', 'fdir', 'jabs_tgt', 'tgt_fdir', 'shape', 'llb', 'hand_ego', 'hand_abs']:
            batch[key] = torch.cat([data[i][key][None] for i in range(b)])
    if 'mid_snip' in data[0]:
        batch['mid_snip'] = torch.tensor([data[i]['mid_snip'] for i in range(b)])
    if 'shape' in data[0]:
        max_shape = torch.max(batch['shape'], dim=0).values
    if 'vox' in data[0]:
        batch['vox'] = torch.cat([F.pad(data[i]['vox'], (0, int(max_shape[2] - batch['shape'][i, 2]), 0, int(max_shape[1] - batch['shape'][i, 1]), 0, int(max_shape[0] - batch['shape'][i, 0])), value=1)[None] for i in range(b)])
    if 'ref' in data[0]:
        batch['ref'] = torch.cat([F.pad(data[i]['ref'], (0, 0, 0, int(max_shape[2] - batch['shape'][i, 2]), 0, int(max_shape[1] - batch['shape'][i, 1]), 0, int(max_shape[0] - batch['shape'][i, 0])))[None] for i in range(b)])

    return batch

import torch
import torch.nn as nn

from utils.occu import get_delta_v, project
from dataset import normalize, denormalize


class TransformerBlock(nn.Module):
    def __init__(self, feat_dim, n_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, 
            num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*4),
            nn.ELU(),
            nn.Linear(feat_dim*4, feat_dim),
        )
        self.ln2 = nn.LayerNorm(feat_dim)

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x)[0])
        x = self.ln2(x + self.mlp(x))
        return x
    
class ToVecTransfBlock(nn.Module):
    def __init__(self, feat_dim, n_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, 
            num_heads=n_heads, dropout=dropout, batch_first=True)
        self.query  = nn.Parameter(torch.randn(1, feat_dim))
        self.ln_1   = nn.LayerNorm(feat_dim)
        self.mlp    = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim*4, feat_dim),
            nn.Dropout(dropout),
        )
        self.ln_2 = nn.LayerNorm(feat_dim)

    def forward(self, x):
        bs = x.size(0)
        q = self.query[None].expand(bs, -1, -1)
        x = self.attn(q, x, x)[0] # [N, 1, 512]
        x = self.ln_1(x + q)
        x = self.ln_2(x + self.mlp(x))
        return x.squeeze(1)

class CtrlTransf(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        mconfig     = config.MODEL.CTRL_TRANSF
        FEATURE_DIM = mconfig.FEATURE_DIM
        self.RANDOM = config.MODEL.get('RANDOM', False)
        self.proj_joints = nn.Linear(config.MODEL.JOINT_INDIM, FEATURE_DIM)
        self.proj_traj = nn.Linear(config.MODEL.TRAJ_DIM, FEATURE_DIM)
        in_dims = [config.MODEL.JOINT_INDIM, config.MODEL.TRAJ_DIM]
        if config.TRAIN.USE_TARGET:
            tgt_dim = 15
            if config.TRAIN.get('CLOSE_SW', False) and config.TRAIN.get('CLOSE_LABEL', False):
                tgt_dim += 1
            self.proj_tgt = nn.Linear(tgt_dim, FEATURE_DIM)
            in_dims.append(tgt_dim)
        if config.TRAIN.USE_VOX:
            if config.TRAIN.get('USE_BPS', False):
                bps_dim = int(config.ASSETS.BASIS_PATH.split('.')[0].split('_')[1])
                in_dims.append(bps_dim)
                self.proj_vox = nn.Linear(bps_dim, FEATURE_DIM)
            else:
                gsize = config.TRAIN.GRID_SIZE
                grid_dim = gsize[0] * gsize[1] * gsize[2]
                in_dims.append(grid_dim)
                self.proj_vox = nn.Linear(grid_dim, FEATURE_DIM)
        if config.TRAIN.get('USE_HAND', False):
            hand_dim = (config.TRAIN.HAND_PAST_KF + config.TRAIN.HAND_FUTURE_KF + 1) * 2 * 3
            self.proj_hand = nn.Linear(hand_dim, FEATURE_DIM)
            in_dims.append(hand_dim)

        self.transf_blocks = nn.Sequential(
            *[TransformerBlock(FEATURE_DIM, mconfig.NUM_HEADS) for _ in range(mconfig.MID_LAYERS)])
        self.to_vec_block = ToVecTransfBlock(FEATURE_DIM, mconfig.NUM_HEADS, mconfig.TRANSF_DROPOUT)
        if self.RANDOM:
            self.out_proj = nn.Linear(FEATURE_DIM * 2, config.MODEL.OUT_DIM)
        else:
            self.out_proj = nn.Linear(FEATURE_DIM, config.MODEL.OUT_DIM)

        if config.TRAIN.get('USE_FIELD', False):
            if config.TRAIN.get('USE_BPS', False):
                self.alpha = config.TRAIN.ALPHA_COEFF / bps_dim
            else:
                self.alpha = config.TRAIN.ALPHA_COEFF / grid_dim
            # if config.TRAIN.SPECIFY_ALPHA:
            #     self.alpha = nn.Parameter(torch.ones([]) * config.TRAIN.ALPHA_VALUE)
            # else:
            #     self.alpha = nn.Parameter(torch.ones([]) * config.TRAIN.ALPHA_COEFF / gsize[0]**3)
            # if config.TRAIN.FIX_ALPHA:
            #     self.alpha.requires_grad = False
            # if config.TRAIN.CLIP_DV:
            #     self.clip_coeff = nn.Parameter(torch.ones([]) * self.config.TRAIN.CLIP_COEFF, requires_grad=False)
            # self.dist_offset = nn.Parameter(torch.ones([]) * config.TRAIN.get('DIST_OFFSET', 0.0), requires_grad=False)

    def forward(self, x, d_vecs=None, vel_norm=None, vel=None, return_vel=False):
        vecs = []
        vecs.append(self.proj_joints(x['joints'])) # [bs, jdim]
        vecs.append(self.proj_traj(x['traj']))
        if self.config.TRAIN.USE_TARGET:
            vecs.append(self.proj_tgt(x['tgt']))
        if self.config.TRAIN.USE_VOX:
            vecs.append(self.proj_vox(x['vox']))
        if self.config.TRAIN.get('USE_HAND', False):
            vecs.append(self.proj_hand(x['hand']))
        x = torch.stack(vecs, dim=1) # [bs, ctrl, 512]

        x = self.transf_blocks(x)
        x = self.to_vec_block(x) # [bs, 512]
        if self.RANDOM:
            x = torch.cat([x, torch.randn_like(x)], dim=1)
        x = self.out_proj(x)

        if d_vecs is not None:
            d_vecs = d_vecs.detach()
            future_kf = self.config.TRAIN.FUTURE_KF
            # assert future_kf == 1
            vel_mean, vel_std = vel_norm
            # v = vel.clone().detach() # [bs, 3]

            v = x[:, 264:264 + 3].clone() # [bs, 3] TODO detach
            v = denormalize(v, vel_mean, vel_std)
            v[:, 2] = 0.0
            delta_v = self.alpha * get_delta_v(d_vecs, v, self.config.TRAIN.DIST_OFFSET) # [bs, 3]
            # max(dv) in v direction = projection of v onto dv
            if self.config.TRAIN.get('CLIP_DV', False):
                max_dv    = -project(v, delta_v[:, None]).squeeze(1)
                norms_dv  = torch.linalg.vector_norm(delta_v+1e-7, dim=-1, keepdim=True)
                norms_mdv = torch.linalg.vector_norm(max_dv, dim=-1, keepdim=True)
                mask = torch.logical_or(norms_dv < norms_mdv, norms_dv < 1e-5)
                delta_v = torch.where(mask, delta_v, delta_v / norms_dv * norms_mdv * self.config.TRAIN.CLIP_COEFF)
                if self.config.TRAIN.get('USE_FLOSS', False):
                    norms_v = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
                    ratio = norms_dv / norms_v
                    mask = norms_v < 1e-3
                    ratio[mask] = 0.0
                    floss = ratio**2
                    floss = floss.mean() # Calculate floss in the model for faster training
                if self.config.TRAIN.get('USE_VLOSS', False):
                    if 'norms_v' not in locals():
                        norms_v = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
                    vloss = norms_v**self.config.TRAIN.get('VLOSS_POW', 2)
                    vloss = vloss.mean()
                # if self.config.TRAIN.get('DV_LOG', False):
                #     new_norms = torch.log(norms_dv + 1.0)
                #     # delta_v = delta_v * new_norms / norms_dv 
                #     delta_v = torch.where(norms_dv < 1e-6, delta_v, delta_v * new_norms / norms_dv)
                # else:
                #     mask = norms_dv < self.config.TRAIN.DV_MAX
                #     delta_v = torch.where(mask, delta_v, delta_v / norms_dv * self.config.TRAIN.DV_MAX)
                # delta_v = (self.config.TRAIN.DV_MAX / norms_dv)[:, None] * delta_v

            # delta_v = delta_v / vel_std

            # assert not torch.isnan(x).any()
            nxt_traj = x[:, 264:264+3].view(-1, 1, 3)
            if self.config.TRAIN.get('NO_VEL', False):
                out_traj_vel = nxt_traj
            else:
                nxt_vel = x[:, 264+future_kf*3:264+future_kf*3+3].view(-1, 1, 3)
                out_traj_vel = torch.cat([nxt_traj, nxt_vel], dim=1)
            # out_traj_vel = x[:, 264:264+6].view(-1, 2, 3)
            out_traj_vel = denormalize(out_traj_vel, vel_mean, vel_std)
            # norms = torch.linalg.vector_norm(out_traj_vel, dim=-1, keepdim=True)
            # new_norms = torch.log(norms + 1.0)
            # out_traj_vel = torch.where(norms < 1e-6, out_traj_vel, out_traj_vel * new_norms / norms)

            if self.config.TRAIN.get('CLIP_VEL', False):
                norms = torch.linalg.vector_norm(out_traj_vel, dim=-1, keepdim=True)
                mask = norms < self.config.TRAIN.VEL_MAX
                out_traj_vel = torch.where(mask, out_traj_vel, out_traj_vel / norms * self.config.TRAIN.VEL_MAX)
            # out_limb_traj_vel = x[:, 270+4:270+4+8*3].view(-1, 8, 3)
            dv = delta_v.view(-1, 1, 3)
            out_traj_vel_ = normalize(out_traj_vel + dv, vel_mean, vel_std)
            # out_limb_traj_vel = out_limb_traj_vel + dv
            # x[:, 264:264+6] = out_traj_vel_.view(-1, 6)
            x[:, 264:264+3] = out_traj_vel_[:, 0]
            if not self.config.TRAIN.get('NO_VEL', False):
                x[:, 264+future_kf*3:264+future_kf*3+3] = out_traj_vel_[:, 1]
            # x[:, 270+4:270+4+8*3] = out_limb_traj_vel.view(-1, 8*3)

        # assert no nan in x
        # assert not torch.isnan(x).any()
        if not self.config.TRAIN.get('USE_FIELD', False):
            return x
        return_dict = {'pred': x}
        if return_vel:
            return_dict.update({'vel': out_traj_vel[:, 0], 'dv': delta_v})
        if self.config.TRAIN.get('USE_FLOSS', False):
            return_dict['floss'] = floss
        if self.config.TRAIN.get('USE_VLOSS', False):
            return_dict['vloss'] = vloss
        return return_dict
        # return x, out_traj_vel[:, 0], delta_v
    

import torch

from body_model.body_model import BodyModel
from body_model.lbs import batch_rigid_transform, batch_rodrigues


@torch.no_grad()
def get_template_joints(body_model: BodyModel):
    """Ego center at origin."""
    v_template = body_model.init_v_template # [1, 6890, 3]
    J_regressor = body_model.J_regressor[:22] # [22, 6890]
    J = torch.matmul(J_regressor, v_template.squeeze(0)) # [22, 3]
    center_offset = J[0].clone()
    joints = J - center_offset
    return joints, center_offset

def smpl_fk(J, rotations, parents):
    """
    J: [22, 3]
    rotations: [N, 22, 3]
    parents: [22]
    return: [N, 22, 3]
    """
    rot_mats = batch_rodrigues(rotations.view(-1, 3)).view(*rotations.size(), 3)
    J_transformed, A = batch_rigid_transform(rot_mats, J.expand(rot_mats.size(0), -1, -1), parents)
    return J_transformed

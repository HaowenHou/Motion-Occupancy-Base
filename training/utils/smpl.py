import torch


def smpl_forward(model, orient=None, bpose=None, poses=None, betas=None, trans=None, rm_offset=False, offset=None, return_verts=True, pad_bpose=False):
    """
    Input is tensors.
    orient: poses = [orient, bpose], shaped as [N, dim], instead of [N, J, 3].
    rm_offset is also applied on vertices.
    """
    if poses is not None:
        orient = poses[:, :3]
        bpose = poses[:, 3:72] # smplh needs 22 while smpl needs 24.
    if bpose is not None:
        bpose = bpose[:, :69]
        if pad_bpose and bpose.shape[1] < 69: # Pad zeros.
            bpose = torch.cat([bpose, torch.zeros_like(bpose[:, :1]).expand(-1, 69-bpose.shape[1])], dim=1)
    smpl_output = model(betas=betas, global_orient=orient, body_pose=bpose, transl=trans, return_verts=return_verts, dense_verts=return_verts, return_full_pose=True)
    joints = smpl_output.joints
    vertices = smpl_output.vertices
    if rm_offset:
        offset = offset if offset is not None else \
            smpl_forward(model, betas=betas[:1] if betas is not None else None)['joints'][0, 0] # [3]
        # To validate whether the offset calculation is correct
        # offset1 = smpl_output.joints[:, 0] # [N, 3]
        # if trans is not None:
        #     offset1 = offset1 - trans
        #     assert all(torch.allclose(offset1[0], row) for row in offset1)
        #     assert all(torch.allclose(offset1[0], offset))
        joints = joints - offset
        if vertices is not None:
            vertices = vertices - offset
    return_dict = {
        'joints': joints[:, :22],
        'vertices': vertices,
    }
    return return_dict
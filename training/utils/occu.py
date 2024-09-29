import torch
import numpy as np
import trimesh
import pytorch3d
import torch.nn.functional as F
from pytorch3d.loss.point_mesh_distance import point_face_distance, point_edge_distance
from pytorch3d.structures import Pointclouds
from trimesh.intersections import slice_mesh_plane
from dataset import ELIMBS
from utils.quaternion import qbetween, qrot, rad_to_quat
from utils.smpl import smpl_forward
# from leap.tools.libmesh import check_mesh_contains

def calc_sdf(verts, sdf_dict, return_gradient=False):
    if 'centroid' in sdf_dict:
        sdf_centroid = sdf_dict['centroid']
        sdf_scale = sdf_dict['scale']
        sdf_grids = sdf_dict['grid']


        batch_size, num_vertices, _ = verts.shape
        verts    = verts.reshape(1, -1, 3)  # [B, V, 3]
        vertices = (verts - sdf_centroid) / sdf_scale  # convert to [-1, 1]
        sdf_values = F.grid_sample(sdf_grids,
                                    vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                    padding_mode='border',
                                    align_corners=True
                                    # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                    ).reshape(batch_size, num_vertices)
        if return_gradient:
            sdf_gradients = sdf_dict['gradient_grid']
            gradient_values = F.grid_sample(sdf_gradients,
                                    vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3),
                                    # [2,1,0] permute because of grid_sample assumes different dimension order, see below
                                    padding_mode='border',
                                    align_corners=True
                                    # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                    ).permute(2, 1, 0, 3, 4).reshape(batch_size, num_vertices, 3)
            gradient_values = gradient_values / torch.norm(gradient_values, dim=-1, keepdim=True).clip(min=1e-12)
            return sdf_values, gradient_values
        '''
        # illustration of grid_sample dimension order, assume first dimension to be innermost
        # import torch
        # import torch.nn.functional as F
        # import numpy as np
        # sz = 5
        # input_arr = torch.from_numpy(np.arange(sz * sz).reshape(1, 1, sz, sz)).float()
        # indices = torch.from_numpy(np.array([-1, -1, -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1, -1, 0.5, 0.5, -1]).reshape(1, 1, 7, 2)).float()
        # out = F.grid_sample(input_arr, indices, align_corners=True)
        # print(input_arr)
        # print(out)
        '''
        return sdf_values
    elif 'grid_min' in sdf_dict:
        sdf_grids = sdf_dict['sdf_torch'].to(verts.device)
        sdf_max = torch.tensor(sdf_dict['grid_max']).reshape(1, 1, 3).to(verts.device)
        sdf_min = torch.tensor(sdf_dict['grid_min']).reshape(1, 1, 3).to(verts.device)

        # vertices = torch.tensor(vertices).reshape(1, -1, 3)
        batch_size, num_vertices, _ = verts.shape
        verts = ((verts - sdf_min) / (sdf_max - sdf_min) * 2 - 1)
        sdf_values = F.grid_sample(sdf_grids,
            verts[:, :, [2, 1, 0]].view(-1, num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
            padding_mode='border',
            align_corners=True
            # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
            )
        return sdf_values.reshape(batch_size, num_vertices)


# borrowed from pytorch3d.loss.point_mesh_distance
def point_mesh_face_distance(meshes, pcls, min_triangle_area=0.005,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    edges_packed = meshes.edges_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    segms = verts_packed[edges_packed]  # (S, 2, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    segms_first_idx = meshes.mesh_to_edges_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )
    # point to edge distance: shape (P,)
    point_to_edge = point_edge_distance(
        points, points_first_idx, segms, segms_first_idx, max_points
    )
    
    return torch.minimum(point_to_face, point_to_edge)


def get_bps(bps, unit, size):
    bps = bps * (size[0] - 1) / 2
    bps[..., 1] += (size[0] - 1) / 4.
    return bps * unit
    
def query_bps(occu_g, llb, occu_g_ref, occu_shape, crt_pos, crt_rad, bps, unit, device, poses=None, model=None):
    crt_pos, crt_rad = crt_pos.to(device), crt_rad.to(device)
    
    g2l_quat = rad_to_quat(crt_rad)
    basis    = qrot(g2l_quat.expand(bps.shape[0], -1), bps)
    basis    = basis + crt_pos - llb
    
    coord = basis // unit
    coord = coord.long() # B, P, 3
    coord[..., 0] = torch.clamp(coord[..., 0], min=0, max=occu_shape[0] - 1)
    coord[..., 1] = torch.clamp(coord[..., 1], min=0, max=occu_shape[1] - 1)
    coord[..., 2] = torch.clamp(coord[..., 2], min=0, max=occu_shape[2] - 1)
    
    gidx = torch.arange(coord.shape[0], device=device)
    ref  = occu_g_ref[coord[gidx, 0], coord[gidx, 1], coord[gidx, 2]]
    occu_l = torch.linalg.vector_norm(basis - (ref * unit + 0.5 * unit), dim=-1)
    
    return occu_l
    
def query_bps_on_mesh_field(crt_pos, crt_rad, bps, unit, device, scene=None, triscene=None, sdf_dict=None):
    bs, p  = len(crt_pos), len(bps)
    crt_pos, crt_rad = crt_pos.to(device), crt_rad.to(device)
    ori_basis = bps[None].expand(bs, -1, -1) # B, P, 3
    
    g2l_quat = rad_to_quat(crt_rad)
    basis    = qrot(g2l_quat[:, None].expand(-1, p, -1), ori_basis)
    basis    = basis + crt_pos[:, None]
    
    if sdf_dict is not None:
        sdf    = calc_sdf(basis, sdf_dict)
        occu_l = torch.abs(sdf)
        d_vecs = torch.zeros_like(ori_basis)
        bidx, gidx = torch.where(sdf <= 0)
        d_vecs[bidx, gidx] = ori_basis[bidx, gidx]
        # Ignore low and high voxels to reduce data noise
        bid, oid = torch.where(ori_basis[..., 2] < 0.1)
        d_vecs[bid, oid] *= 0
        bid, oid = torch.where(ori_basis[..., 2] > 1.6)
        d_vecs[bid, oid] *= 0
    elif scene is not None and triscene is not None:
        ps       = Pointclouds(basis.flatten(0, 1)[None])
        occu_l   = point_mesh_face_distance(scene, ps).reshape(bs, p)
        
        
        d_vecs   = torch.zeros_like(ori_basis)
        basis_np       = basis.flatten(0, 1).cpu().numpy()
        ray_origins    = crt_pos[:, None].expand(-1, p, -1).flatten(0, 1).cpu().numpy()
        bps_loc        = np.linalg.norm(basis_np - ray_origins, axis=-1)
        ray_directions = (basis_np - ray_origins) / bps_loc[:, None]
        limit_max = np.max(basis_np, axis=0)
        limit_min = np.min(basis_np, axis=0)
        norm_min  = limit_max - limit_min
        norm_max  = limit_min - limit_max
        tmpscene  = slice_mesh_plane(triscene, [norm_min[0], 0, 0], limit_min)
        tmpscene  = slice_mesh_plane(tmpscene, [0, norm_min[1], 0], limit_min)
        tmpscene  = slice_mesh_plane(tmpscene, [0, 0, norm_min[2]], limit_min)
        tmpscene  = slice_mesh_plane(tmpscene, [norm_max[0], 0, 0], limit_max)
        tmpscene  = slice_mesh_plane(tmpscene, [0, norm_max[1], 0], limit_max)
        tmpscene  = slice_mesh_plane(tmpscene, [0, 0, norm_max[2]], limit_max)
        if len(tmpscene.faces) > 0:
            locations, index_ray, _ = tmpscene.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False,)
            inter_loc      = bps_loc + 1
            inter_loc[index_ray] = np.linalg.norm(locations - ray_origins[index_ray], axis=-1)
            occu_idx = np.where(inter_loc < bps_loc)[0]
            bidx = torch.arange(bs, device=device)[:, None].expand(-1, p).flatten()[occu_idx]
            gidx = torch.arange(p, device=device)[None, :].expand(bs, -1).flatten()[occu_idx]
            
            d_vecs[bidx, gidx] = ori_basis[bidx, gidx]
            # Ignore low and high voxels to reduce data noise
            bid, oid = torch.where(ori_basis[..., 2] < 0.05)
            d_vecs[bid, oid] *= 0
            bid, oid = torch.where(ori_basis[..., 2] > 1.6)
            d_vecs[bid, oid] *= 0
    else:
        return torch.zeros(bs, p, device=device), torch.zeros_like(ori_basis)
    
    return occu_l, d_vecs
    
def query_bps_on_mesh(crt_pos, crt_rad, bps, unit, device, scene=None, sdf_dict=None):
    bs, p  = len(crt_pos), len(bps)
    crt_pos, crt_rad = crt_pos.to(device), crt_rad.to(device)
    ori_basis = bps[None].expand(bs, -1, -1) # B, P, 3
    
    g2l_quat = rad_to_quat(crt_rad)
    basis    = qrot(g2l_quat[:, None].expand(-1, p, -1), ori_basis)
    basis    = basis + crt_pos[:, None]
    
    
    if sdf_dict is not None:
        sdf    = calc_sdf(basis, sdf_dict)
        occu_l = torch.abs(sdf)
    elif scene is not None:
        ps     = Pointclouds(basis.flatten(0, 1)[None])
        occu_l = point_mesh_face_distance(scene, ps).reshape(bs, p)
    else:
        occu_l = torch.zeros(bs, p, device=device)
    
    return occu_l
    
def query_bps_batched(occu_g, llb, occu_g_ref, occu_shape, crt_pos, crt_rad, bps, unit, device, poses=None, model=None):
    crt_pos, crt_rad = crt_pos.to(device), crt_rad.to(device)
    bs  = len(occu_g)
    basis = bps[None].expand(bs, -1, -1) # B, P, 3
    
    g2l_quat = rad_to_quat(crt_rad)
    basis    = qrot(g2l_quat[:, None].expand(-1, basis.shape[1], -1), basis)
    basis    = basis + crt_pos[:, None] - llb.view(bs, 1, 3)
    
    coord = basis // unit
    coord = coord.long() # B, P, 3
    coord = torch.clamp(coord, min=0)
    coord[..., 0] = torch.clamp(coord[..., 0], max=occu_shape[:, None, 0] - 1)
    coord[..., 1] = torch.clamp(coord[..., 1], max=occu_shape[:, None, 1] - 1)
    coord[..., 2] = torch.clamp(coord[..., 2], max=occu_shape[:, None, 2] - 1)
    
    bidx = torch.arange(bs, device=device)[:, None].expand(-1, coord.shape[1]).flatten()
    gidx = torch.arange(coord.shape[1], device=device)[None, :].expand(bs, -1).flatten()
    ref  = occu_g_ref[bidx, coord[bidx, gidx, 0], coord[bidx, gidx, 1], coord[bidx, gidx, 2]].reshape(bs, coord.shape[1], 3)
    occu_l = torch.linalg.vector_norm(basis - (ref * unit + 0.5 * unit), dim=-1)
    
    return occu_l

    
def query_bps_batched_field(occu_g, llb, occu_g_ref, occu_shape, crt_pos, crt_rad, bps, unit, device):
    crt_pos, crt_rad = crt_pos.to(device), crt_rad.to(device)
    bs  = len(occu_g)
    ori_basis = bps[None].expand(bs, -1, -1) # B, P, 3
    
    g2l_quat = rad_to_quat(crt_rad)
    basis    = qrot(g2l_quat[:, None].expand(-1, ori_basis.shape[1], -1), ori_basis)
    basis    = basis + crt_pos[:, None] - llb.view(bs, 1, 3)
    
    coord = basis // unit
    coord = coord.long() # B, P, 3
    coord = torch.clamp(coord, min=0)
    coord[..., 0] = torch.clamp(coord[..., 0], max=occu_shape[:, None, 0] - 1)
    coord[..., 1] = torch.clamp(coord[..., 1], max=occu_shape[:, None, 1] - 1)
    coord[..., 2] = torch.clamp(coord[..., 2], max=occu_shape[:, None, 2] - 1)
    
    bidx = torch.arange(bs, device=device)[:, None].expand(-1, coord.shape[1]).flatten()
    gidx = torch.arange(coord.shape[1], device=device)[None, :].expand(bs, -1).flatten()
    ref  = occu_g_ref[bidx, coord[bidx, gidx, 0], coord[bidx, gidx, 1], coord[bidx, gidx, 2]].reshape(bs, coord.shape[1], 3)
    occu_l = torch.linalg.vector_norm(basis - (ref * unit + 0.5 * unit), dim=-1)
    
    # Ignore un-occupied voxels
    occu_b = torch.zeros(bs, basis.shape[1], device=device)
    occu_b[bidx, gidx] = occu_g[bidx, coord[bidx, gidx, 0], coord[bidx, gidx, 1], coord[bidx, gidx, 2]]
    bid, oid = torch.where(occu_b > 0)
    d_vecs   = torch.zeros_like(ori_basis) # [bs, 15625, 3]
    d_vecs[bid, oid] = ori_basis[bid, oid]
    d_vecs[..., 2]   *= 0
    # Ignore low and high voxels to reduce data noise
    bid, oid = torch.where(ori_basis[..., 2] < 0.1)
    d_vecs[bid, oid] *= 0
    bid, oid = torch.where(ori_basis[..., 2] > 1.6)
    d_vecs[bid, oid] *= 0
    return occu_l, d_vecs


def get_grid(unit, size, device=None, offset_y=True):
    unit = float(unit)
    grid = create_meshgrid3d(size[0], size[1], size[2])[0] - (torch.Tensor(size)-1) / 2.0
    if offset_y:
        grid[..., 1] += (size[1] - 1) / 4.0
    oidx = create_meshgrid3d(size[0], size[1], size[2])[0].view(-1, 3).long()
    grid = grid * unit
    occu_l = torch.zeros(size[0], size[1], size[2]).float()
    if device is not None:
        grid = grid.to(device)
        oidx = oidx.to(device)
        occu_l = occu_l.to(device)
    return grid.view(-1, 3), oidx, occu_l

def query_occu(occu_g, llb, crt_pos, crt_rad, unit, grid_size, device, poses=None, model=None):
    # voxel, unit, llb = calc_global_occu(model_male, trans, poses=poses)
    grid, oidx, occu_l = get_grid(unit=unit, size=grid_size)
    grid, oidx, occu_l = grid.to(device), oidx.to(device), occu_l.to(device)

    g2l_quat = rad_to_quat(crt_rad)
    grid = qrot(g2l_quat.expand(len(grid), -1), grid)
    grid = grid + crt_pos - llb # In llb space.

    grid = grid // unit
    grid = grid.long()
    x_mask = torch.logical_and(grid[:, 0] >= 0, grid[:, 0] < occu_g.shape[0])
    y_mask = torch.logical_and(grid[:, 1] >= 0, grid[:, 1] < occu_g.shape[1])
    z_mask = torch.logical_and(grid[:, 2] >= 0, grid[:, 2] < occu_g.shape[2])
    mask = torch.logical_and(torch.logical_and(x_mask, y_mask), z_mask)
    gidx  = torch.where(mask)[0]
    occu_l[oidx[gidx, 0], oidx[gidx, 1], oidx[gidx, 2]] = occu_g[grid[gidx, 0], grid[gidx, 1], grid[gidx, 2]]
    gidx  = torch.where(~mask)[0]
    occu_l[oidx[gidx, 0], oidx[gidx, 1], oidx[gidx, 2]] = 1.
    return occu_l

def query_occu_batched(occu_g, llb, crt_pos, crt_rad, unit, grid_size, device, grid=None, oidx=None, occu_l=None):
    '''
        Get occupancy volume in egocentric coordinate
    '''
    # voxel, unit, llb = calc_global_occu(model_male, trans, poses=poses)
    crt_pos, crt_rad = crt_pos.to(device), crt_rad.to(device)
    bs = len(occu_g)
    if grid is None:
        grid, oidx, occu_l = get_grid(unit=unit, size=grid_size)
        grid, oidx, occu_l = grid.to(device), oidx.to(device), occu_l.to(device)
        grid = grid[None].expand(bs, -1, -1)
        oidx = oidx[None].expand(bs, -1, -1)
        occu_l = occu_l[None].expand(bs, -1, -1, -1)
    else:
        grid, oidx, occu_l = grid.clone(), oidx.clone(), occu_l.clone()

    g2l_quat = rad_to_quat(crt_rad)
    grid = qrot(g2l_quat[:, None].expand(-1, grid.shape[1], -1), grid)
    grid = grid + crt_pos[:, None] - llb.view(bs, 1, 3) # In llb space.

    grid = grid // unit
    grid = grid.long()
    x_mask = torch.logical_and(grid[..., 0] >= 0, grid[..., 0] < occu_g.shape[-3])
    y_mask = torch.logical_and(grid[..., 1] >= 0, grid[..., 1] < occu_g.shape[-2])
    z_mask = torch.logical_and(grid[..., 2] >= 0, grid[..., 2] < occu_g.shape[-1])
    mask = torch.logical_and(torch.logical_and(x_mask, y_mask), z_mask)
    bidx, gidx = torch.where(mask)
    occu_l[bidx, oidx[bidx, gidx, 0], oidx[bidx, gidx, 1], oidx[bidx, gidx, 2]] = occu_g[bidx, grid[bidx, gidx, 0], grid[bidx, gidx, 1], grid[bidx, gidx, 2]]
    bidx, gidx = torch.where(~mask)
    occu_l[bidx, oidx[bidx, gidx, 0], oidx[bidx, gidx, 1], oidx[bidx, gidx, 2]] = 1.
    return occu_l

def occug_to_occul(grid, occul, occu_g, unit):
    grid = grid // unit
    grid = grid.long()
    x_mask = torch.logical_and(grid[..., 0] >= 0, grid[..., 0] < occu_g.shape[-3])
    y_mask = torch.logical_and(grid[..., 1] >= 0, grid[..., 1] < occu_g.shape[-2])
    z_mask = torch.logical_and(grid[..., 2] >= 0, grid[..., 2] < occu_g.shape[-1])
    mask = torch.logical_and(torch.logical_and(x_mask, y_mask), z_mask)
    occul = occul.flatten(1) # [bs, g^3]
    bidx, gidx = torch.where(mask)
    occul[bidx, gidx] = occu_g[bidx, grid[bidx, gidx, 0], grid[bidx, gidx, 1], grid[bidx, gidx, 2]]
    occul[~mask] = 1.0
    return occul

def query_occu_batched_field(occu_g, llb, crt_pos, crt_rad, unit, grid_size, device, 
                             grid=None, oidx=None, occu_l=None,
                             jego=None, config=None, tgt_limb_abs=None):
    '''
        Get occupancy volume in egocentric coordinate
    '''
    REAL_FIELD = config is not None and config.TRAIN.get('REAL_FIELD', False)
    DROP_VOX = config is not None and config.STAGE == 'TRAIN' and config.TRAIN.get('DROP_VOX', False)
    CLOSE_SW = config is not None and config.TRAIN.get('CLOSE_SW', False)
    crt_pos, crt_rad = crt_pos.to(device), crt_rad.to(device)
    bs = len(occu_g)
    if grid is None:
        grid_t, oidx, occul_t = get_grid(unit=unit, size=grid_size, device=device)
        grid_t = grid_t[None].expand(bs, -1, -1).contiguous()
        oidx = oidx[None].expand(bs, -1, -1).contiguous()
        occul_t = occul_t[None].expand(bs, -1, -1, -1).contiguous()
    else:
        grid_t, oidx, occul_t = grid, oidx, occu_l

    # Calculate canonical occupancy
    g2l_quat = rad_to_quat(crt_rad)
    grid_roted = qrot(g2l_quat[:, None].expand(-1, grid_t.shape[1], -1), grid_t)
    grid = grid_roted + crt_pos[:, None]
    occul = occug_to_occul(grid - llb.view(bs, 1, 3), occul_t.clone(), occu_g, unit)

    if REAL_FIELD:
        # Calculate global feet pos
        feet_pos = jego[:, ELIMBS[1:3]] # [bs, 2, 3]
        feet_pos = qrot(g2l_quat[:, None].expand(-1, 2, -1), feet_pos)
        crt_pos_0h = crt_pos.clone()
        crt_pos_0h[..., 2] = 0.0
        feet_pos = feet_pos + crt_pos_0h[:, None]
        center = torch.mean(feet_pos, dim=1) # [bs, 3]
        if CLOSE_SW:
            tgt_center = torch.mean(tgt_limb_abs[:, [1, 2]], dim=1) # [bs, 3]
            center[..., 2] = 0.0
            tgt_center[..., 2] = 0.0
            center_dist = torch.linalg.vector_norm(center - tgt_center, dim=-1) # [bs]
            close = center_dist < 0.5 # [bs]
        higher_foot = torch.max(feet_pos[..., 2], dim=-1)[0] # [bs]
        center[..., 2] = higher_foot
        lb = torch.clamp(higher_foot, min=0.1)
        ub = lb + 0.9
        # fgrid_t, _, _ = get_grid(unit=unit, size=grid_size, device=device, offset_y=False)
        # fgrid_t = fgrid_t.expand(bs, -1, -1).contiguous()
        # fgrid_roted = qrot(g2l_quat[:, None].expand(-1, fgrid_t.shape[1], -1), fgrid_t)
        # fgrid = fgrid_roted + center[:, None]
        fgrid = grid_roted + center[:, None]
        masked = (fgrid[..., 2] < lb[:, None]) | (fgrid[..., 2] > ub[:, None])
        field_occul = occug_to_occul(fgrid - llb.view(bs, 1, 3), occul_t.clone(), occu_g, unit)
        field_occul[masked] = 0.0

        d_vecs = torch.zeros_like(grid_t) # [bs, 15625, 3]
        bid, oid = torch.where(field_occul == 1.0)
        # d_vecs[bid, oid] = fgrid_t[bid, oid].clone()
        d_vecs[bid, oid] = grid_t[bid, oid].clone()
        d_vecs[..., 2] = 0.0
        if CLOSE_SW:
            d_vecs[close] = 0.0
    else:
        d_vecs = torch.zeros_like(grid_t)
        bid, oid = torch.where(occul == 1.0)
        d_vecs[bid, oid] = grid_t[bid, oid].clone()
        d_vecs[..., 2] = 0.0
        # Ignore low and high voxels to reduce data noise
        bid, oid = torch.where((grid[..., 2] < 0.1  // unit) | (grid[..., 2] >= 1.6 // unit))
        d_vecs[bid, oid] = 0.0

    # Dropout. feet and above.
    if DROP_VOX:
        feet_pos = jego[:, ELIMBS[1:3]]
        lb = torch.max(feet_pos[..., 2], dim=-1)[0] # [bs]
        ub = torch.max(jego[..., 2], dim=-1)[0] # [bs]
        bound = torch.rand_like(lb) * (ub - lb) + lb
        vox_drop = grid[..., 2] > bound[:, None]
        occul[vox_drop] = 0.0

    return_dict = {'occul': occul, 'd_vecs': d_vecs}
    if REAL_FIELD and CLOSE_SW:
        return_dict['close'] = (~close)[:, None].float() # [bs, 1], 0: close
    return return_dict

def query_occu_batched_field_mesh(crt_pos, crt_rad, unit, grid_size, device, grid=None, oidx=None, occul=None,
                                  scene=None, triscene=None, sdf_dict=None, fill=None, limit_min=None, limit_max=None,
                                  jego=None, config=None, tgt_limb_abs=None):
    '''
        Get occupancy volume in egocentric coordinate
    '''
    REAL_FIELD = config is not None and config.TRAIN.get('REAL_FIELD', False)
    CLOSE_SW = config is not None and config.TRAIN.get('CLOSE_SW', False)
    crt_pos, crt_rad = crt_pos.to(device), crt_rad.to(device)
    bs = len(crt_pos)
    if grid is None:
        grid_t, oidx, occul_t = get_grid(unit=unit, size=grid_size, device=device)
        grid_t = grid_t[None].expand(bs, -1, -1).contiguous()
        oidx = oidx[None].expand(bs, -1, -1).contiguous()
        occul_t = occul_t[None].expand(bs, -1, -1, -1).contiguous()
    else:
        grid_t, oidx, occul_t = grid, oidx, occul

    g2l_quat = rad_to_quat(crt_rad)
    grid_roted = qrot(g2l_quat[:, None].expand(-1, grid_t.shape[1], -1), grid_t)
    grid = grid_roted + crt_pos[:, None]

    if sdf_dict is not None:
        sdf        = calc_sdf(grid, sdf_dict)
        occul   = sdf <= 0
        if limit_max is not None:
            occul = torch.logical_or(occul, grid[..., 0] > limit_max[0])
            occul = torch.logical_or(occul, grid[..., 1] > limit_max[1])
            occul = torch.logical_or(occul, grid[..., 0] < limit_min[0])
            occul = torch.logical_or(occul, grid[..., 1] < limit_min[1])
        occul_wofill = occul.clone()
        # randomly add obstacle as 3*3*z boxes
        fill = None
        if fill is not None:
            occul = occul.float().view_as(occul_t) # [bs, 25, 25, 25]
            bidx   = torch.arange(bs)
            occu_map    = torch.sum(occul[..., 2:] <= 0, dim=-1)
            occu_map[:, 9:18] = True
            occu_map[:, :, 5:14] = True
            occu_rate   = occul.mean((1, 2, 3))
            while torch.any(occu_rate < fill):
                fcoord   = torch.randint(1, occul.shape[1] - 1, (bs, 2), device=device)  # locations to fill
                fmask_c  = ~occu_map[bidx, fcoord[bidx, 0], fcoord[bidx, 1]] # indexes can fill
                fmask_n  = occu_rate < fill                                  # indexes need to fill
                fmask    = torch.logical_and(fmask_n, fmask_c)
                if torch.any(fmask):
                    fidx,    = torch.where(fmask)
                    coin     = np.random.rand()
                    occul[fidx, fcoord[fidx, 0] - 1:fcoord[fidx, 0] + 2, fcoord[fidx, 1] - 1:fcoord[fidx, 1] + 2, :11] = 1.0
                    occul[fidx, fcoord[fidx, 0] - 1:fcoord[fidx, 0] + 2, fcoord[fidx, 1] - 1:fcoord[fidx, 1] + 2, 20:] = 1.0
                    if coin < 0.5:
                        occul[fidx, fcoord[fidx, 0] - 1:fcoord[fidx, 0] + 2, fcoord[fidx, 1] - 1:fcoord[fidx, 1] + 2] = 1.0
                    else:
                        occul[fidx, fcoord[fidx, 0] - 1:fcoord[fidx, 0] + 2, fcoord[fidx, 1] - 1:fcoord[fidx, 1] + 2, :11] = 1.0
                        occul[fidx, fcoord[fidx, 0] - 1:fcoord[fidx, 0] + 2, fcoord[fidx, 1] - 1:fcoord[fidx, 1] + 2, 20:] = 1.0
                    occu_rate   = occul.mean((1, 2, 3))
            occul = occul.view(bs, -1)
        # Save canonical occupancy for double check
        occul_grid = grid.clone()
        occul_grid[~occul.bool()] = 0.0
        occul   = occul.float()
        
    if REAL_FIELD:
        # Calculate global feet pos
        feet_pos = jego[:, ELIMBS[1:3]]
        feet_pos = qrot(g2l_quat[:, None].expand(-1, 2, -1), feet_pos)
        crt_pos_0h = crt_pos.clone()
        crt_pos_0h[..., 2] = 0.0
        feet_pos = feet_pos + crt_pos_0h[:, None]
        center = torch.mean(feet_pos, dim=1) # [bs, 3]
        if CLOSE_SW:
            tgt_center = torch.mean(tgt_limb_abs[:, [1, 2]], dim=1) # [bs, 3]
            center[..., 2] = 0.0
            tgt_center[..., 2] = 0.0
            center_dist = torch.linalg.vector_norm(center - tgt_center, dim=-1) # [bs]
            close = center_dist < 0.5 # [bs]
        higher_foot = torch.max(feet_pos[..., 2], dim=-1)[0] # [bs]
        center[..., 2] = higher_foot
        lb = torch.clamp(higher_foot, min=0.1)
        ub = lb + 0.9
        fgrid = grid_roted + center[:, None]
        masked = (fgrid[..., 2] < lb[:, None]) | (fgrid[..., 2] > ub[:, None])
        sdf = calc_sdf(fgrid, sdf_dict)
        field_occul = sdf <= 0
        if limit_max is not None:
            field_occul = torch.logical_or(field_occul, fgrid[..., 0] > limit_max[0])
            field_occul = torch.logical_or(field_occul, fgrid[..., 1] > limit_max[1])
            field_occul = torch.logical_or(field_occul, fgrid[..., 0] < limit_min[0])
            field_occul = torch.logical_or(field_occul, fgrid[..., 1] < limit_min[1])
        field_occul = field_occul.float()
        field_occul[masked] = 0.0
        d_vecs = torch.zeros_like(grid_t)
        bid, oid = torch.where(field_occul == 1.0)
        d_vecs[bid, oid] = grid_t[bid, oid].clone()
        d_vecs[..., 2] = 0.0
        if CLOSE_SW:
            d_vecs[close] = 0.0
    else:
        d_vecs = torch.zeros_like(grid_t)
        bidx, gidx = torch.where(occul_wofill == 1.0)
        d_vecs[bidx, gidx] = grid_t[bidx, gidx]
        d_vecs[..., 2] = 0.0
        # Ignore low and high voxels to reduce data noise
        bid, oid = torch.where((grid[..., 2] < 0.1  // unit) | (grid[..., 2] >= 1.6 // unit))
        d_vecs[bid, oid] = 0.0

    return_dict = {'occul': occul, 'd_vecs': d_vecs, 'occul_grid': occul_grid}
    if REAL_FIELD and CLOSE_SW:
        return_dict['close'] = (~close)[:, None].float() # [bs, 1], 0: close
    return return_dict

def project(vdir, vecs, ignore_neg=False):
    """
    Calculate the projections of vdir on vecs.
    velocity direction (vdir): [bs, 3]
    vecs: [bs, n, 3]
    """
    # Calculate dot product along the last dimension
    dot_products = torch.sum(vdir[:, None, :] * vecs, dim=-1)
    # Calculate the magnitude squared of vecs
    norms_squared = torch.sum(vecs**2, dim=-1)
    # Calculate the projections
    # Slower
    # projections = torch.zeros_like(vecs)
    epsilon = 1e-7
    mask = norms_squared.abs() > epsilon
    # projections[mask] = dot_products[mask][..., None] * vecs[mask] / (norms_squared[mask][..., None])
    # Faster
    projections = dot_products[..., None] * vecs / (norms_squared[..., None]+1e-7)
    projections[~mask] = 0.0
    # Ignore projections where dot_products is negative
    if ignore_neg:
        projections[dot_products < 0] = 0.0
    return projections

def dist_func(vdir_proj, d_vecs, offset):
    dists = torch.linalg.vector_norm(d_vecs, dim=-1, keepdim=True) # [bs, n]

    delta_vs = torch.zeros_like(vdir_proj)
    mask = (dists > offset).squeeze(-1)
    # delta_vs[mask] = vdir_proj[mask] * 2 * torch.clamp((1 / (dists[mask]-offset+1e-4)**0.82 - 1.1), min=0.0)
    delta_vs[mask] = vdir_proj[mask] * 2 * torch.clamp((1 / (dists[mask]-offset+1e-4)**0.85 - 1.6), min=0.0)
    # delta_vs = vdir_proj * 2 * (1 / (dists-offset+1e-4)**0.85 - 1.6)
    return delta_vs # may be zero

    # dists = torch.where(dists < 0.25, torch.zeros_like(dists), dists)
    # delta_vs = vdir_proj *2* (1 / (dists-0.25+eps)**0.85 - 1.6)
    # # delta_vs = dists * vdir_proj
    # # assert not torch.isnan(delta_vs).any()
    # delta_vs = torch.nan_to_num(delta_vs)
    # return delta_vs

def get_delta_v(d_vecs, v, offset):
    vdir_proj = -project(v, d_vecs, ignore_neg=True) # [bs, n, 3]
    delta_vs = dist_func(vdir_proj, d_vecs, offset)
    delta_v = torch.sum(delta_vs, dim=1) # [bs, 3] before multiplied by alpha
    return delta_v

    # vdir_proj = -project(v, d_vecs, ignore_neg=True) # [bs, n, 3]
    # dists = torch.linalg.vector_norm(d_vecs, dim=-1) # [bs, n]
    # # delta_vs = vdir_proj / (dists[..., None])**2 # [bs, n, 3]
    # # delta_vs = vdir_proj / (dists[..., None]-0.2)**2 # [bs, n, 3]
    # # delta_vs = vdir_proj / (dists[..., None]-offset).abs() # [bs, n, 3]
    # # delta_vs = vdir_proj / torch.max((dists[..., None]-offset), torch.tensor(1e-1)) # [bs, n, 3]
    # delta_vs = vdir_proj * 10 * (-dists[..., None]/2 + 1.0)
    # mask = dists < 1e-7
    # delta_vs[mask] = 0.0
    # delta_v = torch.sum(delta_vs, dim=1) # [bs, 3] before multiplied by alpha
    # return delta_v

def create_meshgrid3d(
        width: int,
        height: int,
        depth: int,
        device=torch.device('cpu'),
        dtype=torch.float32,
    ) -> torch.Tensor:
        """ Generate a coordinate grid in range [-0.5, 0.5].

        Args:
            depth (int): grid dim
            height (int): grid dim
            width (int): grid dim
        Return:
            grid tensor with shape :math:`(1, D, H, W, 3)`.
        """
        xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
        zs = torch.linspace(0, depth - 1, depth, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid([xs, ys, zs]), dim=-1).unsqueeze(0)  # 1xDxHxWx3

def get_global_demo(model, voxel, unit, grid, lb, poses, trans, betas, idx, g2l_quat):
    smpl_output = smpl_forward(model, poses=poses[idx:idx+1, :72], trans=trans[idx:idx+1], rm_offset=True)
    colors = torch.zeros_like(grid)
    h_min = torch.min(grid[:, 2])
    h_max = torch.max(grid[:, 2])
    x_min = torch.min(grid[:, 0])
    x_max = torch.max(grid[:, 0])
    y_min = torch.min(grid[:, 1])
    y_max = torch.max(grid[:, 1])
    colors[:, 0] = (grid[:, 2] - h_min) / (h_max - h_min)
    colors[:, 1] = (grid[:, 0] - x_min) / (x_max - x_min)
    colors[:, 2] = (grid[:, 1] - y_min) / (y_max - y_min)
    colors = colors.numpy()
    g2l_quat = g2l_quat[idx:idx+1].expand(grid.shape[0], -1)
    grid = qrot(g2l_quat, grid)
    grid = grid + trans[idx:idx + 1] - lb # In llb space.
    
    verts = smpl_output['vertices'].detach().numpy()[0] - lb
    mesh = trimesh.Trimesh(vertices=verts, faces=model.faces)
    return [1 - voxel, unit], mesh, [grid.numpy(), colors]

def get_cano_demo(model, voxel, unit, occu, oidx, grid, lb, revidx, poses, trans, betas, idx, g2l_quat, new_orient):
    g2l_quat = g2l_quat[idx:idx+1].expand(grid.shape[0], -1)
    grid = qrot(g2l_quat, grid)
    grid = grid + trans[idx:idx + 1] - lb # In llb space.

    grid = grid // unit
    grid = grid.long()
    x_mask = torch.logical_and(grid[:, 0] >= 0, grid[:, 0] < voxel.shape[0])
    y_mask = torch.logical_and(grid[:, 1] >= 0, grid[:, 1] < voxel.shape[1])
    z_mask = torch.logical_and(grid[:, 2] >= 0, grid[:, 2] < voxel.shape[2])
    mask = torch.logical_and(torch.logical_and(x_mask, y_mask), z_mask)
    gidx  = torch.where(mask)[0]
    occu[oidx[gidx, 0], oidx[gidx, 1], oidx[gidx, 2]] = voxel[grid[gidx, 0], grid[gidx, 1], grid[gidx, 2]]
    gidx  = torch.where(~mask)[0]
    occu[oidx[gidx, 0], oidx[gidx, 1], oidx[gidx, 2]] = 1.
    occu = 1 - occu

    smpl_output = smpl_forward(model, orient=new_orient[idx:idx+1, :3], bpose=poses[idx:idx+1, 3:72], rm_offset=True)
    verts = smpl_output['vertices'].detach().numpy()[0]
    verts[..., 0] += 12 * unit
    verts[..., 1] += 6 * unit
    verts[..., 2] += 12 * unit
    mesh = trimesh.Trimesh(vertices=verts, faces=model.faces)
    return occu, mesh

def get_voxel(voxel, unit, lb, size=[25, 25, 25]):
    voxel = 1 - torch.from_numpy(voxel).float()
    unit = float(unit)
    grid = create_meshgrid3d(size[0], size[1], size[2])[0] - torch.Tensor(size)[None] // 2
    grid[..., 1] += 6
    oidx = create_meshgrid3d(size[0], size[1], size[2])[0].view(-1, 3).long()
    # grid[..., -1] += size[0] / 2. # h, w, d, 3
    grid = grid * unit
    occu = torch.zeros(size[0], size[1], size[2]).float()
    revidx = torch.arange(size[0] * size[1] * size[2]).view(size[0], size[1], size[2])
    return voxel, unit, occu, oidx, grid.view(-1, 3), torch.from_numpy(lb), revidx

def calc_global_occu(model, trans, poses=None, orient=None, bpose=None):
    smpl_output = smpl_forward(model, poses=poses, orient=orient, bpose=bpose, trans=trans, rm_offset=True)
    vertices = smpl_output['vertices'].cpu().detach().numpy()
    # SEQ_DIR = 'vert_seq'
    # os.makedirs(SEQ_DIR, exist_ok=True)
    # for i in range(10):
    #     mesh = trimesh.Trimesh(vertices=vertices[i], faces=model.faces)
    #     mesh.export(f'{SEQ_DIR}/mesh_{i:02}.obj')
    # exit()

    nf, nv   = vertices.shape[:2]
    lb       = np.min(vertices, axis=1)
    ub       = np.max(vertices, axis=1)
    llb      = np.min(lb, axis=0)
    uub      = np.max(ub, axis=0)
    size     = uub - llb
    
    unit = 0.08 
    size = size / unit
    size = size.astype(int) + 1
    grid = model.coap.create_meshgrid3d(size[2], size[1], size[0]).numpy()[0].reshape(-1, 3)
    grid = grid * unit + llb[None]
    ng = grid.shape[0]
    occu = np.zeros(ng)
    oidx = np.arange(ng).reshape(size[0], size[1], size[2])
    
    for i in range(int(nf)):
        vs  = vertices[i]
        tmp = ((vs - llb[None]) // unit).astype(int)
        raw_idx = np.unique(oidx[tmp[:, 0], tmp[:, 1], tmp[:, 2]])
        occu[raw_idx] = 1
        lb_ = lb[i:i + 1]
        ub_ = ub[i:i + 1]
        m1  = np.logical_and(grid >= lb_, grid <= ub_)
        m2  = np.logical_and(np.logical_and(m1[:, 0], m1[:, 1]), m1[:, 2])
        m3  = np.logical_and(m2, occu == 0)
        if np.sum(m2) > 0:
            idx = np.where(m2)[0]
            mesh = trimesh.Trimesh(vs, model.faces)
            occu[idx] = np.maximum(occu[idx], check_mesh_contains(mesh, grid[idx]))
    occu = occu.reshape(size[0], size[1], size[2])
    return occu, float(unit), llb[None]
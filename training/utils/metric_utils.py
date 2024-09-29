import torch


def calculate_distances(limbs, tgt_limbs, INC_LASTS, DIST_THRES):
    """Calculate closest distances between limbs and target limbs.
    Args:
        limbs: [bs, n, 5, 2]
        tgt_limbs: [bs, 5, 2]
    """
    bs, n, _, _ = limbs.shape
    
    tgt_limbs = tgt_limbs.unsqueeze(1).expand(-1, n, -1, -1)
    distances = torch.linalg.vector_norm(limbs - tgt_limbs, dim=-1).sum(-1)
    
    ids = torch.empty(bs, dtype=torch.long).fill_(n-1)
    for i in range(bs):
        count = 0
        for j in range(1, n):
            if distances[i, j] < DIST_THRES and distances[i, j] > distances[i, j-1]:
                count += 1
                if count >= INC_LASTS:
                    ids[i] = j - INC_LASTS
                    break
            else:
                count = 0
    bids = torch.arange(bs)
    dists = distances[bids, ids]
    return dists, ids

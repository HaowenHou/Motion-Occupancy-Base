import os
import pickle
import random


if __name__ == '__main__':
    DATA_DIR = 'datasets/amass_ref'
    SPLIT_DIR = 'datasets/splits/amass_splits/'
    SNIP_DICT_NAME = 'mid_snip_dict.pkl'
    with open(os.path.join(SPLIT_DIR, SNIP_DICT_NAME), 'rb') as f:
        mid_snip_dict_ori = pickle.load(f)
    mid_snip_dict = {k: v for k, v in mid_snip_dict_ori.items() if v}

    # Split data.
    TEST_RATIO = 0.1
    random.seed(0)
    mids = list(mid_snip_dict.keys())
    test_ids = sorted(random.sample(mids, int(len(mids) * TEST_RATIO)))
    train_ids = [mid for mid in mids if mid not in test_ids]

    # Save splits.
    os.makedirs(SPLIT_DIR, exist_ok=True)
    def save_ids(ids, split):
        with open(os.path.join(SPLIT_DIR, f'{split}.txt'), 'w', encoding='utf-8') as f:
            for idx in ids:
                f.write(f'{idx}\n')
    save_ids(train_ids, 'train')
    save_ids(test_ids, 'test')

    # # Convert to numpy array.
    # def ids_to_np(ids, mid_snip_dict):
    #     lst = [(mid, start_fid, end_fid) for mid in ids for start_fid, end_fid in mid_snip_dict[mid]]
    #     return np.array(lst)
    # train_ary = ids_to_np(train_ids, mid_snip_dict)
    # test_ary = ids_to_np(test_ids, mid_snip_dict)

    # # Load from npy.
    # data_dict = {}
    # for mid in mid_snip_dict.keys():
    #     file_path = os.path.join(DATA_DIR, f'{mid:08}.npy')
    #     data_np = np.load(file_path)
    #     data_dict[mid] = data_np

    # # Get the length distribution.

    # # Sample a 10-frame snippet.
    # for mid, st_fid, end_fid in train_ary:
    #     if end_fid - st_fid == 10:
    #         sampled = (mid, st_fid, end_fid) # (72, 5, 15)
    #         print(sampled)
    #         break

    
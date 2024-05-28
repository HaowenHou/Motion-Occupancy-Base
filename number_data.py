"""
This script generates a csv file,
which numbers all the data file, and records the path.
CSV format: num,path(without ext)
"""

import os

if __name__ == '__main__':
    DATASETS_ROOT_DIR = 'path/to/datasets'

    data_dirname = DATASETS_ROOT_DIR.split('/')[-1]
    num_path_dict = {}
    cnt = 0
    for root, dirs, files in os.walk(DATASETS_ROOT_DIR):
        for file_ in files:
            if file_.endswith('.pkl') or file_.endswith('.npz'):
                # Only retains the part of path in AMASS_DIR.
                rel_path = os.path.join(root, file_).split(data_dirname + '/')[-1]
                rel_path = os.path.splitext(rel_path)[0]
                num_path_dict[cnt] = rel_path
                cnt += 1
    
    # Dump to csv file.
    with open('data_num_path.csv', 'w') as f:
        for num, path in num_path_dict.items():
            f.write('{},{}\n'.format(num, path))
            
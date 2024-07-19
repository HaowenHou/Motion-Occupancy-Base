# Motion Occupancy Base 

Official repository for gathering data of [*Revisit Human-Scene Interaction via Space Occupancy*](https://foruck.github.io/occu-page/) (ECCV 2024).

This repository specifies the pipeline for gathering Motion Occupancy Base (MOB).

## Environment setup

Create a conda environment from `environment.yml`: `conda env create -f environment.yml`

## Data preparation

### Gather existing datasets

This tutorial takes AMASS as an example. For the datasets in AMASS, download from [AMASS](https://amass.is.tue.mpg.de/), and put them under a directory like below:

```
.
├── ACCAD
├── BioMotionLab_NTroje
├── BMLhandball
├── BMLmovi
├── CMU
├── DFaust_67
├── EKUT
├── Eyes_Japan_Dataset
├── HumanEva
├── KIT
├── MPI_HDM05
├── MPI_Limits
├── MPI_mosh
├── SFU
├── SSM_synced
├── TCD_handMocap
├── TotalCapture
└── Transitions_mocap
```

### Generate Motion Occupancy Base

1. Configure

    In `number_data.py`, set `DATASETS_ROOT_DIR` to the path of the datasets.

    In `config.yml`,

    - `DATASET_DIR`: The path of the datasets
    - `MID_SNIP_DICT_DIR`: The pkl file to save, which contains motion id and corresponding snippet, to be introduced below.
    - `NPY_SAVE_DIR`: The path to save the processed motion data
    - `MALE_BM_PATH` & `FEMALE_BM_PATH`: The path to smplh_model/male_or_female/model.npz
    - `DEVICE`, `BATCH_SIZE`, etc. can be specified.

1. Number the data

    Run `number_data.py` , which numbers all the raw data and saves the mapping in `circle_num_path.csv`, formatted as `<num>,<stem_filename>`.

2. Extract the data snippets

    Run `extract_data_snippets.py`, which extracts valid data snippets, and saves them into `mid_snip_dict.pkl`.
    This file contains a list of tuples, each representing a snippet `(<motion_id>, <start_frame>, <end_frame>)`.

3. Process the data and save

    Run `generate_data.py`, which processes the data snippets, and saves them into `.npy` files.
    
    Here are some sample npy files processed from AMASS: [Google Drive link](https://drive.google.com/drive/folders/1VC4FJnfARNQbdYiaVKjHMnRAGjXvXNHp?usp=sharing)

## Citation

BibTeX:

```BibTeX
@article{liu2023revisit,
  title={Revisit Human-Scene Interaction via Space Occupancy},
  author={Liu, Xinpeng and Hou, Haowen and Yang, Yanchao and Li, Yong-Lu and Lu, Cewu},
  journal={arXiv preprint arXiv:2312.02700},
  year={2023}
}
```
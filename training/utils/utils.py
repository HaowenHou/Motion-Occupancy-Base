import argparse
from collections import OrderedDict
import shutil
import zipfile
from omegaconf import OmegaConf
import os
import logging
import time
import torch
import numpy as np
import random


def get_config_path():
    parser = argparse.ArgumentParser(description="Specify path to config file.")
    parser.add_argument("-c", "--config", default="configs/config.yml", help="Path to the configuration file.")
    args = parser.parse_args()
    return args.config

def create_logger(config: OmegaConf, to_file=True) -> logging.Logger:
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f'{config.STAGE}_{current_time}_{eval(f"config.{config.STAGE}.{config.STAGE}_NAME")}'
    config.RUN_NAME = run_name

    log_format = '%(asctime)s %(message)s'
    time_format = '%Y-%m-%d %H:%M:%S'
    logger = logging.getLogger()
    logger.setLevel(eval(f'logging.{config.LOGGING_LEVEL.upper()}'))
    formatter = logging.Formatter(log_format, datefmt=time_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if to_file:
        result_dir = os.path.join(config.ASSETS.RESULT_DIR, run_name)
        os.makedirs(result_dir, exist_ok=True)
        log_file = os.path.join(result_dir, run_name + '.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def backup_config_file(config, config_path):
    result_dir = os.path.join(config.ASSETS.RESULT_DIR, config.RUN_NAME)
    config_filename = config.RUN_NAME + '_config.yaml'
    shutil.copyfile(config_path, os.path.join(result_dir, config_filename))

# def dict_to_device(data: dict[str: torch.Tensor], device: torch.device) -> dict:
#     return {k: v.to(device, non_blocking=True) for k, v in data.items()}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def add_to_zip(zipf, path, arcname=None):
    """Recursively add files and directories to the zip"""
    if os.path.isfile(path):
        zipf.write(path, arcname if arcname else path)
    elif os.path.isdir(path):
        if arcname is None:
            arcname = path
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                full_path = os.path.join(dirpath, f)
                rel_path = os.path.join(arcname, f)
                zipf.write(full_path, rel_path)

def create_zip(zip_filename, files, dirs):
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for f in files:
            add_to_zip(zipf, f)
        for d in dirs:
            add_to_zip(zipf, d)

def backup_code(config, config_path):
    files = [config_path, 'train.py', 'infer.py', 'dataset.py', 
             'utils/*.py', 'models/*.py']
    dirs = []
    result_dir = os.path.join(config.ASSETS.RESULT_DIR, config.RUN_NAME)
    zip_filename = config.RUN_NAME + '_code.zip'
    create_zip(os.path.join(result_dir, zip_filename), files, dirs)

def get_save_schedule(restart_period, restart_mult):
    result = []
    current_term = restart_period

    for i in range(10):
        result.append(current_term)
        current_term += restart_period * (restart_mult ** (i + 1))
    return result
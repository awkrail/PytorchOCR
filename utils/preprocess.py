import os
import yaml
import torch

from argparse import ArgumentParser
from utils.logging import get_logger

class ArgParser(ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument("-c", "--config", type=str, required=True, help="configuration file for training/evaluating.")

    def parse_args(self):
        args = super(ArgParser, self).parse_args()
        return args

def load_config(file_path):
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config

def preprocess(is_train=False):
    args = ArgParser().parse_args()
    config = load_config(args.config)

    if is_train:
        save_model_dir = config["Global"]["save_model_dir"]
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, "config.yml"), "w") as f:
            yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = "{}/train.log".format(save_model_dir)
    else:
        log_file = None

    log_ranks = config["Global"].get("log_ranks", "0")
    logger = get_logger(log_file=log_file, log_ranks=log_ranks)
    use_gpu = config["Global"].get("use_gpu", False)

    algorithm = config["Architecture"]["algorithm"]
    assert algorithm in [
        "CRNN"
    ]

    device = "gpu:{}".format(0) if use_gpu else "cpu"
    logger.info("train with PyTorch {} and device {}".format(torch.__version__, device))
    return config, device, logger

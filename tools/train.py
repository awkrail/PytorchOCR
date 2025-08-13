import argparse
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def load_config(file_path):
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config


def main(config_path):
    config = load_config(config_path)
    device = 'cpu' # todo: -> gpu

    # build dataloader
    train_dataloader = build_dataloader(config, "Train", device)
    val_dataloader = build_dataloader(config, "Eval", device)

    # build loss
    loss_class = build_loss(config["Loss"])

    # build optimizer
    optimizer, lr_scheduler = build_optmizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        model=model,
    )

    # build metric
    eval_class = build_metric(config["Metric"])


    # load pretrained model
    pre_best_model_dict = load_model(
        config,
        model,
        optimizer,
        config["Architecture"]["model_type"]
    )

    train(
        config,
        train_dataloader,
        val_dataloader,
        device,
        model,
        loss_class,
        optimizer,
        lr_scheduler,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="configuration file for training a model, e.g., configs/rec/rec_en_number_lite_train.yml")
    args = parser.parse_args()
    main(args.config)

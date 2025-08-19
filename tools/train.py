import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import torch
import torch.nn as nn

from utils.preprocess import preprocess

from pytorchocr.dataset.dataset import build_dataloader
from pytorchocr.loss.loss import build_loss



def main(config, device, logger):
    # build dataloader
    train_dataloader = build_dataloader(config, "Train", device, logger)
    val_dataloader = build_dataloader(config, "Eval", device, logger)

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
    config, device, logger = preprocess(is_train=True)
    main(config, device, logger)

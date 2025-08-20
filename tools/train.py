import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.preprocess import preprocess

from pytorchocr.dataset.dataset import build_dataloader
from pytorchocr.loss.loss import build_loss
from pytorchocr.metric.metric import build_metric
from pytorchocr.postprocess.postprocess import build_postprocess
from pytorchocr.model.model import build_model
from pytorchocr.optimizer.optimizer import build_optimizer

def load_pretrained_weights(model, global_config, logger):
    if "checkpoints" in global_config:
        checkpoint_path = global_config["checkpoints"]
        pretrained_dict = torch.load(checkpoint_path)
        model_dict = model.state_dict()

        filtered_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        logger.info("loaded pretrained weights: {}".format(checkpoint_path))
    else:
        logger.warning("checkpoints in config is NULL, so the weights are loaded from random weights")


def main(config, device, logger):
    # build dataloader
    train_dataloader = build_dataloader(config, "Train", device, logger)
    val_dataloader = build_dataloader(config, "Eval", device, logger)

    # build loss
    loss = build_loss(config["Loss"])

    # build metric
    metric = build_metric(config["Metric"])

    # build postprocess
    post_processor = build_postprocess(config["PostProcess"], config["Global"])

    # build model
    if config["Global"]["task"] == "rec":
        character_num = len(post_processor.character)
        config["Architecture"]["Head"]["out_channels"] = character_num

    model = build_model(config["Architecture"])
    load_pretrained_weights(model, config["Global"], logger)

    # build optimizer
    optimizer, lr_scheduler = build_optmizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        model=model,
    )

    """
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
    """

if __name__ == "__main__":
    config, device, logger = preprocess(is_train=True)
    main(config, device, logger)

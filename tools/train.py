import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
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
        model.load_state_dict(pretrained_dict)
        logger.info("Loaded pretrained weights: {}".format(checkpoint_path))
    else:
        logger.warning("Checkpoints in config is None, so the weights are loaded from random weights")


def evaluate(
    model,
    device,
    val_dataloader,
    evaluator,
    post_processor,
    ):
    model.eval()
    pbar = tqdm(
        total = len(val_dataloader),
        desc = "eval model: ",
        position = 0,
        leave = True,
    )

    with torch.no_grad():
        evaluator.reset()

        for idx, batch in enumerate(val_dataloader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length'].to(device)

            preds = model(images)
            post_result = post_processor(preds['res'])
            pred_words = [x[0] for x in post_result]
            evaluator(pred_words, batch['word'])
            pbar.update(1)

    accuracy = evaluator.calculate_accuracy()
    ned = evaluator.calculate_ned()
    pbar.close()
    return accuracy, ned


def train(
    logger,
    config,
    train_dataloader,
    val_dataloader,
    device,
    model,
    criterion,
    evaluator,
    post_processor,
    optimizer,
    lr_scheduler
    ):

    epoch_num = config['Global']['epoch_num']
    eval_batch_step = config['Global']['eval_batch_step'][1]
    save_path = "{}/best_checkpoint_{}_{}_{}.pth"
    max_accuracy = 0

    for epoch in range(epoch_num):
        for idx, batch in enumerate(train_dataloader):
            model.train()
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length'].to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels, lengths)

            loss.backward()
            optimizer.step()

            if idx % eval_batch_step == 0:
                accuracy, ned = evaluate(model, device, val_dataloader, evaluator, post_processor)
                logger.info("Epoch: {} Iter {} accuracy = {} NED = {}".format(epoch, idx, accuracy, ned))

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    save_model_path = save_path.format(config["Global"]["save_model_dir"], epoch, idx, max_accuracy)
                    torch.save(model.state_dict(), save_model_path)
                    logger.info("Max accuracy is updated, accuracy = {} NED = {}".format(max_accuracy, ned))
                    logger.info("Model saved: {}".format(save_model_path))

        lr_scheduler.step()


def main(config, device, logger):
    # build dataloader
    train_dataloader = build_dataloader(config, "Train", device, logger)
    val_dataloader = build_dataloader(config, "Eval", device, logger)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    model.to(device)

    # build optimizer
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        model=model,
    )

    train(
        logger,
        config,
        train_dataloader,
        val_dataloader,
        device,
        model,
        loss,
        metric,
        post_processor,
        optimizer,
        lr_scheduler,
    )

if __name__ == "__main__":
    config, device, logger = preprocess(is_train=True)
    main(config, device, logger)

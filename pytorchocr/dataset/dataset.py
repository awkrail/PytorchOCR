from torch.utils.data import DataLoader
from .rec_dataset import RecognitionDataset

DATASET_DICT = {
    "RecognitionDataset" : RecognitionDataset,
}


def build_dataloader(config, mode, device, logger):
    assert 'dataset' in config[mode], "dataset should be included in config"
    assert 'name' in config[mode]['dataset'], "name should be included in config['mode']['dataset']"
    assert config[mode]['dataset']['name'] in DATASET_DICT, "{} is not registered as dataset".format(config[mode]['dataset']['name'])

    dataset_name = config[mode]['dataset']['name']
    dataset = DATASET_DICT[dataset_name](config, mode, logger)

    loader_config = config[mode]["loader"]
    batch_size = loader_config["batch_size_per_card"]
    drop_last = loader_config["drop_last"]
    shuffle = loader_config["shuffle"]
    num_workers = loader_config["num_workers"]

    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False,
    )

    return data_loader

from .rec_ctc_loss import CTCLoss

LOSS_DICT = {
    "CTCLoss" : CTCLoss,
}

def build_loss(config):
    assert 'name' in config, "name should be included in loss config"
    assert config['name'] in LOSS_DICT, "{} is not registered as loss".format(config[name])
    return LOSS_DICT[config['name']](**config)

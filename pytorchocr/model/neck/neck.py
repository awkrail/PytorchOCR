from .rnn import SequenceEncoder

NECK_DICT = {
    "SequenceEncoder" : SequenceEncoder,
}

def build_neck(config):
    assert "name" in config, "name should be set in neck config"
    assert config["name"] in NECK_DICT, "{} is not registered in NECK_DICT".format(config["name"])
    return NECK_DICT[config["name"]](**config)

from .ctc_head import CTCHead

HEAD_DICT = {
    "CTCHead" : CTCHead,
}

def build_head(config):
    assert "name" in config, "name should be set in head config"
    assert config["name"] in HEAD_DICT, "{} is not registered in HEAD_DICT".format(config["name"])
    return HEAD_DICT[config["name"]](**config)

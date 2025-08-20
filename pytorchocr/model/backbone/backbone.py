from .mobilenet_v3 import MobileNetV3

BACKBONE_DICT = {
    "MobileNetV3" : MobileNetV3,
}

def build_backbone(config):
    assert "name" in config, "name should be set in backbone config"
    config["name"] in BACKBONE_DICT, "{} is not registered in BACKBONE_DICT".format(config["name"])
    return MobileNetV3(**config)

import torch.nn as nn

from pytorchocr.model.backbone.backbone import build_backbone
from pytorchocr.model.neck.neck import build_neck
from pytorchocr.model.head.head import build_head

class RecognitionModel(nn.Module):
    def __init__(self, config):
        super(RecognitionModel, self).__init__()
        in_channels = config.get("in_channels", 3)
        model_type = config["model_type"]

        # Backbone: CNN, e.g., MobileNetV3
        assert "Backbone" in config, "Backbone should be set in config."
        config["Backbone"]["in_channels"] = in_channels
        self.backbone = build_backbone(config["Backbone"])
        in_channels = self.backbone.out_channels

        # Neck: RNN, e.g., LSTM
        assert "Neck" in config, "Neck should be set in config."
        config["Neck"]["in_channels"] = in_channels
        self.neck = build_neck(config["Neck"])
        in_channels = self.neck.out_channels

        # Head: 
        assert "Head" in config, "Head should be set in config."
        config["Head"]["in_channels"] = in_channels
        self.head = build_head(config["Head"])

    def forward(self, x, data=None):
        pass


MODEL_DICT = {
    "RecognitionModel" : RecognitionModel,
}

def build_model(config):
    assert "model_type" in config, "model_type should be set in architecture."
    if config["model_type"] == "rec":
        return RecognitionModel(config)
    else:
        raise NotImplementedError()

import torch.nn as nn

from pytorchocr.modeling.backbone.backbone import build_backbone
from pytorchocr.modeling.head.head import build_head

class RecognitionModel(nn.Module):
    def __init__(self, config):
        super(RecognitionModel, self).__init__()
        import ipdb; ipdb.set_trace()

    def forward(self, x, data=None):
        pass

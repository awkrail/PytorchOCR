import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels = 6625,
        return_feats = False,
        **kwargs
    ):
        super(CTCHead, self).__init__()
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            bias = True,
        )
        self.out_channels = out_channels
        self.return_feats = return_feats

    def forward(self, x, data = None):
        predicts = self.fc(x)
        result = { 'res' : predicts }
        if self.return_feats:
            result['feat'] = x
        return result

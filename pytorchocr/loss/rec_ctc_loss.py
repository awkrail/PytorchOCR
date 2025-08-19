import torch.nn as nn

class CTCLoss(nn.Module):
    def __init__(
        self,
        use_focal_loss = False, 
        **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(reduction = 'none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        pass

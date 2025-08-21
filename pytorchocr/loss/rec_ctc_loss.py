import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    def __init__(
        self,
        use_focal_loss = False, 
        **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = True)
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, labels, label_lengths):
        predicts = predicts['res']

        batch_size = predicts.size(0)
        predicts = predicts.log_softmax(2)
        predicts = predicts.permute(1, 0, 2)

        pred_lengths = torch.tensor([predicts.size(0)] * batch_size, dtype=torch.long).to('cuda')
        loss = self.loss_func(predicts, labels, pred_lengths, label_lengths)

        return loss.mean()

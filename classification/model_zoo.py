import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, gamma=0.2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, label_score, target):
        """

        :param label_score: [B, C]
        :param target: [B, C]
        :return:
        """
        logit = F.softmax(label_score, dim=-1)
        logit = torch.gather(logit, 1, target.unsqueeze(1))
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * torch.log(logit) * (1 - logit) ** self.gamma  # focal loss

        return loss.mean()

import torch
import torch.nn as nn

# Custom AlphaPredictionLoss
class AlphaPredictionLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(AlphaPredictionLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, alpha_pred, alpha_true):
        squared_diff = (alpha_pred - alpha_true) ** 2
        loss = torch.sqrt(squared_diff + self.epsilon ** 2)
        loss = loss.sum()
        return loss

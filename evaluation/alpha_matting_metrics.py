import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from torchmetrics import Metric

#Reference: Deep Image Matting: A Comprehensive Survey (https://arxiv.org/pdf/2304.04672.pdf)

class AlphaMattingMetrics(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sad", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sad_t", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mad", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # self.add_state("grad_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("conn", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.num_samples = 0

    def update(self, predicted_alpha, true_alpha):
        self.sad += torch.sum(torch.abs(predicted_alpha - true_alpha))
        self.mse += torch.mean((predicted_alpha - true_alpha) ** 2)
        self.sad_t += torch.sum(torch.abs(predicted_alpha - true_alpha)[predicted_alpha - true_alpha <= 0.1])
        self.mad += torch.mean(torch.abs(predicted_alpha - true_alpha))

#         pred_alpha_grad_x = F.conv2d(predicted_alpha, torch.Tensor([[-1, 1]]).to(predicted_alpha.device), padding=1)
#         pred_alpha_grad_y = F.conv2d(predicted_alpha, torch.Tensor([[-1], [1]]).to(predicted_alpha.device), padding=1)

#         true_alpha_grad_x = F.conv2d(true_alpha, torch.Tensor([[-1, 1]]).to(true_alpha.device), padding=1)
#         true_alpha_grad_y = F.conv2d(true_alpha, torch.Tensor([[-1], [1]]).to(true_alpha.device), padding=1)

#         grad_error_x = torch.mean(torch.abs(pred_alpha_grad_x - true_alpha_grad_x))
#         grad_error_y = torch.mean(torch.abs(pred_alpha_grad_y - true_alpha_grad_y))
#         self.grad_error += grad_error_x + grad_error_y

        binarized_pred = (predicted_alpha > 0.1).float()
        binarized_true = (true_alpha > 0.1).float()
        pred_connected = torch.sum(binarized_pred)
        true_connected = torch.sum(binarized_true)
        intersection = torch.sum(binarized_pred * binarized_true)
        self.conn += 1.0 - (2.0 * intersection) / (pred_connected + true_connected)

        self.num_samples += predicted_alpha.size(0)

    def compute(self):
        return {
            "SAD": self.sad / self.num_samples,
            "MSE": self.mse / self.num_samples,
            "SAD-T": self.sad_t / self.num_samples,
            "MAD": self.mad / self.num_samples,
            # "GRAD": self.grad_error / self.num_samples,
            "CONN": self.conn / self.num_samples,
        }
#usage example 
# class YourLightningModule(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.metrics = AlphaMattingMetrics()

#     def forward(self, x):
#         # Your model forward pass here
#         pass

#     def training_step(self, batch, batch_idx):
#         # Training logic here
#         preds = self.forward(batch)
#         loss = your_loss_function(preds, batch['ground_truth'])
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         preds = self.forward(batch)
#         self.metrics.update(preds, batch['ground_truth'])

#     def validation_epoch_end(self, outputs):
#         metrics = self.metrics.compute()
#         self.log_dict(metrics, prog_bar=True)

#     def configure_optimizers(self):
#         # Configure your optimizer and scheduler here
#         pass

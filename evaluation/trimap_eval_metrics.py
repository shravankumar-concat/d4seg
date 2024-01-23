import torch
import pytorch_lightning as pl
from torchmetrics import Metric
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MeanSquaredError(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        mse = mean_squared_error(target.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy())
        self.mse_sum += torch.tensor(mse)
        self.num_samples += preds.size(0)

    def compute(self):
        return self.mse_sum / self.num_samples

class MeanAbsoluteDifference(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mad_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        mad = mean_absolute_error(target.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy())
        self.mad_sum += torch.tensor(mad)
        self.num_samples += preds.size(0)

    def compute(self):
        return self.mad_sum / self.num_samples

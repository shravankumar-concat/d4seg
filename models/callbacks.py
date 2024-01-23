import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import os

# Callback for visualization
class VisualizationCallback(pl.Callback):
    def __init__(self, model, get_predicted_mask_func, dataloader):
        self.model = model
        self.get_predicted_mask_func = get_predicted_mask_func
        self.dataloader = dataloader

    def on_epoch_end(self, trainer, pl_module):
        batch = next(iter(self.dataloader))
        test_image = batch['image'][0]
        test_image = test_image.to(self.model.device)
        self.model.eval()
        with torch.no_grad():
            pred_mask = self.get_predicted_mask_func(self.model, test_image.unsqueeze(0))
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_mask.cpu().numpy().squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        try:
            log_dir = trainer.log_dir
            figure_path = os.path.join(log_dir, f'epoch_{trainer.current_epoch}_predicted_mask.png')
        except TypeError:
            base_path = "/home/shravan/documents/deeplearning/github/segmentation_models/notebooks/trimap_generation/results"
            figure_path = f"{base_path}/epoch_{trainer.current_epoch}_predicted_mask.png"
        plt.savefig(figure_path)
        plt.close()
        self.model.train()
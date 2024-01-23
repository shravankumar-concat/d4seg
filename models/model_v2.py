import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
# from torchmetrics import IoU
import segmentation_models_pytorch as smp
from einops import rearrange
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from networks.unet_plain_1024 import UNetPlain1024
from networks.att_unet import UNetWithAttention2

from networks.mit import MiTB5
from networks.segformer import SegFormerHead
from networks.custom_segformer import MySegmentationModel

from losses.trimap_losses import AlphaPredictionLoss
from evaluation.alpha_matting_metrics import AlphaMattingMetrics

# Define the LightningModule for training
class SegFormerLightning(pl.LightningModule):
    def __init__(self, config):
        super(SegFormerLightning, self).__init__()
        
        if config["arch"]=="Unet":
            self.model = UNetWithAttention2(config['in_channels'], config['out_classes'])
        elif config["arch"]=="Unet_Plain":
            self.model = UNetPlain1024(config['in_channels'], config['out_classes'])
        else:
            self.model = MySegmentationModel()  #MiT-B5
            
        # Create an instance of the UNet model
        self.config = config
        
        # Dynamic selection of loss functions based on loss_fn_names argument        
        self.loss_weights = [float(weight) for weight in config['loss_weights']]
        self.loss_fn_names = config['loss_fns']
        
        # Calculate and save the combined loss string as a hyperparameter
        self.combined_loss_string = self.calculate_combined_loss_string()
        
        self.loss_fns = self._select_loss_functions()
        
        self.metrics = AlphaMattingMetrics()
        self.lr = config['lr']

        self.save_hyperparameters()  # Save hyperparameters for tracking
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

        # Define a step function that reduces the learning rate by a factor of 0.5 at epoch 100
        def lr_scheduler_step(epoch):
            if epoch < 100:
                return 1.0  # Keep the initial learning rate
            else:
                return 0.5  # Reduce the learning rate by a factor of 0.5
            
        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',         # Monitor the minimum valid_loss
            factor=0.5,         # Reduce LR by a factor of 0.5
            patience=5,         # Number of epochs with no improvement after which LR will be reduced
            verbose=True,       # Print a message when LR is reduced
            threshold=0.0001,   # Minimum change in the monitored quantity to qualify as an improvement
        )


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Adjust the LR scheduler at the end of each epoch
                "monitor": "val_loss",  # Monitor the valid_loss for LR scheduling
            }
        }


    def shared_step(self, batch, stage):
        inputs, targets = batch["image"], batch["mask"]
        # print(inputs.shape, targets.shape)
        outputs = self(inputs)
        
        # Calculate combined loss
        loss = self.compute_combined_loss(self.loss_fns, outputs, targets)
        self.log(f'{stage}_loss', loss)
        
        self.metrics.update(outputs, targets)
        
        return loss

    def shared_epoch_end(self, outputs, stage):
        pass

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 'train')
        return loss

    def training_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 'val')
        return loss

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'val')
        metrics = self.metrics.compute()
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 'test')
        return loss

    def test_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'test')
        
    def _select_loss_functions(self):
        loss_name_mapping = {
            'DiceLoss': smp.losses.DiceLoss(smp.losses.BINARY_MODE),
            'FocalLoss': smp.losses.FocalLoss(smp.losses.BINARY_MODE),
            'LovaszLoss': smp.losses.LovaszLoss(smp.losses.BINARY_MODE),
            'JaccardLoss': smp.losses.JaccardLoss(mode=smp.losses.BINARY_MODE),
            'AlphaLoss': AlphaPredictionLoss(epsilon=1e-6),  # Keep AlphaPredictionLoss
        }

        selected_losses = []
        for loss_name in self.loss_fn_names:
            if loss_name in loss_name_mapping:
                selected_losses.append(loss_name_mapping[loss_name])
            else:
                raise ValueError(f"Unsupported loss function: {loss_name}")

        return selected_losses    
    
    def calculate_combined_loss_string(self):
        loss_fn_names_str = ', '.join([f"{weight:.2f}*{loss_name}" 
                                       for weight, loss_name in zip(
                                           self.loss_weights, self.loss_fn_names)]
                                     )
        return f"Combined Loss ({loss_fn_names_str})"
    
    
    def compute_combined_loss(self, loss_fns, logits_mask, mask):
        combined_loss = 0.0
        individual_losses = {}
        for weight, loss_fn in zip(self.loss_weights, loss_fns):
            loss = loss_fn(logits_mask, mask)
            combined_loss += weight * loss
            individual_losses[loss_fn.__class__.__name__] = loss.item()
        
        # Log individual loss values
        for loss_name, loss_value in individual_losses.items():
            self.log(f"individual_loss_{loss_name}", loss_value, on_epoch=True)
            
        return combined_loss
        
    def eval_model_and_log(self, dataloader):
        self.model.eval()
        batch = next(iter(dataloader))
        
        # Calculate the length of the batch
        batch_size = len(batch["image"])

        # Choose a random index within the batch
        random_idx = random.randint(0, batch_size - 1)

        image = batch["image"].to(self.device)
        mask = batch["mask"].to(self.device)

        with torch.no_grad():
            logits_mask = self.forward(image)
            # prob_mask = logits_mask.sigmoid()
            prob_mask = logits_mask.clamp(0,1)
        idx = random_idx
        image_array = image[idx].cpu().numpy().squeeze().transpose(1, 2, 0)
        mask_array = mask[idx].cpu().numpy().squeeze()
        mask_array = np.expand_dims(mask_array, axis=-1)
        prob_mask_array = prob_mask[idx].cpu().numpy().squeeze()

        # image_pil = Image.fromarray(np.uint8(image_array))
        image_pil = Image.fromarray(np.uint8(image_array * 255))
        mask_pil = Image.fromarray(np.uint8(mask_array.squeeze() * 255))
        prob_mask_pil = Image.fromarray(np.uint8(prob_mask_array*255))

        return image_pil, mask_pil, prob_mask_pil  

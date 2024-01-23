import os
import sys
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from utils.dice_score import dice_coeff
from PIL import Image
import wandb
import numpy as np
from einops import rearrange
import random

class CarSegmentationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.resize_height = config['resize_height']
        self.resize_width = config['resize_width']
        
        self.model = smp.create_model(
            config['arch'],
            encoder_name=config['encoder_name'],
            in_channels=config['in_channels'],
            classes=config['out_classes'],
        )
        
        # self.car_segmenter_inference_model = CarSegmenterInferenceModel(self.model)

        params = smp.encoders.get_preprocessing_params(config['encoder_name'])
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        # Dynamic selection of loss functions based on loss_fn_names argument        
        self.loss_weights = [float(weight) for weight in config['loss_weights']]
        self.loss_fn_names = config['loss_fns']
        
        # Calculate and save the combined loss string as a hyperparameter
        self.combined_loss_string = self.calculate_combined_loss_string()
        
        self.loss_fns = self._select_loss_functions()

        self.save_hyperparameters()  # Save hyperparameters for tracking

    
    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask


    def _select_loss_functions(self):
        loss_name_mapping = {
            'DiceLoss': smp.losses.DiceLoss(smp.losses.BINARY_MODE),
            'FocalLoss': smp.losses.FocalLoss(smp.losses.BINARY_MODE),
            'LovaszLoss': smp.losses.LovaszLoss(smp.losses.BINARY_MODE),
            'JaccardLoss': smp.losses.JaccardLoss(mode=smp.losses.BINARY_MODE),
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

    def shared_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4, f"Expected 4 but image has ndim = {image.ndim}"
        b, c, h, w = image.shape
        assert h % 32 == 0 and w % 32 == 0, f"{h % 32} != 0 and {w % 32} == 0 for image shape{image.shape}"
        self.log("batch_size", b)

        mask = batch["mask"]
        assert mask.ndim == 4, f"Expected 4 but mask has ndim = {mask.ndim}, mask shape is: {mask.shape}"
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        prob_mask = logits_mask.sigmoid()
        
        # Compute the Dice score
        dice_score = dice_coeff(prob_mask.float(), mask.float(), reduce_batch_first=False)        
       # Combine losses based on selected loss functions and weights
        loss = self.compute_combined_loss(self.loss_fns, logits_mask, mask)
        # loss = self.compute_combined_loss(self.loss_fns, prob_mask, mask)

        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice_score": dice_score.detach(),
        }
        

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, "train")
        self.log("train_loss", output["loss"], prog_bar=True, logger=True)
        self.log("train_dice_coeff", output["dice_score"], prog_bar=True, logger=True)
        return output

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, "valid")
        self.log("val_loss", output["loss"], prog_bar=True, logger=True)
        self.log("val_dice_coeff", output["dice_score"], prog_bar=True, logger=True)
        return output

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")    
    

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
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
            prob_mask = logits_mask.sigmoid()
        idx = random_idx
        image_array = image[idx].cpu().numpy().squeeze().transpose(1, 2, 0)
        mask_array = mask[idx].cpu().numpy().squeeze()
        mask_array = np.expand_dims(mask_array, axis=-1)
        prob_mask_array = prob_mask[idx].cpu().numpy().squeeze()

        image_pil = Image.fromarray(np.uint8(image_array))
        mask_pil = Image.fromarray(np.uint8(mask_array.squeeze() * 255))
        prob_mask_pil = Image.fromarray(np.uint8(prob_mask_array*255))

        return image_pil, mask_pil, prob_mask_pil  
    
    
    def compute_dice_coeff(self, dataloader):
        total_dice = 0.0
        num_samples = 0

        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                image = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)

                logits_mask = self.forward(image)
                prob_mask = logits_mask.sigmoid()

                dice = dice_coeff(prob_mask.float(), mask.float(), reduce_batch_first=False)
                total_dice += dice.sum().item()
                num_samples += dice.numel()

        self.model.train()

        return total_dice / num_samples
    
# Helper Functions for Validation and logging
    def preprocess(self, im_path):
        image = Image.open(im_path)
        image = np.array(image)
        image = np.array(Image.fromarray(image).resize(
            (self.resize_height, self.resize_width), 
            Image.Resampling.BILINEAR))
        # image = np.moveaxis(image, -1, 0)
        image = rearrange(image, 'h w c -> 1 c h w')
        image = torch.from_numpy(image)
        return image

    def get_predicted_mask(self, image):
        with torch.no_grad():
            self.model.eval()
            logits = self.model(image)
        prob_mask = logits.sigmoid()
        return prob_mask

    def segment_PIL(self, img_path):
        image_tensor = self.preprocess(img_path)
        mask_tensor = self.get_predicted_mask(image_tensor)
        
        image_array = image_tensor.numpy().squeeze().transpose(1, 2, 0)
        mask_array = mask_tensor.numpy().squeeze()
        mask_array = np.expand_dims(mask_array, axis=-1)
        
        seg_img = image_array * mask_array
        
        image_pil = Image.fromarray(np.uint8(image_array))
        mask_pil = Image.fromarray(np.uint8(mask_array.squeeze() * 255))
        seg_img_pil = Image.fromarray(np.uint8(seg_img))
        
        return image_pil, mask_pil, seg_img_pil    
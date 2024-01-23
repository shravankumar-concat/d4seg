import os
import random
import numpy as np
import torch
from networks.mit import MiTB5
from networks.segformer import SegFormerHead
import pytorch_lightning as pl
import torch.nn as nn

# # Define the custom segmentation model class
# class MySegmentationModel(nn.Module):
#     def __init__(self, encoder, segmentation_head):
#         super(MySegmentationModel, self).__init__()
#         self.encoder = encoder
#         self.segmentation_head = segmentation_head
#         self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

#     def forward(self, x):
#         features = self.encoder(x)
#         seg_logits = self.segmentation_head(features)
#         seg_logits_upsampled = self.upsample(seg_logits)
#         return seg_logits_upsampled
    
class MySegmentationModel(nn.Module):
    def __init__(self):
        super(MySegmentationModel, self).__init__()

        # Instantiate the encoder as you mentioned
        self.encoder = MiTB5()
        self.encoder.load_official_state_dict(filename='mit_b5.pth')
        self.encoder.reset_input_channel(4, pretrained=True)

        # Define input arguments for segmentation head
        input_args = dict(
            num_classes=1,
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            dropout_ratio=0.1,
            embedding_dim=768
        )

        # Instantiate the segmentation head
        self.segmentation_head = SegFormerHead(**input_args)

        # Define the upsampling layer to increase resolution from 256x256 to 1024x1024
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Forward pass through the encoder
        features = self.encoder(x)

        # Forward pass through the segmentation head
        seg_logits = self.segmentation_head(features)

        # Upsample the segmentation logits to 1024x1024
        seg_logits_upsampled = self.upsample(seg_logits)

        return seg_logits_upsampled
    

# Define the LightningModule for training
class SegFormerLightning(pl.LightningModule):
    def __init__(self, model, alpha_loss):
        super(SegFormerLightning, self).__init__()
        self.model = model
        self.alpha_loss = alpha_loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch["image"], batch["mask"]
        outputs = self(inputs)
        loss = self.alpha_loss(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch["image"], batch["mask"]
        outputs = self(inputs)
        loss = self.alpha_loss(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch["image"], batch["mask"]
        outputs = self(inputs)
        loss = self.alpha_loss(outputs, targets)
        self.log('test_loss', loss)
        return loss
    


# Instantiate your segmentation model
model = MySegmentationModel()

# Print the model architecture
print(model)

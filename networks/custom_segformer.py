import torch
import torch.nn as nn

from networks.mit import MiTB5
from networks.segformer import SegFormerHead

# Define the custom segmentation model class
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

        # Upsample the segmentation logits to 1024x1024/ to match with input shape
        seg_logits_upsampled = self.upsample(seg_logits)
        
        # Apply sigmoid activation to the output
        alpha_pred = torch.sigmoid(seg_logits_upsampled)
        
        return alpha_pred
    

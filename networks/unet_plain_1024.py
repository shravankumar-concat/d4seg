import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        x =  self.relu1(self.bn1(self.conv1(x)))
        x =  self.relu2(self.bn2(self.conv2(x)))
        pool = self.pool(x)
        return x,pool
    
class Center(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Center, self).__init__()
        self.down_block = DownBlock(in_channels, out_channels)

    def forward(self, x):
        x, _ = self.down_block(x)
        return x    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.upconv = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([skip, x], dim=1) ## concatenating along the channel dimension
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x
    
# Main Module - Plain UNet Architecture     
class UNetPlain1024(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetPlain1024, self).__init__()

        self.down0 = DownBlock(in_channels, 8)
        self.down1 = DownBlock(8, 16)
        self.down2 = DownBlock(16, 32)
        self.down3 = DownBlock(32, 64)
        self.down4 = DownBlock(64, 128)
        self.down5 = DownBlock(128, 256)
        self.down6 = DownBlock(256, 512)

        self.center = Center(512, 1024)

        self.up6 = UpBlock(1024, 512)
        self.up5 = UpBlock(512, 256)
        self.up4 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.up1 = UpBlock(32, 16)
        self.up0 = UpBlock(16, 8)

        self.seg_logits = nn.Conv2d(8, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        down0, down0_pool   = self.down0(x)
        down1, down1_pool   = self.down1(down0_pool)
        down2, down2_pool   = self.down2(down1_pool)
        down3, down3_pool   = self.down3(down2_pool)
        down4, down4_pool   = self.down4(down3_pool)
        down5, down5_pool   = self.down5(down4_pool)
        down6, down6_pool   = self.down6(down5_pool)

        center = self.center(down6_pool)

        up6 = self.up6(center, down6)
        up5 = self.up5(up6, down5)
        up4 = self.up4(up5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)
        up0 = self.up0(up1, down0)
        seg_logits = self.seg_logits(up0)
        alpha_pred = torch.sigmoid(seg_logits)

        return alpha_pred
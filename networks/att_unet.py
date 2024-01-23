import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self,in_channels):
        super(AttentionGate,self).__init__()
        self.inter_channel = in_channels

        self.theta_x = nn.Conv2d(in_channels,self.inter_channel,kernel_size=1, stride=1)
        self.theta_x_bn = nn.BatchNorm2d(self.inter_channel)

        self.phi_g = nn.Conv2d(in_channels*2,self.inter_channel,kernel_size=1, stride=1)
        self.phi_g_bn = nn.BatchNorm2d(self.inter_channel)


        self.psi_f = nn.Conv2d(self.inter_channel,1,kernel_size=1, stride=1)
        self.psi_f_bn = nn.BatchNorm2d(1)

        self.theta_phi_relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x ,g):
        theta_x = self.theta_x_bn(self.theta_x(x))
        phi_g = self.phi_g_bn(self.phi_g(g))
        if theta_x.shape != phi_g.shape:
            raise ValueError(
                f"Shape mismatch: theta_x.shape {theta_x.shape}, phi_g.shape {phi_g.shape}"
            )

        theta_x_phi_g =  theta_x + phi_g

        psi_f = self.psi_f_bn(self.psi_f(self.theta_phi_relu(theta_x_phi_g)))
        rate = self.sigmoid(psi_f)

        attn_out = x*rate
        return attn_out

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
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.Upsample(scale_factor=2, mode='nearest')

        self.att_gate = AttentionGate(out_channels)

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
        att = self.att_gate(skip, x)
        x = torch.cat([att, x], dim=1) ## concatenating along the channel dimension
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNetWithAttention, self).__init__()

        self.down0a = DownBlock(in_channels, 16)
        self.down0 = DownBlock(16, 32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.up4 = UpBlock(1024, 512)
        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)
        self.up0 = UpBlock(64, 32)
        self.up0a = UpBlock(32, 16)

        self.seg_logits = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        down0a, down0a_pool = self.down0a(x)
        down0, down0_pool   = self.down0(down0a_pool)
        down1, down1_pool   = self.down1(down0_pool)
        down2, down2_pool   = self.down2(down1_pool)
        down3, down3_pool   = self.down3(down2_pool)
        down4, down4_pool   = self.down4(down3_pool)

        center = self.center(down4_pool)
        
        up4 = self.up4(center, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)
        up0 = self.up0(up1, down0)
        up0a = self.up0a(up0, down0a)
        seg_logits = self.seg_logits(up0a)
        # Apply sigmoid activation to the output
        alpha_pred = torch.sigmoid(seg_logits)

        return alpha_pred
    
class UNetWithAttention2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithAttention2, self).__init__()

        self.down0 = DownBlock(in_channels, 8)
        self.down1 = DownBlock(8, 16)
        self.down2 = DownBlock(16, 32)
        self.down3 = DownBlock(32, 64)
        self.down4 = DownBlock(64, 128)
        self.down5 = DownBlock(128, 256)
        self.down6 = DownBlock(256, 512)

        self.center = DownBlock(512, 1024)

        self.up6 = UpBlock(1024, 512)
        self.up5 = UpBlock(512, 256)
        self.up4 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.up1 = UpBlock(32, 16)
        self.up0 = UpBlock(16, 8)

        self.seg_logits = nn.Conv2d(8, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # print(f"input shape: {x.shape}\n")
        down0, down0_pool   = self.down0(x)
        down1, down1_pool   = self.down1(down0_pool)
        down2, down2_pool   = self.down2(down1_pool)
        down3, down3_pool   = self.down3(down2_pool)
        down4, down4_pool   = self.down4(down3_pool)
        down5, down5_pool   = self.down5(down4_pool)
        down6, down6_pool   = self.down6(down5_pool)

        center,_ = self.center(down6_pool)
        # print(f"center: {center.shape}\n")
        
        up6 = self.up6(center, down6)
        up5 = self.up5(up6, down5)
        up4 = self.up4(up5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)
        up0 = self.up0(up1, down0)
        seg_logits = self.seg_logits(up0)
        alpha_pred = torch.sigmoid(seg_logits)

        # print(f"output shape: {alpha_pred.shape}\n")

        return alpha_pred
    
# # Instantiate the model
# input_channels = 4
# num_classes = 1
# model = UNetWithAttention(input_channels, num_classes)

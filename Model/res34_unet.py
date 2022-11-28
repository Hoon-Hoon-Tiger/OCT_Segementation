import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_down_sampling=False):
        super(ResidualBlock, self).__init__()
        
        self.convBlock = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=2)

        self.residual_function = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=2 if is_down_sampling else 1),
            ConvBlock(in_channels=out_channels, out_channels=out_channels)
        )
        self.is_down_sampling = is_down_sampling

    def forward(self, x):
        if self.is_down_sampling:
            return self.residual_function(x) + self.convBlock(x)
        else:
            return self.residual_function(x) + x
        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)                               

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class Res34_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Res34_UNet, self).__init__()
        self.n_channels = n_channels # 1
        self.n_classes = n_classes # 10
        self.bilinear = bilinear
        
        # 512 512 1 -> 512 512 64
        self.inc = DoubleConv(n_channels, 64)
        
        # 512 512 64 -> 256 256 128
        self.conv2_x = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 256 256 64
            ResidualBlock(64, 128, is_down_sampling=True),
            # 256 256 128
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        
        # 256 256 128 -> 128 128 256
        self.conv3_x = nn.Sequential(
            ResidualBlock(128, 256, is_down_sampling=True),
            # 128 128 256
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        
        # 128 128 256 -> 64 64 512
        self.conv4_x = nn.Sequential(
            ResidualBlock(256, 512, is_down_sampling=True),
            # 64 64 512
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )
        
        # 64 64 512 -> 32 32 1024
        self.conv5_x = nn.Sequential(
            ResidualBlock(512, 1024, is_down_sampling=True),
            # 32 32 1024
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024)
        )
        
        factor = 2 if bilinear else 1
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)        
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)
        
        # (32 32 1024 -> 64 64 512) + "64 64 512" => 64 64 1024 -> 64 64 512
        x = self.up1(x5, x4)
        
        # (64 64 512 -> 128 128 256) + "128 128 256" => 128 128 512 -> 128 128 256
        x = self.up2(x, x3)
        
        # (128 128 258 -> 256 256 128) + "256 256 128" => 256 256 256 -> 256 256 128 
        x = self.up3(x, x2)
        
        # (256 256 128 -> 512 512 64) + "512 512 64" => 512 512 128 -> 512 512 64
        x = self.up4(x, x1)
        
        # 512 512 64 -> 512 512 n_classes 
        logits = self.outc(x)
        return logits
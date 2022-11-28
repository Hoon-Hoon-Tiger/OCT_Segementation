import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import crop

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()

        if not mid_channels:
            mid_channels = out_channels
        
        '''
        # ? 왜 Conv2d에 bias를 False로 설정했는지?
        # -> 연산량을 줄이기 위해서 False를 적용해두었다고 생각한다.
        # -> 좀 더 나은 모델을 위해서는 True로 해주면 좋을 것 같다.
        # ? size를 유지한다.
        # -> 공식 torch 코드에서는 double_conv을 거치면서 사이즈를 유지시킨다. 
        '''
        # ! 나는 논문 그대로 padding을 주지 않고 conv를 진행한다. 
    
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )

        # TODO: upsampling 
        self.de_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )
        
    # TODO: crop(up sampling된 x의 크기에 맞춰서 skip_connection의 테두리를 잘라낸다)
    def crop(self, up_sampled_x, skip_connection):
        # input is CHW
        x_height = up_sampled_x.size()[2]
        x_width = up_sampled_x.size()[3]
        diffY = skip_connection.size()[2] - x_height
        diffX = skip_connection.size()[3] - x_width
        
        # ! 공식 torch 코드에서는 skip connection을 crop하지 않고 x(input)을 pad했다.
        up_sampled_x = F.pad(up_sampled_x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # ! 나는 논문상에 나와 있듯이 skip_connection의 테두리 부분을 잘라냈다.
        return crop(skip_connection, diffX // 2, diffY // 2, x_width, x_height)

    def forward(self, x, skip_connection):
        # TODO: upsampling한 x와 crop된 skip_connection을 cat하여 de_conv를 진행
        up_sampled_x = self.up(x)
        cropped_skip_connection = self.crop(up_sampled_x, skip_connection)
        print('up_sampled_x >>>', np.shape(up_sampled_x))
        print('cropped_skip_connection >>>', np.shape(cropped_skip_connection))
        return self.de_conv(torch.cat([up_sampled_x, cropped_skip_connection], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 572 572 1
        x1 = self.inc(x)
        # 568 568 64
        x2 = self.down1(x1)
        # 280 280 128
        x3 = self.down2(x2)
        # 136 136 256
        x4 = self.down3(x3)
        # 64 64 512
        x5 = self.down4(x4)
        # 28 28 1024

        x = self.up1(x5, x4)
        # 52 52 512
        x = self.up2(x, x3)
        # 100 100 256
        x = self.up3(x, x2)
        # 196 196 128
        x = self.up4(x, x1)
        # 388 388 64
        logits = self.outc(x)
        # 388 388 2      
        return logits
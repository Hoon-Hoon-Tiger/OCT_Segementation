import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()

        # padding을 주지 않아서 사이즈가 달라졌었다.
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

        '''
        # blocks 변수가 단순히 python list여서 gpu에 올리는 과정에서 누락되었었다.
        self.blocks = []
        for _ in range(depth):
            conv = ConvBlock(in_channels, out_channels,
                             stride=2 if is_down_sampling else stride)
            self.blocks.append(conv)
        self.blocks = nn.ModuleList(self.blocks)
    
    def forward(self, x):
        fx = x
        for block in self.blocks:
            fx = block(fx)
        return fx + x
        '''
        # 커널 사이즈를 1 x 1, padding 0, stride 2로 하여 사이즈만 줄여주는 식으로
        # 기존의 conv는 3 x 3 로 압축 또는 누락되는 부분이 있다고 할 수 있다.
        # 따라서, 단순히 x의 역할을 하기 위해서 사이즈를 맞춰주기 위함으로
        # 위와 같이 해줄 수 있다.

        # 또한, ConvBlock을 생성하고 다운 샘플링을 하지 않는 단계에서는 메모리만 차지하고 있기 때문에 비효율적이다.
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


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.conv3_x = nn.Sequential(
            ResidualBlock(64, 128, is_down_sampling=True),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        self.conv4_x = nn.Sequential(
            ResidualBlock(128, 256, is_down_sampling=True),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        self.conv5_x = nn.Sequential(
            ResidualBlock(256, 512, is_down_sampling=True),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
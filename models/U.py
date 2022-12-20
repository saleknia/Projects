import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

reduce_factor = 4

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Sequential(
                        nn.Conv2d(in_channels                , in_channels//reduce_factor , kernel_size=1, padding=0),
                        nn.Conv2d(in_channels//reduce_factor , mid_channels//reduce_factor, kernel_size=3, padding=1),
                        nn.Conv2d(mid_channels//reduce_factor, mid_channels               , kernel_size=1, padding=0),
                        ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Sequential(
                        nn.Conv2d(mid_channels                , mid_channels//reduce_factor , kernel_size=1, padding=0),
                        nn.Conv2d(mid_channels//reduce_factor , out_channels//reduce_factor, kernel_size=3, padding=1),
                        nn.Conv2d(out_channels//reduce_factor , out_channels               , kernel_size=1, padding=0),
                        ),            
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
        
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = DecoderBottleneckLayer(in_channels=in_channels, out_channels=in_channels//2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class U(nn.Module):
    def __init__(self, n_channels=1, n_classes=9):
        super(U, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        in_channels = 32
        
        self.inc = DoubleConv_(n_channels, in_channels)
        
        self.down1 = Down(in_channels*1, in_channels*2 ) # 64
        self.down2 = Down(in_channels*2, in_channels*4 ) # 128
        self.down3 = Down(in_channels*4, in_channels*8 ) # 256
        self.down4 = Down(in_channels*8, in_channels*16) # 512
        
        self.up1 = Up(in_channels*16, in_channels*8)
        self.up2 = Up(in_channels*8 , in_channels*4)
        self.up3 = Up(in_channels*4 , in_channels*2)

        self.final_conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x3 = self.up1(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up3(x2, x1)

        x = self.final_conv1(x1)
        x = self.final_relu1(x)
        x = self.final_conv2(x)
        x = self.final_relu2(x)
        x = self.final_conv3(x)
        
        return x
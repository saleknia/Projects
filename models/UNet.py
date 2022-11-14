import torch.nn as nn
import torch
import timm

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
                                ConvBatchNorm(in_channels, in_channels//2, activation='ReLU', kernel_size=1, padding=0),
                                nn.Upsample(scale_factor=2)   
                                )
        nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = _make_nConv(in_channels, out_channels, nb_Conv, activation)
    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        in_channels = 64
        self.encoder = timm.create_model('hrnet_w18_small', pretrained=True, features_only=True)
        self.encoder.conv1.stride = (1, 1)

        # torch.Size([8, 64, 112, 112])
        # torch.Size([8, 128, 56, 56])
        # torch.Size([8, 256, 28, 28])
        # torch.Size([8, 512, 14, 14])
        # torch.Size([8, 1024, 7, 7])

        self.up4 = UpBlock(1024, 512, nb_Conv=2)
        self.up3 = UpBlock(512 , 256, nb_Conv=2)
        self.up2 = UpBlock(256 , 128, nb_Conv=2)
        self.up1 = UpBlock(128 , 64 , nb_Conv=2)

        self.final_conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        # Question here
        x = x.float()
        x1, x2, x3, x4, x5 = self.encoder(x)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        x = self.final_conv1(x)
        x = self.final_relu1(x)
        x = self.final_conv2(x)
        x = self.final_relu2(x)
        out = self.final_conv3(x)

        return out











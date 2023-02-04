from torchvision import models as resnet_model
import torch.nn as nn
import torch

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', reduce=False, reduction_rate=1):
    layers = []
    if reduce:
        layers.append(ConvBatchNorm_r(in_channels, out_channels, activation, reduction_rate))
    else:
        layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        if reduce:
            layers.append(ConvBatchNorm_r(out_channels, out_channels, activation, reduction_rate))
        else:
            layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class ConvBatchNorm_r(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', reduction_rate=1):
        super(ConvBatchNorm_r, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels//reduction_rate,kernel_size=3, padding=1)
        self.norm_1 = nn.BatchNorm2d(out_channels//reduction_rate)
        self.activation_1 = get_activation(activation)

        self.conv_2 = nn.Conv2d(out_channels//reduction_rate, out_channels, kernel_size=1, padding=0)
        self.norm_2 = nn.BatchNorm2d(out_channels)
        self.activation_2 = get_activation(activation)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.activation_2(x)
        return x

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', reduce=False, reduction_rate=1):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, reduce=reduce, reduction_rate=reduction_rate)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', reduce=False, reduction_rate=1):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, reduce=reduce, reduction_rate=reduction_rate)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class SEUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.maxpool   = 
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.

        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print("111")
        else:
            logits = self.outc(x)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits


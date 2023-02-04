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
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, reduce=reduce, reduction_rate=reduction_rate)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class CSFR(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.down     = nn.Upsample(scale_factor=0.5)
        self.up       = nn.Upsample(scale_factor=2.0)
        self.final_up = nn.Upsample(scale_factor=2.0)

        self.conv_up   = _make_nConv(in_channels=channels, out_channels=channels, nb_Conv=2, activation='ReLU', dilation=1, padding=1)
        self.conv_down = _make_nConv(in_channels=channels, out_channels=channels, nb_Conv=2, activation='ReLU', dilation=1, padding=1)
        self.conv_fuse = _make_nConv(in_channels=channels, out_channels=channels, nb_Conv=2, activation='ReLU', dilation=1, padding=1)

        self.reduction = ConvBatchNorm(in_channels=2*channels, out_channels=channels, activation='ReLU', kernel_size=1, padding=0, dilation=1)

    def forward(self, x1, x2):

        x2 = self.reduction(x2)

        x1_d  = self.down(x1)
        x2_up = self.up(x2)

        up   = self.conv_up(x1+x2_up)
        down = self.conv_down(x1_d+x2)
        down = self.final_up(down)

        x = self.conv_fuse(down + up)

        return x

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
        self.firstbn   = resnet.bn1
        self.firstrelu = resnet.relu
        self.maxpool   = resnet.maxpool 
        self.encoder1  = resnet.layer1
        self.encoder2  = resnet.layer2
        self.encoder3  = resnet.layer3
        self.encoder4  = resnet.layer4

        self.up3 = UpBlock(in_channels=512, out_channels=256, nb_Conv=2)
        self.up2 = UpBlock(in_channels=256, out_channels=128, nb_Conv=2)
        self.up1 = UpBlock(in_channels=128, out_channels=64 , nb_Conv=2)

        self.CSFR_3 = CSFR(256)
        self.CSFR_2 = CSFR(128)
        self.CSFR_1 = CSFR(64)

        self.final_conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.ConvTranspose2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = torch.cat([x, x, x], dim=1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e0 = self.maxpool(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e3 = self.CSFR_3(e3, e4)
        e2 = self.CSFR_2(e2, e3)
        e1 = self.CSFR_1(e1, e2)

        e = self.up3(e4, e3)
        e = self.up2(e , e2)
        e = self.up1(e , e1)

        e = self.final_conv1(e)
        e = self.final_relu1(e)
        e = self.final_conv2(e)
        e = self.final_relu2(e)
        e = self.final_conv3(e)

        return e










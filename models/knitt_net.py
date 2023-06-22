import math
import torch
import random
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import einops
import timm
from torchvision import models as resnet_model
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

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


class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv=2, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up     = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class knitt(nn.Module):
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

        self.convnext_v2 = knitt_net_conv()

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/conv.pth', map_location='cuda')
        pretrained_teacher = loaded_data_teacher['net']
        self.convnext_v2.load_state_dict(pretrained_teacher)

        self.transformer = knitt_net_trans()

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/trans.pth', map_location='cuda')
        pretrained_teacher = loaded_data_teacher['net']
        self.transformer.load_state_dict(pretrained_teacher)

    def forward(self, x):
        # # Question here
        x_input = x.float()
        B, C, H, W = x.shape

        x = self.convnext_v2(x_input)
        e = self.transformer(x_input)

        return e + x


class knitt_net(nn.Module):
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

        self.convnext_v2 = timm.create_model('convnext_tiny', features_only=True, pretrained=False)

        self.up2_1 = UpBlock(in_channels=384, out_channels=192)
        self.up1_1 = UpBlock(in_channels=192, out_channels=96 )

        self.head_1 = nn.Sequential(nn.Conv2d(96,  self.n_classes, 1, padding=0), nn.Upsample(scale_factor=4.0))

    def forward(self, x):
        # # Question here
        x_input = x.float()
        B, C, H, W = x.shape

        x0, x1, x2, x3 = self.convnext_v2(x_input)

        x1 = self.up2_1(x2, x1)
        x0 = self.up1_1(x1, x0)

        x = self.head_1(x0)

        return x



class knitt_net_trans(nn.Module):
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

        self.transformer = timm.create_model('maxvit_tiny_rw_224', pretrained=True, features_only=True)

        self.up2_2 = UpBlock(in_channels=256, out_channels=128)
        self.up1_2 = UpBlock(in_channels=128, out_channels=64 )

        self.head_2 = nn.Sequential(nn.Conv2d(64,  self.n_classes, 1, padding=0), nn.Upsample(scale_factor=4.0))

    def forward(self, x):
        # # Question here
        x_input = x.float()
        B, C, H, W = x.shape


        e0, e1, e2, e3, e4 = self.transformer(x_input)

        e2 = self.up2_2(e3, e2)
        e1 = self.up1_2(e2, e1)

        e = self.head_2(e1)

        return e





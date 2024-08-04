import torch
from torchvision import models as resnet_model
from torch import nn
import timm
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
import ml_collections


class final_head(nn.Module):
    def __init__(self, num_classes=1, scale_factor=2.0):
        super(final_head, self).__init__()

        self.head = nn.Sequential(
            nn.ConvTranspose2d(48, 48, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, num_classes, 4, 2, 1), 
        )

    def forward(self, x):
        out = self.head(x)
        return out

class cnn_decoder(nn.Module):
    def __init__(self, num_classes=1.0, scale_factor=2.0):
        super(cnn_decoder, self).__init__()
        
        self.up_2 = UpBlock(384, 192)
        self.up_1 = UpBlock(192, 96)
        self.up_0 = UpBlock(96 , 48)

        self.final_head = final_head(num_classes=1, scale_factor=2.0)

    def forward(self, x0, x1, x2, x3):

        x = self.up_2(x3, x2)
        x = self.up_1(x1, x)
        x = self.up_0(x0, x)

        x = self.final_head(x)

        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn   = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Linear_Eca_block(nn.Module):
    """docstring for Eca_block"""
    def __init__(self):
        super(Linear_Eca_block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1d  = nn.Conv1d(1, 1, kernel_size=5, padding=int(5/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gamma=2, b=1):
        #N, C, H, W = x.size()
        y = self.avgpool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)


class HybridAttention(nn.Module):
    def __init__(self, channels):
        super(HybridAttention, self).__init__()

        self.eca     = Linear_Eca_block()
        self.conv    = BasicConv2d(channels, channels, 3, 1, 1)
        self.down_c  = BasicConv2d(channels, 1, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.final_conv = ConvBatchNorm(in_channels=channels*2 , out_channels=channels, activation='ReLU', kernel_size=1, padding=0)

    def forward(self, x_t, x_c):

        sa  = self.sigmoid(self.down_c(x_c))
        gc  = self.eca(x_t)
        x_c = self.conv(x_c)
        x_c = x_c * gc
        x_t = x_t * sa
        x = self.final_conv(torch.cat((x_t, x_c), 1))
        
        return x

class knitt_net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(knitt_net, self).__init__()

        self.model = timm.create_model('timm/efficientvit_b2.r224_in1k', pretrained=True, features_only=True)
        self.cnn_decoder = cnn_decoder()

        # self.reduce_0 = ConvBatchNorm(in_channels=48 , out_channels=48, activation='ReLU', kernel_size=1, padding=0)
        # self.reduce_1 = ConvBatchNorm(in_channels=96 , out_channels=48, activation='ReLU', kernel_size=1, padding=0)
        # self.reduce_2 = ConvBatchNorm(in_channels=192, out_channels=48, activation='ReLU', kernel_size=1, padding=0)
        # self.reduce_3 = ConvBatchNorm(in_channels=384, out_channels=48, activation='ReLU', kernel_size=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape

        t0, t1, t2, t3 = self.model(x)

        # t0 = self.reduce_0(t0)
        # t1 = self.reduce_1(t1)
        # t2 = self.reduce_2(t2)
        # t3 = self.reduce_3(t3)

        out = self.cnn_decoder(t0, t1, t2, t3)

        return out


import torch.nn as nn
import torch

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
                              kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv=2, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up     = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(out_channels*2, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return (self.nConvs(x))

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')








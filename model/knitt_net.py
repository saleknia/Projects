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
from efficientvit.seg_model_zoo import create_seg_model

class final_head(nn.Module):
    def __init__(self, base_channel=64, num_classes=1, scale_factor=2.0):
        super(final_head, self).__init__()
        
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channel, base_channel//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(base_channel//2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channel//2, num_classes, kernel_size=3, stride=1, padding=1, bias=True),
        )
 

    def forward(self, x):
        out = self.head(x)
        return out

class cnn_decoder(nn.Module):
    def __init__(self, base_channel=64, num_classes=1.0, scale_factor=2.0):
        super(cnn_decoder, self).__init__()
        
        self.up_2 = UpBlock(base_channel*8, base_channel*4)
        self.up_1 = UpBlock(base_channel*4, base_channel*2)
        self.up_0 = UpBlock(base_channel*2, base_channel*1)

        self.final_head = final_head(base_channel=base_channel, num_classes=1, scale_factor=2)

        # self.fusion = SAM(base_channel=base_channel)

    def forward(self, x0, x1, x2, x3):
        
        d3 = self.up_2(x3, x2)
        d2 = self.up_1(d3, x1)
        d1 = self.up_0(d2, x0)

        # d = self.fusion(d1, d2, d3)

        x = self.final_head(d1)

        return x

class SAM(nn.Module):
    def __init__(self, base_channel):
        super(SAM, self).__init__()

        self.conv_3 = BasicConv2d(base_channel*4, base_channel*1, 1, 1, 0)
        self.conv_2 = BasicConv2d(base_channel*2, base_channel*1, 1, 1, 0)
        self.conv_1 = BasicConv2d(base_channel*1, base_channel*1, 1, 1, 0)

        self.up_2 = nn.Upsample(scale_factor=2)
        self.up_4 = nn.Upsample(scale_factor=4)

        self.down   = BasicConv2d(base_channel*3, 3, 1, 1, padding=0)

        self.softmax = nn.Softmax(dim=1)
        self.relu    = nn.ReLU()

    def forward(self, d1, d2, d3):
        
        d1 = self.relu(self.conv_1(d1))
        d2 = self.up_2(self.relu(self.conv_2(d2)))
        d3 = self.up_4(self.relu(self.conv_3(d3)))

        d = torch.cat([d1, d2, d3], dim=1)
        att = self.down(d)
        att = self.softmax(att)

        att1 = att[:,0,:,:].unsqueeze(1)
        att2 = att[:,1,:,:].unsqueeze(1)
        att3 = att[:,2,:,:].unsqueeze(1)

        x = (att1 * d1) + (att2 * d2) + (att3 * d3)

        return x

class SAWM(nn.Module):
    def __init__(self, channels):
        super(SAWM, self).__init__()

        self.conv_3 = BasicConv2d(base_channel*4, base_channel*1, 1, 1, 0)
        self.conv_2 = BasicConv2d(base_channel*2, base_channel*1, 1, 1, 0)
        self.conv_1 = BasicConv2d(base_channel*1, base_channel*1, 1, 1, 0)

        self.up_2 = nn.Upsample(scale_factor=2)
        self.up_4 = nn.Upsample(scale_factor=4)

        self.down   = BasicConv2d(base_channel*3, 3, 1, 1, padding=0)

        self.softmax = nn.Softmax(dim=1)
        self.relu    = nn.ReLU()

    def forward(self, d1, d2, d3):
        
        d1 = self.relu(self.conv_1(d1))
        d2 = self.up_2(self.relu(self.conv_2(d2)))
        d3 = self.up_4(self.relu(self.conv_3(d3)))

        d = torch.cat([d1, d2, d3], dim=1)
        att = self.down(d)
        att = self.softmax(att)

        att1 = att[:,0,:,:].unsqueeze(1)
        att2 = att[:,1,:,:].unsqueeze(1)
        att3 = att[:,2,:,:].unsqueeze(1)

        x = (att1 * d1) + (att2 * d2) + (att3 * d3)

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

    def forward(self, x, skip_x):

        sa  = self.sigmoid(self.down_c(skip_x))
        gc  = self.eca(x)
        skip_x = self.conv(skip_x)
        skip_x = skip_x * gc
        x = x * sa
        return x, skip_x

class knitt_net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(knitt_net, self).__init__()

        self.encoder     = timm.create_model('timm/efficientvit_b2.r224_in1k', pretrained=True, features_only=True)
        self.cnn_decoder = cnn_decoder(base_channel=48)       

        base_channel = 48

        self.conv_3 = BasicConv2d(base_channel*16, base_channel*8, 1, 1, 0)
        self.conv_2 = BasicConv2d(base_channel*8 , base_channel*4, 1, 1, 0)
        self.conv_1 = BasicConv2d(base_channel*4 , base_channel*2, 1, 1, 0) 
        self.conv_0 = BasicConv2d(base_channel*2 , base_channel*1, 1, 1, 0) 

        self.avg = nn.AvgPool2d(2, stride=2)

        self.up = nn.Upsample(scale_factor=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape

        x0, x1, x2, x3 = self.encoder(x)
        t0, t1, t2, t3 = self.encoder(self.avg(x)) 

        t0 = self.up(t0)    
        t1 = self.up(t1)    
        t2 = self.up(t2)    
        t3 = self.up(t3)    

        x0 = self.conv_0(torch.cat([x0, t0], dim=1))
        x1 = self.conv_1(torch.cat([x1, t1], dim=1))
        x2 = self.conv_2(torch.cat([x2, t2], dim=1))
        x3 = self.conv_3(torch.cat([x3, t3], dim=1))
        
        out = self.cnn_decoder(x0, x1, x2, x3)

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

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        x = self.up(x)
        # x, skip_x = self.fusion(x, skip_x)
        x = torch.cat([x, skip_x], dim=1)  # dim 1 is the channel dimension
        return (self.nConvs(x))

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')








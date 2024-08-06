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
    def __init__(self, base_channel=96, num_classes=1, scale_factor=2.0):
        super(final_head, self).__init__()

        self.head = nn.Sequential(
            nn.ConvTranspose2d(base_channel, base_channel//2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel//2, base_channel//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channel//2, num_classes, 4, 2, 1), 
        )

    def forward(self, x):
        out = self.head(x)
        return out

class cnn_decoder(nn.Module):
    def __init__(self, base_channel=96, num_classes=1.0, scale_factor=2.0):
        super(cnn_decoder, self).__init__()

        self.up_1 = UpBlock(base_channel*4, base_channel*2)
        self.up_0 = UpBlock(base_channel*2, base_channel*1)

        self.final_head = final_head(base_channel=base_channel, num_classes=1, scale_factor=2)

    def forward(self, x0, x1, x2):
        
        x = self.up_1(x2, x1)
        x = self.up_0(x , x0)

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

        # self.eca     = Linear_Eca_block()
        # self.conv    = BasicConv2d(channels, channels, 3, 1, 1)
        # self.down_c  = BasicConv2d(channels, 1, 3, 1, padding=1)
        # self.sigmoid = nn.Sigmoid()
        self.final_conv = ConvBatchNorm(in_channels=channels*2 , out_channels=channels, activation='ReLU', kernel_size=1, padding=0)

    def forward(self, x_t, x_c):

        # sa  = self.sigmoid(self.down_c(x_c))
        # gc  = self.eca(x_t)
        # x_c = self.conv(x_c)
        # x_c = x_c * gc
        # x_t = x_t * sa
        x = self.final_conv(torch.cat((x_t, x_c), 1))
        
        return x

class knitt_net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(knitt_net, self).__init__()

        # self.cls = timm.create_model('timm/maxvit_tiny_tf_224.in1k', pretrained=True, features_only=True)
        # self.cnn_decoder = cnn_decoder()
        # self.up  = nn.Upsample(scale_factor=2.0)
        
        # model = create_seg_model(name="b2", dataset="ade20k", weight_url="/content/drive/MyDrive/b2.pt").backbone

        # model.input_stem.op_list[0].conv.stride  = (1, 1)
        # model.input_stem.op_list[0].conv.padding = (1, 1)

        # self.seg = model

        # self.HA_3 = HybridAttention(channels=384)
        # self.HA_2 = HybridAttention(channels=192)
        # self.HA_1 = HybridAttention(channels=96)
        # self.HA_0 = HybridAttention(channels=48)

        # self.reduce_0 = ConvBatchNorm(in_channels=64 , out_channels=48 , activation='ReLU', kernel_size=1, padding=0)
        # self.reduce_1 = ConvBatchNorm(in_channels=128, out_channels=96 , activation='ReLU', kernel_size=1, padding=0)
        # self.reduce_2 = ConvBatchNorm(in_channels=256, out_channels=192, activation='ReLU', kernel_size=1, padding=0)
        # self.reduce_3 = ConvBatchNorm(in_channels=512, out_channels=384, activation='ReLU', kernel_size=1, padding=0)

        self.enc_0       = timm.create_model('convnext_tiny', pretrained=True, features_only=True, out_indices=[0,1,2])
        self.cnn_decoder = cnn_decoder(base_channel=96)

        self.enc_1 = timm.create_model('convnext_tiny', pretrained=True, features_only=True, out_indices=[0,1,2])
        self.enc_1.stem_0.stride  = (2, 2) 
        self.enc_1.stem_0.padding = (2, 2) 


        self.avgpool = nn.AvgPool2d(2, stride=2)

        self.fusion_0 = ConvBatchNorm(in_channels=192, out_channels=96 , activation='ReLU', kernel_size=1, padding=0)
        self.fusion_1 = ConvBatchNorm(in_channels=384, out_channels=192, activation='ReLU', kernel_size=1, padding=0)
        self.fusion_2 = ConvBatchNorm(in_channels=768, out_channels=384, activation='ReLU', kernel_size=1, padding=0)



    def forward(self, x):
        b, c, h, w = x.shape

        # stem, t0, t1, t2, t3 = self.cls(x)

        # t0 = self.up(self.reduce_0(t0))
        # t1 = self.up(self.reduce_1(t1))
        # t2 = self.up(self.reduce_2(t2))
        # t3 = self.up(self.reduce_3(t3))

        # y = self.seg(x)
        # s0, s1, s2, s3 = y['stage1'], y['stage2'], y['stage3'], y['stage4']

        # x0 = self.HA_0(t0, s0)        
        # x1 = self.HA_1(t1, s1)        
        # x2 = self.HA_2(t2, s2)
        # x3 = self.HA_3(t3, s3)

        # out = self.cnn_decoder(x0, x1, x2, x3)

        # out = self.cnn_decoder(s0, s1, s2, s3)


        # t0 = self.up(t0)
        # t1 = self.up(t1)
        # t2 = self.up(t2)
        # t3 = self.up(t3)

        # out = self.cnn_decoder(t0, t1, t2, t3)
        
        t0, t1, t2 = self.enc_0(x)

        s0, s1, s2 = self.enc_1(x)

        s0 = self.avgpool(s0)
        s1 = self.avgpool(s1)
        s2 = self.avgpool(s2)
        
        x0 = self.fusion_0(torch.cat([s0, t0], dim=1))
        x1 = self.fusion_1(torch.cat([s1, t1], dim=1))
        x2 = self.fusion_2(torch.cat([s2, t2], dim=1))
       
        out = self.cnn_decoder(x0, x1, x2)

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
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

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








from torchvision import models as resnet_model
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Softmax
import einops
import timm

import numpy as np
import torch
from torch import nn
from torch.nn import init

class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        return out

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

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)

        self.up     = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

        self.att    = ParallelPolarizedSelfAttention(channel=in_channels//2)

    def forward(self, x, skip_x):
        out = self.up(x)
        out = self.att(out)
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
        self.firstbn   = resnet.bn1
        self.firstrelu = resnet.relu
        # self.maxpool   = resnet.maxpool 
        self.encoder1  = resnet.layer1
        self.encoder2  = resnet.layer2
        self.encoder3  = resnet.layer3
        self.encoder4  = resnet.layer4

        self.up3 = UpBlock(in_channels=512, out_channels=256, nb_Conv=2)
        self.up2 = UpBlock(in_channels=256, out_channels=128, nb_Conv=2)
        self.up1 = UpBlock(in_channels=128, out_channels=64 , nb_Conv=2)
        
        # self.final_conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        # self.final_relu1 = nn.ReLU(inplace=True)
        # self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.final_relu2 = nn.ReLU(inplace=True)

        # self.final_conv3 = nn.ConvTranspose2d(32, n_classes, kernel_size=2, stride=2)
        
        # self.final_conv3 = nn.Conv2d(32, n_classes, 1, padding=0)

        self.final_conv =  nn.ConvTranspose2d(64, n_classes, (2,2), 2)

        # self.final_up   = nn.Upsample(scale_factor=4.0)


    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.cat([x, x, x], dim=1)


        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        # e0 = self.maxpool(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e = self.up3(e4, e3)
        e = self.up2(e , e2)
        e = self.up1(e , e1)

        # e = self.final_conv1(e)
        # e = self.final_relu1(e)
        # e = self.final_conv2(e)
        # e = self.final_relu2(e)
        # e = self.final_conv3(e)

        e = self.final_conv(e)

        return e
        





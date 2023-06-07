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

        # resnet = resnet_model.resnet34(pretrained=True)

        model = torchvision.models.convnext_tiny(weights='DEFAULT').features

        # self.firstconv = resnet.conv1
        # self.firstbn   = resnet.bn1
        # self.firstrelu = resnet.relu
        # # self.maxpool   = resnet.maxpool 
        # self.encoder1  = resnet.layer1
        # self.encoder2  = resnet.layer2
        # self.encoder3  = resnet.layer3
        # self.encoder4  = resnet.layer4

        self.encoder1  = model[0:2]
        self.encoder2  = model[2:4]
        self.encoder3  = model[4:6]
        self.encoder4  = model[6:8]

        self.up3 = UpBlock(in_channels=768, out_channels=384, nb_Conv=2)
        self.up2 = UpBlock(in_channels=384, out_channels=192, nb_Conv=2)
        self.up1 = UpBlock(in_channels=192, out_channels=96 , nb_Conv=2)
        
        self.final_conv1 = nn.ConvTranspose2d(96, 48, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(48, 48, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.ConvTranspose2d(48, 2, kernel_size=2, stride=2)

        # self.final_conv3 = nn.Conv2d(32, n_classes, 1, padding=0)

        # self.final_conv =  nn.ConvTranspose2d(64, n_classes, (2,2), 2)

        # self.final_up   = nn.Upsample(scale_factor=4.0)


    def forward(self, x):

        # x = x.unsqueeze(dim=1)
        
        b, c, h, w = x.shape
        e0 = torch.cat([x, x, x], dim=1)


        # e0 = self.firstconv(x)
        # e0 = self.firstbn(e0)
        # e0 = self.firstrelu(e0)
        # # e0 = self.maxpool(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e = self.up3(e4, e3)
        e = self.up2(e , e2)
        e = self.up1(e , e1)

        e = self.final_conv1(e)
        e = self.final_relu1(e)
        e = self.final_conv2(e)
        e = self.final_relu2(e)
        e = self.final_conv3(e)

        # e = self.final_conv(e)

        return e
        





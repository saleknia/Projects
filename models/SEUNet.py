from torchvision import models as resnet_model
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Softmax
import einops

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


    def forward(self, x, skip_x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x + skip_x

class decoder_1(nn.Module):
    def __init__(self):
        super(decoder_1, self).__init__()

        self.up3 = DecoderBottleneckLayer(in_channels=512, out_channels=256)
        self.up2 = DecoderBottleneckLayer(in_channels=256, out_channels=128)
        self.up1 = DecoderBottleneckLayer(in_channels=128, out_channels=64 )

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
                
        self.tp_conv2 = nn.Conv2d(32, 1, 1, 1, 0)

    def forward(self, e1, e2, e3, e4):
        e = self.up3(e4, e3) 
        e = self.up2(e , e2) 
        e = self.up1(e , e1)

        e = self.tp_conv1(e)
        e = self.conv2(e)
        e = self.tp_conv2(e)

        return e

class decoder_2(nn.Module):
    def __init__(self):
        super(decoder_2, self).__init__()

        self.up3 = UpBlock_2(in_channels=512, out_channels=256)
        self.up2 = UpBlock_2(in_channels=256, out_channels=128)
        self.up1 = UpBlock_2(in_channels=128, out_channels=64 )

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
                
        self.tp_conv2 = nn.Conv2d(32, 1, 1, 1, 0)

    def forward(self, e1, e2, e3, e4):
        e = self.up3(e4, e3) 
        e = self.up2(e , e2) 
        e = self.up1(e , e1)

        e = self.tp_conv1(e)
        e = self.conv2(e)
        e = self.tp_conv2(e)

        return e


class decoder_3(nn.Module):
    def __init__(self):
        super(decoder_3, self).__init__()

        self.up3 = UpBlock_2(in_channels=512, out_channels=256)
        self.up2 = UpBlock_2(in_channels=256, out_channels=128)
        self.up1 = UpBlock_2(in_channels=128, out_channels=64 )

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
                
        self.tp_conv2 = nn.Conv2d(32, 1, 1, 1, 0)

    def forward(self, e1, e2, e3, e4):
        e = self.up3(e4, e3) 
        e = self.up2(e , e2) 
        e = self.up1(e , e1)

        e = self.tp_conv1(e)
        e = self.conv2(e)
        e = self.tp_conv2(e)

        return e

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

        for param in resnet.parameters():
            param.requires_grad = False

        self.firstconv = resnet.conv1
        self.firstbn   = resnet.bn1
        self.firstrelu = resnet.relu
        # self.maxpool   = resnet.maxpool 
        self.encoder1  = resnet.layer1
        self.encoder2  = resnet.layer2
        self.encoder3  = resnet.layer3
        self.encoder4  = resnet.layer4

        # self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
        #                               nn.BatchNorm2d(32),
        #                               nn.ReLU(inplace=True),)
        # self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
        #                         nn.BatchNorm2d(32),
        #                         nn.ReLU(inplace=True),)
        
        # # self.tp_conv2 = nn.ConvTranspose2d(32, 1, 2, 2, 0)
        
        # self.tp_conv2 = nn.Conv2d(32, 1, 1, 1, 0)

        self.decoder_1 = decoder_1()
        
        for param in self.decoder_1.parameters():
            param.requires_grad = False

        self.decoder_2 = decoder_2()

        for param in self.decoder_2.parameters():
            param.requires_grad = False

        self.decoder_3 = decoder_3()

        # for param in self.decoder_3.parameters():
        #     param.requires_grad = False
             
    def forward(self, x):
        b, c, h, w = x.shape
        # x = torch.cat([x, x, x], dim=1)
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        # e0 = self.maxpool(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)


        # x = self.decoder_1(e1, e2, e3, e4)
        # y = self.decoder_2(e1, e2, e3, e4)
        z = self.decoder_3(e1, e2, e3, e4)

        return z


def get_activation(activation_type):  
    if activation_type=='Sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', dilation=1, padding=0):
    layers = []
    layers.append(ConvBatchNorm(in_channels=in_channels, out_channels=out_channels, activation=activation, dilation=dilation, padding=padding))

    for i in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(in_channels=out_channels, out_channels=out_channels, activation=activation, dilation=dilation, padding=padding))
    return nn.Sequential(*layers)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SEBlock(nn.Module):
    def __init__(self, channel, r=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, skip_x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        skip_x = torch.mul(skip_x, y)
        return skip_x

class UpBlock_2(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv=2, activation='ReLU'):
        super(UpBlock_2, self).__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = _make_nConv(in_channels=in_channels//2, out_channels=out_channels, nb_Conv=2, activation='ReLU', dilation=1, padding=1)
    
    def forward(self, x, skip_x):
        x = self.up(x) 
        x = x + skip_x
        x = self.conv(x)
        return x 

class UpBlock_3(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv=2, activation='ReLU', img_size=224):
        super(UpBlock_3, self).__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = _make_nConv(in_channels=in_channels, out_channels=out_channels, nb_Conv=2, activation='ReLU', dilation=1, padding=1)
    
    def forward(self, x, skip_x):
        x = self.up(x) 
        x = torch.cat([x, skip_x], dim=1)  # dim 1 is the channel dimension
        x = self.conv(x)
        return x 

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1, dilation=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', dilation=1, padding=0):
    layers = []
    layers.append(ConvBatchNorm(in_channels=in_channels, out_channels=out_channels, activation=activation, dilation=dilation, padding=padding))

    for i in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(in_channels=out_channels, out_channels=out_channels, activation=activation, dilation=dilation, padding=padding))
    return nn.Sequential(*layers)


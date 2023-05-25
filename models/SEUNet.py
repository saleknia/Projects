from torchvision import models as resnet_model
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Softmax


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

        # self.SK = SKAttention(out_channels)

    def forward(self, x, skip_x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        # return self.SK(x, skip_x)
        return x + skip_x

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
        
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

        self.teacher = SEUNet_teacher()
        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_85_17/SEUNet_ISIC2017_best.pth', map_location='cuda')
        pretrained_teacher = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.teacher.load_state_dict(pretrained_teacher)

        for param in self.teacher.parameters():
            param.requires_grad = False

        model = torchvision.models.regnet_x_800mf(weights='DEFAULT')

        self.stem   = model.stem
        self.layer1 = model.trunk_output.block1
        self.layer2 = model.trunk_output.block2
        self.layer3 = model.trunk_output.block3
        self.layer4 = model.trunk_output.block4

        self.up3 = DecoderBottleneckLayer(in_channels=672, out_channels=288)
        self.up2 = DecoderBottleneckLayer(in_channels=288, out_channels=128)
        self.up1 = DecoderBottleneckLayer(in_channels=128, out_channels=64)

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, 1, 2, 2, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        # x = torch.cat([x, x, x], dim=1)

        y_t, e1_t, e2_t, e3_t = self.teacher(x)

        x = self.stem(x)

        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        e3 = self.up3(e4, e3) 
        e2 = self.up2(e3, e2) 
        e1 = self.up1(e2, e1)

        y = self.tp_conv1(e1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        if self.training:
            return y, y_t, e1, e2, e3, e1_t, e2_t, e3_t
        else:
            return y

class SEUNet_teacher(nn.Module):
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

        model = torchvision.models.regnet_x_3_2gf(weights='DEFAULT')

        self.stem   = model.stem
        self.layer1 = model.trunk_output.block1
        self.layer2 = model.trunk_output.block2
        self.layer3 = model.trunk_output.block3
        self.layer4 = model.trunk_output.block4

        self.up3 = DecoderBottleneckLayer(in_channels=1008, out_channels=432)
        self.up2 = DecoderBottleneckLayer(in_channels=432 , out_channels=192)
        self.up1 = DecoderBottleneckLayer(in_channels=192 , out_channels=96)

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(96, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, 1, 2, 2, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        # x = torch.cat([x, x, x], dim=1)

        x = self.stem(x)

        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        e3 = self.up3(e4, e3) 
        e2 = self.up2(e3, e2) 
        e1 = self.up1(e2, e1)

        y = self.tp_conv1(e1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = torch.round(torch.sigmoid(torch.squeeze(y, dim=1)))

        return y, e1, e2, e3




      
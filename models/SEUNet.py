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

        # self.teacher = SEUNet_teacher()
        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_85_17/SEUNet_ISIC2017_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.teacher.load_state_dict(pretrained_teacher)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        # model = torchvision.models.regnet_y_400mf(weights='DEFAULT')

        # self.stem   = model.stem
        # self.layer1 = model.trunk_output.block1
        # self.layer2 = model.trunk_output.block2
        # self.layer3 = model.trunk_output.block3
        # self.layer4 = model.trunk_output.block4

        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.maxpool  = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.up3 = DecoderBottleneckLayer(in_channels=512, out_channels=256)
        self.up2 = DecoderBottleneckLayer(in_channels=256, out_channels=128)
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

        # y_t, e1_t, e2_t, e3_t = self.teacher(x)

        # x = self.stem(x)

        # e1 = self.layer1(x)
        # e2 = self.layer2(e1)
        # e3 = self.layer3(e2)
        # e4 = self.layer4(e3)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e0 = self.maxpool(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e = self.up3(e4, e3) 
        e = self.up2(e , e2) 
        e = self.up1(e , e1)

        y = self.tp_conv1(e)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        # y = (y + y_t) / 2.0


        if self.training:
            return y#, y_t, e1, e2, e3, e1_t, e2_t, e3_t
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

        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.maxpool  = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.up3 = DecoderBottleneckLayer(in_channels=512, out_channels=256)
        self.up2 = DecoderBottleneckLayer(in_channels=256, out_channels=128)
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

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e0 = self.maxpool(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e = self.up3(e4, e3) 
        e = self.up2(e , e2) 
        e = self.up1(e , e1)

        y = self.tp_conv1(e)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y, e1, e2, e3




      
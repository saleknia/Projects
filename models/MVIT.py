import torch
from torchvision import models as resnet_model
from torch import nn
import timm

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
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

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class MVIT(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(MVIT, self).__init__()

        self.transformer = MVIT_V2()

        filters = [96, 192, 384, 768]

        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 48, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(48, 48, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(48, n_classes, 3, padding=1)


    def forward(self, x):
        b, c, h, w = x.shape

        x1, x2, x3, x4 = self.transformer(x)

        d4 = self.decoder4(x4) + x3
        d3 = self.decoder3(d4) + x2
        d2 = self.decoder2(d3) + x1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out

class MVIT_V2(torch.nn.Module):
    def __init__(self):
        super(MVIT_V2, self).__init__()

        self.patch_embed = timm.create_model('mvitv2_tiny').patch_embed
        self.stage_0     = timm.create_model('mvitv2_tiny').stages[0]
        self.stage_1     = timm.create_model('mvitv2_tiny').stages[1]
        self.stage_2     = timm.create_model('mvitv2_tiny').stages[2]
        self.stage_3     = timm.create_model('mvitv2_tiny').stages[3]

    def forward(self, x):

        batch_size, _, H, W = x.shape

        x0, feat_size = self.patch_embed(x)
        x1, feat_size = self.stage_0(x0, feat_size)
        x2, feat_size = self.stage_1(x1, feat_size)
        x3, feat_size = self.stage_2(x2, feat_size)
        x4, feat_size = self.stage_3(x3, feat_size)

        x1 = x1.reshape(batch_size, H//4 , W//4 , -1)
        x2 = x2.reshape(batch_size, H//8 , W//8 , -1)
        x3 = x3.reshape(batch_size, H//16, W//16, -1)
        x4 = x4.reshape(batch_size, H//32, W//32, -1)

        return x1, x2, x3, x4



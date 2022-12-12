import torch.nn as nn
import torch
from torchvision import models as resnet_model
import torch.nn.functional as F

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', dilation=1, padding=0):
    layers = []
    layers.append(ConvBatchNorm(in_channels=in_channels, out_channels=out_channels, activation=activation, dilation=dilation, padding=padding))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(in_channels=out_channels, out_channels=out_channels, activation=activation, dilation=dilation, padding=padding))
    return nn.Sequential(*layers)

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

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1

        return out

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

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = _make_nConv(in_channels=(in_channels//2)+out_channels, out_channels=out_channels, nb_Conv=nb_Conv, activation=activation, dilation=1, padding=1)

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)  # dim 1 is the channel dimension
        x = self.conv(x)
        return x

class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[64, 128, 256, 512], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class SEUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.counter = 0
        # Question here

        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #####################################################################
        #####################################################################
        #####################################################################

        self.FAMBlock1_1 = FAMBlock(channels=64)
        self.FAMBlock2_1 = FAMBlock(channels=128)
        self.FAMBlock3_1 = FAMBlock(channels=256)
        self.FAM1_1 = nn.ModuleList([self.FAMBlock1_1 for i in range(6)])
        self.FAM2_1 = nn.ModuleList([self.FAMBlock2_1 for i in range(4)])
        self.FAM3_1 = nn.ModuleList([self.FAMBlock3_1 for i in range(2)])

        self.up3_1 = UpBlock(512, 256, nb_Conv=2)
        self.up2_1 = UpBlock(256, 128, nb_Conv=2)
        self.up1_1 = UpBlock(128, 64 , nb_Conv=2)

        self.final_conv1_1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.final_relu1_1 = nn.ReLU(inplace=True)
        self.final_conv2_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2_1 = nn.ReLU(inplace=True)
        self.final_conv3_1 = nn.Conv2d(32, n_classes, 3, padding=1)

        #####################################################################
        #####################################################################
        #####################################################################

        self.FAMBlock1_2 = FAMBlock(channels=64)
        self.FAMBlock2_2 = FAMBlock(channels=128)
        self.FAMBlock3_2 = FAMBlock(channels=256)
        self.FAM1_2 = nn.ModuleList([self.FAMBlock1_2 for i in range(6)])
        self.FAM2_2 = nn.ModuleList([self.FAMBlock2_2 for i in range(4)])
        self.FAM3_2 = nn.ModuleList([self.FAMBlock3_2 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder3 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder2 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder1 = DecoderBottleneckLayer(filters[1], filters[0])

        self.final_conv1_2 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1_2 = nn.ReLU(inplace=True)
        self.final_conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2_2 = nn.ReLU(inplace=True)
        self.final_conv3_2 = nn.Conv2d(32, n_classes, 3, padding=1)

        #####################################################################
        #####################################################################
        #####################################################################

        # self.FAMBlock1_3 = FAMBlock(channels=64)
        # self.FAMBlock2_3 = FAMBlock(channels=128)
        # self.FAMBlock3_3 = FAMBlock(channels=256)
        # self.FAM1_3 = nn.ModuleList([self.FAMBlock1_3 for i in range(6)])
        # self.FAM2_3 = nn.ModuleList([self.FAMBlock2_3 for i in range(4)])
        # self.FAM3_3 = nn.ModuleList([self.FAMBlock3_3 for i in range(2)])

        # feature_channels = [64, 128, 256, 512]
        # self.PPN = PSPModule(feature_channels[-1])
        # self.FPN = FPN_fuse(feature_channels, fpn_out=64)

        # self.final_conv1_3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        # self.final_relu1_3 = nn.ReLU(inplace=True)
        # self.final_conv2_3 = nn.Conv2d(32, 32, 3, padding=1)
        # self.final_relu2_3 = nn.ReLU(inplace=True)
        # self.final_conv3_3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        # Question here
        x = x.float()

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)


        if self.training:

            if self.counter % 2 == 0:

                for i in range(2):
                    e3 = self.FAM3_1[i](e3)
                for i in range(4):
                    e2 = self.FAM2_1[i](e2)
                for i in range(6):
                    e1 = self.FAM1_1[i](e1)

                x = self.up3_1(e4, e3)
                x = self.up2_1(x , e2)
                x = self.up1_1(x , e1)

                out = self.final_conv1_1(x)
                out = self.final_relu1_1(out)
                out = self.final_conv2_1(out)
                out = self.final_relu2_1(out)
                out = self.final_conv3_1(out)

            if self.counter % 2 == 1:

                e1, e2, e3, e4 = e1.detach(), e2.detach(), e3.detach(), e4.detach()

                for i in range(2):
                    e3 = self.FAM3_2[i](e3)
                for i in range(4):
                    e2 = self.FAM2_2[i](e2)
                for i in range(6):
                    e1 = self.FAM1_2[i](e1)

                d4 = self.decoder3(e4) + e3
                d3 = self.decoder2(d4) + e2
                d2 = self.decoder1(d3) + e1

                out = self.final_conv1_2(d2)
                out = self.final_relu1_2(out)
                out = self.final_conv2_2(out)
                out = self.final_relu2_2(out)
                out = self.final_conv3_2(out)

        else:

            for i in range(2):
                e31 = self.FAM3_1[i](e3)
            for i in range(4):
                e21 = self.FAM2_1[i](e2)
            for i in range(6):
                e11 = self.FAM1_1[i](e1)

            x1 = self.up3_1(e4, e31)
            x1 = self.up2_1(x1, e21)
            x1 = self.up1_1(x1, e11)

            out_1 = self.final_conv1_1(x1)
            out_1 = self.final_relu1_1(out_1)
            out_1 = self.final_conv2_1(out_1)
            out_1 = self.final_relu2_1(out_1)
            out_1 = self.final_conv3_1(out_1)

            for i in range(2):
                e32 = self.FAM3_2[i](e3)
            for i in range(4):
                e22 = self.FAM2_2[i](e2)
            for i in range(6):
                e12 = self.FAM1_2[i](e1)

            d4 = self.decoder3(e4) + e32
            d3 = self.decoder2(d4) + e22
            d2 = self.decoder1(d3) + e12

            out_2 = self.final_conv1_2(d2)
            out_2 = self.final_relu1_2(out_2)
            out_2 = self.final_conv2_2(out_2)
            out_2 = self.final_relu2_2(out_2)
            out_2 = self.final_conv3_2(out_2)

            out = (out_1, out_2)

        self.counter = self.counter + 1
        return out







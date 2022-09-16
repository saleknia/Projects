import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


class seg_head(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.scale_4 = nn.Upsample(scale_factor=2)
        self.scale_3 = nn.Upsample(scale_factor=2)
        self.scale_2 = nn.Upsample(scale_factor=2)
        self.conv_4 =  nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1))
        self.conv_3 =  nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1))
        self.conv_2 =  nn.Conv2d(128, 64 , kernel_size=(1,1), stride=(1,1))

        self.conv = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1,1))
        self.BN_out = nn.BatchNorm2d(64)
        self.RELU6_out = nn.ReLU6()

        self.out = nn.Conv2d(64, num_class, kernel_size=(1,1), stride=(1,1))

    def forward(self, up4, up3, up2, up1):
        up2 = torchvision.ops.stochastic_depth(input=up2, p=0.5, mode='batch')
        up3 = torchvision.ops.stochastic_depth(input=up3, p=0.5, mode='batch')
        up4 = torchvision.ops.stochastic_depth(input=up4, p=0.5, mode='batch')
        up4 = self.scale_4(self.conv_4(up4))
        up3 = up3 + up4
        up3 = self.scale_3(self.conv_3(up3))
        up2 = up3 + up2
        up2 = self.scale_2(self.conv_2(up2))
        up = up2 + up1
        
        up = self.conv(up)
        up = self.BN_out(up)
        up = self.RELU6_out(up)
        up = self.out(up)

        return up

class se_block(nn.Module):
    def __init__(self, in_channels, squeeze=4):
        super(se_block, self).__init__()
        self.SQ_1 = SQ(in_channels=in_channels)
        self.SQ_2 = SQ(in_channels=in_channels)
        self.SQ_3 = SQ(in_channels=in_channels)
        self.SQ_4 = SQ(in_channels=in_channels)
        self.out =  nn.Sequential(
                                    nn.Conv2d(in_channels*4, in_channels, kernel_size=1),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True)
                                )
    def forward(self, x):
        x = self.SQ_1(x) + self.SQ_2(x) + self.SQ_3(x) + self.SQ_4(x) 
        output = self.out(x)
        return output, x

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SQ(nn.Module):
    def __init__(self, in_channels):
        super(SQ, self).__init__()
        self.conv =  nn.Sequential(
                                    nn.Conv2d(in_channels, in_channels*4, kernel_size=1),
                                    nn.BatchNorm2d(in_channels*4),
                                    nn.ReLU(inplace=True)
                                )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):

    def __init__(self, img_ch=3, output_ch=1):
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        base = 64
        self.Conv1 = ConvBlock(img_ch, base)
        self.Conv2 = ConvBlock(base, base*2)
        self.Conv3 = ConvBlock(base*2, base*4)
        self.Conv4 = ConvBlock(base*4, base*8)
        self.Conv5 = ConvBlock(base*8, base*16)

        self.Up5 = UpConv(base*16, base*8)
        self.Att5 = AttentionBlock(F_g=base*8, F_l=base*8, n_coefficients=base*4)
        self.UpConv5 = ConvBlock(base*16, base*8)

        self.Up4 = UpConv(base*8, base*4)
        self.Att4 = AttentionBlock(F_g=base*4, F_l=base*4, n_coefficients=base*2)
        self.UpConv4 = ConvBlock(base*8, base*4)

        self.Up3 = UpConv(base*4, base*2)
        self.Att3 = AttentionBlock(F_g=base*2, F_l=base*2, n_coefficients=base)
        self.UpConv3 = ConvBlock(base*4, base*2)

        self.Up2 = UpConv(base*2, base)
        self.Att2 = AttentionBlock(F_g=base, F_l=base, n_coefficients=base//2)
        self.UpConv2 = ConvBlock(base*2, base)

        self.Conv = nn.Conv2d(base , output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return out
        # if self.training:
        #     return out, d5, d4, d3, d2, e5, e4, e3, e2, e1
        # else:
        #     return out  
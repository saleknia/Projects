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

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64 , output_ch, kernel_size=1, stride=1, padding=0)
        # # self.Conv_1 = nn.Conv2d(64 , 2, kernel_size=1, stride=1, padding=0)
        # self.Conv_2 = nn.Conv2d(64 , 9, kernel_size=1, stride=1, padding=0)
        # self.Conv_1 = seg_head(num_class=2)



    def forward(self, x, num_head=1.0):
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
        #     return out, d5, d4, d3, d2
        # else:
        #     return out

        # if num_head==1.0:
        #     return self.Conv_1(d2)
        # if num_head==2.0:
        #     return self.Conv_1(d2), self.Conv_2(d2)
       
        # if num_head==1.0:
        #     return self.Conv_1(up4=d5, up3=d4, up2=d3, up1=d2)
        # if num_head==2.0:
        #     return self.Conv_1(up4=d5, up3=d4, up2=d3, up1=d2), self.Conv_2(d2)









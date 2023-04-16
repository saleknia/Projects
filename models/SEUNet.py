from torchvision import models as resnet_model
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Softmax
import einops
import timm

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

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

        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn   = resnet.bn1
        self.firstrelu = resnet.relu
        self.maxpool   = resnet.maxpool 
        self.encoder1  = resnet.layer1
        self.encoder2  = resnet.layer2
        self.encoder3  = resnet.layer3
        self.encoder4  = resnet.layer4

        self.up3 = DecoderBottleneckLayer(in_channels=512, out_channels=256)
        self.up2 = DecoderBottleneckLayer(in_channels=256, out_channels=128)
        self.up1 = DecoderBottleneckLayer(in_channels=128, out_channels=64 )

        self.meta = MetaFormer()
        
        self.final_conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.ConvTranspose2d(32, n_classes, kernel_size=2, stride=2)

    def forward(self, x):
        b, c, h, w = x.shape
        # x = torch.cat([x, x, x], dim=1)
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e0 = self.maxpool(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e1, e2, e3 = self.meta(e1, e2, e3)

        e3 = self.up3(e4) + e3
        e2 = self.up2(e3) + e2
        e1 = self.up1(e2) + e1

        e = self.final_conv1(e1)
        e = self.final_relu1(e)
        e = self.final_conv2(e)
        e = self.final_relu2(e)
        e = self.final_conv3(e)

        return e



class MetaFormer(nn.Module):

    def __init__(self, num_skip=3, skip_dim=[64, 128, 256]):
        super().__init__()

        fuse_dim = 0
        for i in range(num_skip):
            fuse_dim += skip_dim[i]

        self.fuse_conv1 = nn.Conv2d(fuse_dim, skip_dim[0], 1, 1)
        self.fuse_conv2 = nn.Conv2d(fuse_dim, skip_dim[1], 1, 1)
        self.fuse_conv3 = nn.Conv2d(fuse_dim, skip_dim[2], 1, 1)

        self.down_sample1 = nn.AvgPool2d(4)
        self.down_sample2 = nn.AvgPool2d(2)

        self.up_sample1 = nn.Upsample(scale_factor=4)
        self.up_sample2 = nn.Upsample(scale_factor=2)

        self.att_3 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.att_2 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.att_1 = AttentionBlock(F_g=64 , F_l=64 , n_coefficients=32)

    def forward(self, x1, x2, x3):
        """
        x: B, H*W, C
        """
        org1 = x1
        org2 = x2
        org3 = x3

        x1_d = self.down_sample1(x1)
        x2_d = self.down_sample2(x2)

        list1 = [x1_d, x2_d, x3]

        # --------------------Concat sum------------------------------
        fuse = torch.cat(list1, dim=1)

        x1 = self.fuse_conv1(fuse)
        x2 = self.fuse_conv2(fuse)
        x3 = self.fuse_conv3(fuse)

        x1 = self.up_sample1(x1)
        x2 = self.up_sample2(x2)

        x1 = self.att_1(gate=x1, skip_connection=org1)
        x2 = self.att_2(gate=x2, skip_connection=org2)
        x3 = self.att_3(gate=x3, skip_connection=org3)

        return x1, x2, x3

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


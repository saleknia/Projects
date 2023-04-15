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
        
class SEUNet_loss(nn.Module):
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

        self.teacher = SEUNet()
        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_B0_90_00/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_teacher = loaded_data_teacher['net']
        self.teacher.load_state_dict(pretrained_teacher)
        self.teacher = self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn   = resnet.bn1
        self.firstrelu = resnet.relu
        self.maxpool   = resnet.maxpool 
        self.encoder1  = resnet.layer1[0]
        self.encoder2  = resnet.layer2[0]
        self.encoder3  = resnet.layer3[0]
        self.encoder4  = resnet.layer4[0]

        self.up3 = DecoderBottleneckLayer(in_channels=512, out_channels=256)
        self.up2 = DecoderBottleneckLayer(in_channels=256, out_channels=128)
        self.up1 = DecoderBottleneckLayer(in_channels=128, out_channels=64 )
        
        self.final_conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.ConvTranspose2d(32, n_classes, kernel_size=2, stride=2)

    def forward(self, x):
        b, c, h, w = x.shape
        # x = torch.cat([x, x, x], dim=1)

        e1_t, e2_t, e3_t, e4_t = self.teacher(x)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e0 = self.maxpool(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e3 = self.up3(e4) + e3
        e2 = self.up2(e3) + e2
        e1 = self.up1(e2) + e1

        e = self.final_conv1(e1)
        e = self.final_relu1(e)
        e = self.final_conv2(e)
        e = self.final_relu2(e)
        e = self.final_conv3(e)

        if self.training:
            return e, e1, e2, e3, e4, e1_t, e2_t, e3_t, e4_t
        else:
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

        # e3 = self.up3(e4) + e3
        # e2 = self.up2(e3) + e2
        # e1 = self.up1(e2) + e1

        # e = self.final_conv1(e1)
        # e = self.final_relu1(e)
        # e = self.final_conv2(e)
        # e = self.final_relu2(e)
        # e = self.final_conv3(e)

        return e1, e2, e3, e4




import torch.nn as nn
import einops
import timm

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class DABNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(DABNet, self).__init__()

        self.convnext = timm.create_model('convnext_tiny', pretrained=True, features_only=True)

        self.norm_3 = LayerNormProxy(dim=768)
        self.norm_2 = LayerNormProxy(dim=384)
        self.norm_1 = LayerNormProxy(dim=192)
        self.norm_0 = LayerNormProxy(dim=96)


        filters = [96, 192, 384, 768]
        self.decoder3 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder2 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder1 = DecoderBottleneckLayer(filters[1], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(96, 48, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(48, 48, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(48, n_classes, 3, padding=1)
        self.final_upsample = nn.Upsample(scale_factor=2.0)

    def forward(self, x):
        b, c, h, w = x.shape

        c0, c1, c2, c3 = self.convnext(x)

        d0 = self.decoder3(c3) + c2
        d0 = self.decoder2(d0) + c1
        d0 = self.decoder1(d0) + c0

        out = self.final_conv1(d0)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        out = self.final_upsample(out)

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
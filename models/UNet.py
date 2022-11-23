import torch.nn as nn
import torch
import timm
import numpy as np
from torch.nn import init
from collections import OrderedDict
import ml_collections
from torchvision import models as resnet_model

class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[3,5],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V


class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        return out

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
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
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        return self.nConvs(x)

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients, scale_factor):
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
        self.up = nn.Upsample(scale_factor=scale_factor)
    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        gate = self.up(gate)
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        # self.att = ParallelPolarizedSelfAttention(channel=out_channels)
    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)  # dim 1 is the channel dimension
        x = self.conv(x)
        # x = self.att(x)
        return x

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

class FAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1

        return out

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

class GCM(nn.Module):
    def __init__(self):
        super(GCM, self).__init__()
        self.conv_1 = ConvBatchNorm(in_channels=1024, out_channels=256, kernel_size=1, padding=0, dilation=1)
        self.conv_3 = ConvBatchNorm(in_channels=1024, out_channels=256, kernel_size=3, padding=3, dilation=3)
        self.conv_5 = ConvBatchNorm(in_channels=1024, out_channels=256, kernel_size=3, padding=5, dilation=5)
        self.conv_7 = ConvBatchNorm(in_channels=1024, out_channels=256, kernel_size=3, padding=7, dilation=7)

    def forward(self, x):
        y = torch.cat([self.conv_1(x), self.conv_3(x), self.conv_5(x), self.conv_7(x)], dim=1)
        return y


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        in_channels = 16
        self.encoder = timm.create_model('hrnet_w18', pretrained=True, features_only=True)
        self.encoder.incre_modules = None
        self.encoder.conv1.stride = (1, 1)

        transformer = deit_small_distilled_patch16_224(pretrained=True)
        self.patch_embed = transformer.patch_embed

        self.transformers_stage_1 = nn.ModuleList([transformer.blocks[i] for i in range(0 , 8 )])
        self.transformers_stage_2 = nn.ModuleList([transformer.blocks[i] for i in range(8 , 10)])
        self.transformers_stage_3 = nn.ModuleList([transformer.blocks[i] for i in range(10, 12)])

        self.conv_seq_img_1 = ConvBatchNorm(in_channels=384, out_channels=144, activation='ReLU', kernel_size=1, padding=0, dilation=1)
        self.conv_seq_img_2 = ConvBatchNorm(in_channels=384, out_channels=144, activation='ReLU', kernel_size=1, padding=0, dilation=1)
        self.conv_seq_img_3 = ConvBatchNorm(in_channels=384, out_channels=144, activation='ReLU', kernel_size=1, padding=0, dilation=1)

        self.combine_1 = ConvBatchNorm(in_channels=288, out_channels=144, activation='ReLU', kernel_size=3, padding=1, dilation=1)
        self.combine_2 = ConvBatchNorm(in_channels=288, out_channels=144, activation='ReLU', kernel_size=3, padding=1, dilation=1)
        self.combine_3 = ConvBatchNorm(in_channels=288, out_channels=144, activation='ReLU', kernel_size=3, padding=1, dilation=1)

        # self.maxpool = nn.MaxPool2d(2)
        # self.FAMBlock = FAMBlock(in_channels=64, out_channels=64)
        # self.FAM = nn.ModuleList([self.FAMBlock for i in range(6)])

        # torch.Size([8, 16 , 112, 112])
        # torch.Size([8, 32 , 56 , 56])
        # torch.Size([8, 64 , 28 , 28])
        # torch.Size([8, 128, 14 , 14])
        # torch.Size([8, 256, 7  , 7])

        self.up4 = UpBlock(144 , 72, nb_Conv=2)
        self.up3 = UpBlock(72  , 36, nb_Conv=2)
        self.up2 = UpBlock(36  , 18, nb_Conv=2)
        # self.up1 = UpBlock(32  , 16 , nb_Conv=2)

        # self.attention_3 = AttentionBlock(F_g=1024, F_l=512, n_coefficients=512, scale_factor=2.00)
        # self.attention_2 = AttentionBlock(F_g=1024, F_l=256, n_coefficients=256, scale_factor=4.00)
        # self.attention_1 = AttentionBlock(F_g=1024, F_l=128, n_coefficients=128, scale_factor=8.00)
        # self.attention_0 = AttentionBlock(F_g=1024, F_l=64 , n_coefficients=64 , scale_factor=16.0)

        self.final_conv1 = nn.ConvTranspose2d(18, 9, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(9, 9, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(9, n_classes, 3, padding=1)

    def forward(self, x):
        # Question here
        x0 = x.float()
        b, c, h, w = x.shape
        
        x = self.encoder.conv1(x0)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)
        x = self.encoder.conv2(x)
        x = self.encoder.bn2(x)
        x = self.encoder.act2(x)
        x = self.encoder.layer1(x)

        xl = [t(x) for i, t in enumerate(self.encoder.transition1)]
        yl = self.encoder.stage2(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.encoder.transition2)]
        yl = self.encoder.stage3(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.encoder.transition3)]

        emb = self.patch_embed(x0)
        for i in range(8):
            emb = self.transformers_stage_1[i](emb)

        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 384, 14, 14)
        feature_tf = self.conv_seq_img_1(feature_tf)

        xl[3] = self.combine_1(torch.cat([xl[3], feature_tf], dim=1))

        xl = self.encoder.stage4[0](xl)

        for i in range(2):
            emb = self.transformers_stage_2[i](emb)

        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 384, 14, 14)
        feature_tf = self.conv_seq_img_2(feature_tf)

        xl[3] = self.combine_2(torch.cat([xl[3], feature_tf], dim=1))

        xl = self.encoder.stage4[1](xl)

        for i in range(2):
            emb = self.transformers_stage_3[i](emb)

        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 384, 14, 14)
        feature_tf = self.conv_seq_img_3(feature_tf)

        xl[3] = self.combine_3(torch.cat([xl[3], feature_tf], dim=1))

        xl = self.encoder.stage4[2](xl)

        x1, x2, x3, x4 = xl[0], xl[1], xl[2], xl[3]
        
        # x0, x1, x2, x3, x4 = self.encoder(x)
                     
        # for i in range(6):
        #     x0 = self.FAM[i](x0)
        # x0 = self.reduction(x0)

        # x3 = self.attention_3(gate=x4, skip_connection=x3)
        # x2 = self.attention_2(gate=x4, skip_connection=x2)
        # x1 = self.attention_1(gate=x4, skip_connection=x1)
        # x0 = self.attention_0(gate=x4, skip_connection=x0)

        x = self.up4(x4, x3)
        x = self.up3(x , x2)
        x = self.up2(x , x1)
        # x = self.up1(x , x0)

        x = self.final_conv1(x)
        x = self.final_relu1(x)
        x = self.final_conv2(x)
        x = self.final_relu2(x)
        out = self.final_conv3(x)

        return out

# class UNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=1):
#         super(UNet, self).__init__()

#         transformer = deit_tiny_distilled_patch16_224(pretrained=True)
#         resnet = resnet_model.resnet34(pretrained=True)

#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4

#         self.patch_embed = transformer.patch_embed
#         self.transformers = nn.ModuleList(
#             [transformer.blocks[i] for i in range(12)]
#         )

#         self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
#         self.se = SEBlock(channel=1024)
#         self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

#         self.FAMBlock1 = FAMBlock(channels=64)
#         self.FAMBlock2 = FAMBlock(channels=128)
#         self.FAMBlock3 = FAMBlock(channels=256)
#         self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
#         self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
#         self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

#         filters = [64, 128, 256, 512]
#         self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
#         self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
#         self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
#         self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

#         self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.final_relu1 = nn.ReLU(inplace=True)
#         self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.final_relu2 = nn.ReLU(inplace=True)
#         self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)


#     def forward(self, x):
#         b, c, h, w = x.shape

#         e0 = self.firstconv(x)
#         e0 = self.firstbn(e0)
#         e0 = self.firstrelu(e0)

#         e1 = self.encoder1(e0)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         feature_cnn = self.encoder4(e3)

#         emb = self.patch_embed(x)
#         for i in range(12):
#             emb = self.transformers[i](emb)
#         feature_tf = emb.permute(0, 2, 1)
#         feature_tf = feature_tf.view(b, 192, 14, 14)
#         feature_tf = self.conv_seq_img(feature_tf)

#         feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
#         feature_att = self.se(feature_cat)
#         feature_out = self.conv2d(feature_att)

#         for i in range(2):
#             e3 = self.FAM3[i](e3)
#         for i in range(4):
#             e2 = self.FAM2[i](e2)
#         for i in range(6):
#             e1 = self.FAM1[i](e1)
#         d4 = self.decoder4(feature_out) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1

#         out1 = self.final_conv1(d2)
#         out1 = self.final_relu1(out1)
#         out = self.final_conv2(out1)
#         out = self.final_relu2(out)
#         out = self.final_conv3(out)

#         return out


import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution, which can maintain feature map size
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # Using hierarchical feature fusion (HFF) to ease the gridding artifacts which is introduced
        # by the large effective receptive filed of the ESP module
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # combine_in_out = input + combine  #shotcut path
        output = self.bn(combine)
        output = self.act(output)
        return output


# ESP block
class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut / 3)  # K=3,
        n1 = nOut - 2 * n  # (N-(K-1)INT(N/K)) for dilation rate of 2^0, for producing an output feature map of channel=nOut
        self.c1 = C(nIn, n, 1, 1)  # the point-wise convolutions with 1x1 help in reducing the computation, channel=c

        # K=5, dilation rate: 2^{k-1},k={1,2,3,...,K}
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n , 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n , 3, 1, 4)  # dilation rate of 2^2

        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)

        # Using hierarchical feature fusion (HFF) to ease the gridding artifacts which is introduced
        # by the large effective receptive filed of the ESP module
        add1 = d2
        add2 = add1 + d4

        # merge
        combine = torch.cat([d1, add1, add2], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output








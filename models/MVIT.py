import torch
from torchvision import models as resnet_model
from torch import nn
import timm
import math
import torch
import random
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import einops
import timm
from torchvision import models as resnet_model
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple
import ml_collections
from .CTrans import ChannelTransformer

def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 288  
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    # config.transformer.embeddings_dropout_rate = 0.3
    # config.transformer.attention_dropout_rate  = 0.3
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate  = 0.1
    config.transformer.dropout_rate = 0
    # config.patch_sizes = [16,8,4,2]
    config.patch_sizes = [4,2,1]
    config.base_channel = 96 # base channel of U-Net
    config.n_classes = 1
    return config

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
        
class Linear_Eca_block(nn.Module):
    """docstring for Eca_block"""
    def __init__(self):
        super(Linear_Eca_block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=5, padding=int(5/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gamma=2, b=1):
        #N, C, H, W = x.size()
        y = self.avgpool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)

class HybridAttention(nn.Module):
    def __init__(self, channels):
        super(HybridAttention, self).__init__()

        self.eca     = Linear_Eca_block()
        self.conv    = BasicConv2d(channels, channels, 3, 1, 1)
        self.down_c  = BasicConv2d(channels, 1, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.final_conv = ConvBatchNorm(in_channels=channels*2 , out_channels=channels, activation='ReLU', kernel_size=1, padding=0)

    def forward(self, x_t, x_c):

        sa = self.sigmoid(self.down_c(x_c))
        gc = self.eca(x_t)
        x_c = self.conv(x_c)
        x_c = x_c * gc
        x_t = x_t * sa
        x = self.final_conv(torch.cat((x_t, x_c), 1))
        return x

class MVIT(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(MVIT, self).__init__()

        self.transformer = trans()
        self.convnext    = convnext()

        self.up_2 = UpBlock(96, 96)
        self.up_1 = UpBlock(96, 96)

        self.final_conv1 = nn.ConvTranspose2d(96, 48, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(48, 48, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(48, n_classes, 3, padding=1)
        self.final_upsample = nn.Upsample(scale_factor=2.0)

        self.HA_2 = HybridAttention(channels=96)
        self.HA_1 = HybridAttention(channels=96)
        self.HA_0 = HybridAttention(channels=96)

        self.FAMBlock1 = FAMBlock(channels=96)
        self.FAMBlock2 = FAMBlock(channels=96)
        self.FAMBlock3 = FAMBlock(channels=96)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(4)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(4)])

        # self.head = SegFormerHead()

        # self.mtc = ChannelTransformer(get_CTranS_config(), vis=False, img_size=224, channel_num=[96, 96, 96], patchSize=[4, 2, 1])


    def forward(self, x):
        b, c, h, w = x.shape

        t0, t1, t2, out_trans = self.transformer(x)
        c0, c1, c2, out_cnext = self.convnext(x)

        x0 = self.HA_0(t0, c0)
        x1 = self.HA_1(t1, c1)        
        x2 = self.HA_2(t2, c2)

        for i in range(4):
            x2 = self.FAM3[i](x2)
        for i in range(4):
            x1 = self.FAM2[i](x1)
        for i in range(4):
            x0 = self.FAM1[i](x0)

        # x0, x1, x2 = self.mtc(x0, x1, x2)

        x1 = self.up_2(x2, x1)
        x0 = self.up_1(x1, x0)

        # x = self.head(x0, x1, x2)

        out = self.final_conv1(x0)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        out = self.final_upsample(out)

        if self.training:
            return out, out_trans, out_cnext
        else:
            return out

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

class SelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.softmax_spatial=nn.Softmax(-1)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=1.0 - self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=(spatial_weight*x) + x
        out=spatial_out
        
        return out

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self):
        super(SegFormerHead, self).__init__()

        embedding_dim = 96

        self.linear_fuse = BasicConv2d(embedding_dim*3, embedding_dim, 1)

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(96, 48)
        self.fc2 = nn.Linear(48, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.up_2 = nn.Upsample(scale_factor=2.0)
        self.up_3 = nn.Upsample(scale_factor=4.0)

    def forward(self, c1, c2, c3):

        ############## MLP decoder on C1-C3 ###########
        n, _, h, w = c1.shape

        c3 = self.up_3(c3)
        c2 = self.up_2(c2)

        y3 = self.pooling(c3)
        y2 = self.pooling(c2)
        y1 = self.pooling(c1)

        y = y1 + y2 + y3

        coeff = self.sigmoid(self.fc2(self.relu(self.fc1(y.reshape(n, -1)))))

        c1 = c1 * coeff[:,0].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(c1)
        c2 = c2 * coeff[:,1].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(c2)
        c3 = c3 * coeff[:,2].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(c3)

        c = self.linear_fuse(torch.cat([c3, c2, c1], dim=1))

        return c

class trans(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(trans, self).__init__()

        self.trans = CrossFormer(img_size=224,
                                    patch_size=[4, 8, 16, 32],
                                    in_chans= 3,
                                    num_classes=1000,
                                    embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    group_size=[7, 7, 7, 7],
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.0,
                                    drop_path_rate=0.2,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False,
                                    merge_size=[[2, 4], [2,4], [2, 4]])

        self.norm_2 = LayerNormProxy(dim=384)
        self.norm_1 = LayerNormProxy(dim=192)
        self.norm_0 = LayerNormProxy(dim=96)

        # self.up_2 = UpBlock(96, 96)
        # self.up_1 = UpBlock(96, 96)
        
        filters = [96, 192, 384]
        self.decoder2 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder1 = DecoderBottleneckLayer(filters[1], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(96, 48, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(48, 48, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(48, n_classes, 3, padding=1)
        self.final_upsample = nn.Upsample(scale_factor=2.0)

        self.reduce_0 = ConvBatchNorm(in_channels=96 , out_channels=96, activation='ReLU', kernel_size=1, padding=0)
        self.reduce_1 = ConvBatchNorm(in_channels=192, out_channels=96, activation='ReLU', kernel_size=1, padding=0)
        self.reduce_2 = ConvBatchNorm(in_channels=384, out_channels=96, activation='ReLU', kernel_size=1, padding=0)

        # self.mtc = ChannelTransformer(get_CTranS_config(), vis=False, img_size=224, channel_num=[96, 96, 96], patchSize=[4, 2, 1])

    def forward(self, x):
        b, c, h, w = x.shape

        t0, t1, t2 = self.trans(x)

        d0 = self.decoder2(t2) + t1
        d0 = self.decoder1(d0) + t0

        out = self.final_conv1(d0)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        out = self.final_upsample(out)

        t2 = self.reduce_2(self.norm_2(t2)) 
        t1 = self.reduce_1(self.norm_1(t1)) 
        t0 = self.reduce_0(self.norm_0(t0))

        # t = self.up_2(t2, t1)
        # t = self.up_1(t , t0)

        # out = self.final_conv1(t)
        # out = self.final_relu1(out)
        # out = self.final_conv2(out)
        # out = self.final_relu2(out)
        # out = self.final_conv3(out)
        # out = self.final_upsample(out)

        return t0, t1, t2, out

class convnext(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(convnext, self).__init__()

        self.convnext = timm.create_model('convnext_tiny', pretrained=True, features_only=True, out_indices=[0,1,2])

        self.norm_2 = LayerNormProxy(dim=384)
        self.norm_1 = LayerNormProxy(dim=192)
        self.norm_0 = LayerNormProxy(dim=96)


        self.reduce_0 = ConvBatchNorm(in_channels=96 , out_channels=96, activation='ReLU', kernel_size=1, padding=0)
        self.reduce_1 = ConvBatchNorm(in_channels=192, out_channels=96, activation='ReLU', kernel_size=1, padding=0)
        self.reduce_2 = ConvBatchNorm(in_channels=384, out_channels=96, activation='ReLU', kernel_size=1, padding=0)

        filters = [96, 192, 384]
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

        c0, c1, c2 = self.convnext(x)

        d0 = self.decoder2(c2) + c1
        d0 = self.decoder1(d0) + c0

        out = self.final_conv1(d0)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        out = self.final_upsample(out)

        c2 = self.reduce_2(self.norm_2(c2)) 
        c1 = self.reduce_1(self.norm_1(c1)) 
        c0 = self.reduce_0(self.norm_0(c0))

        return c0, c1, c2, out

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

import torch.nn as nn
import torch

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

    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv=2, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up     = nn.Upsample(scale_factor=2)
        # self.up   = nn.ConvTranspose2d(in_channels       ,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels * 2, out_channels, nb_Conv, activation)
        # self.attention = SelfAttention(channel=in_channels)

    def forward(self, x, skip_x):
        out = self.up(x)
        # out = self.attention(out)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return (self.nConvs(x))

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

NEG_INF = -1000000

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops

class Attention(nn.Module):
    r""" Multi-head self attention module with relative position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Gh*Gw, Gh*Gw) or None
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H*W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases) # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view( 
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nG = mask.shape[0]
            attn = attn.view(B_ // nG, nG, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # (B, nG, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        excluded_flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        excluded_flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        excluded_flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        if self.position_bias:
            flops += self.pos.flops(N)
        return flops, excluded_flops


class CrossFormerBlock(nn.Module):
    r""" CrossFormer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Window size.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, group_size=7, interval=8, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        if min(H, W) <= self.group_size:
            # if window size is larger than input resolution, we don't partition windows
            self.lsda_flag = 0
            self.group_size = min(H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # padding
        size_div = self.interval if self.lsda_flag == 1 else self.group_size
        pad_l = pad_t = 0
        pad_r = (size_div - W % size_div) % size_div
        pad_b = (size_div - H % size_div) % size_div
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1

        # group embeddings and generate attn_mask
        if self.lsda_flag == 0: # SDA
            G = Gh = Gw = self.group_size
            x = x.reshape(B, Hp // G, G, Wp // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.reshape(B * Hp * Wp // G**2, G**2, C)
            nG = Hp * Wp // G**2
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Hp // G, G, Wp // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                mask = mask.reshape(nG, 1, G * G)
                attn_mask = torch.zeros((nG, G * G, G * G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        else: # LDA
            I, Gh, Gw = self.interval, Hp // self.interval, Wp // self.interval
            x = x.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.reshape(B * I * I, Gh * Gw, C)
            nG = I ** 2
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
                mask = mask.reshape(nG, 1, Gh * Gw)
                attn_mask = torch.zeros((nG, Gh * Gw, Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        # multi-head self-attention
        x = self.attn(x, Gh, Gw, mask=attn_mask)  # nG*B, G*G, C
        
        # ungroup embeddings
        if self.lsda_flag == 0:
            x = x.reshape(B, Hp // G, Wp // G, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous() # B, Hp//G, G, Wp//G, G, C
        else:
            x = x.reshape(B, I, I, Gh, Gw, C).permute(0, 3, 1, 4, 2, 5).contiguous() # B, Gh, I, Gw, I, C
        x = x.reshape(B, Hp, Wp, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"group_size={self.group_size}, lsda_flag={self.lsda_flag}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # Attention
        size_div = self.interval if self.lsda_flag == 1 else self.group_size
        Hp = math.ceil(H / size_div) * size_div
        Wp = math.ceil(W / size_div) * size_div
        Gh = Hp / size_div if self.lsda_flag == 1 else self.group_size
        Gw = Wp / size_div if self.lsda_flag == 1 else self.group_size
        nG = Hp * Wp / Gh / Gw
        attn_flops, attn_excluded_flops = self.attn.flops(Gh * Gw)
        flops += nG * attn_flops
        excluded_flops = nG * attn_excluded_flops
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops, excluded_flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, patch_size=[2], num_input_patch_size=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
        self.norm = norm_layer(dim)

        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.reductions.append(nn.Conv2d(dim, out_dim, kernel_size=ps, 
                                                stride=stride, padding=padding))

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x).flatten(2).transpose(1, 2).contiguous()
            xs.append(tmp_x)
        x = torch.cat(xs, dim=2)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                out_dim = 2 * self.dim // 2 ** i
            else:
                out_dim = 2 * self.dim // 2 ** (i + 1)
            flops += (H // 2) * (W // 2) * ps * ps * out_dim * self.dim
        return flops


class Stage(nn.Module):
    """ CrossFormer blocks for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        group_size (int): Group size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Ghether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, group_size, interval,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 patch_size_end=[4], num_patch_size=None):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            lsda_flag = 0 if (i % 2 == 0) else 1
            self.blocks.append(CrossFormerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, group_size=group_size, interval=interval,
                                 lsda_flag=lsda_flag,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 num_patch_size=num_patch_size))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, 
                                         patch_size=patch_size_end, num_input_patch_size=num_patch_size)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, H, W)

        B, _, C = x.shape
        feat = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        if self.downsample is not None:
            x = self.downsample(x, H, W)
        return feat, x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    def flops(self):
        flops = 0
        excluded_flops = 0
        for blk in self.blocks:
            blk_flops, blk_excluded_flops = blk.flops()
            flops += blk_flops
            excluded_flops += blk_excluded_flops
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops, excluded_flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // 4, img_size[1] // 4] # only for flops calculation
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = 4
            padding = (ps - 4) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x).flatten(2).transpose(1, 2)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=2)
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = 0
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                dim = self.embed_dim // 2 ** i
            else:
                dim = self.embed_dim // 2 ** (i + 1)
            flops += Ho * Wo * dim * self.in_chans * (self.patch_size[i] * self.patch_size[i])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class CrossFormer(nn.Module):
    r""" CrossFormer
        A PyTorch impl of : `CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention`  -

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each stage.
        num_heads (tuple(int)): Number of attention heads in different layers.
        group_size (int): Group size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Ghether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=[4], in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 group_size=7, crs_interval=[8, 4, 2, 1], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, merge_size=[[2], [2], [2]], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution # [H//4, W//4] of original image size

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers ape
        self.layers = nn.ModuleList()

        num_patch_sizes = [len(patch_size)] + [len(m) for m in merge_size]
        for i_layer in range(self.num_layers):
            patch_size_end = merge_size[i_layer] if i_layer < self.num_layers - 1 else None
            num_patch_size = num_patch_sizes[i_layer]
            layer = Stage(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               group_size=group_size[i_layer],
                               interval=crs_interval[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               patch_size_end=patch_size_end,
                               num_patch_size=num_patch_size)
            self.layers.append(layer)
        checkpoint = torch.load('/content/drive/MyDrive/crossformer-s.pth', map_location='cpu')
        state_dict = checkpoint['model']
        self.load_state_dict(state_dict, strict=False)
        self.layers[3] = None

        # for param in self.layers[1].parameters():
        #     param.requires_grad = False

        # for param in self.layers[0].parameters():
        #     param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        outs = []
        for i, layer in enumerate(self.layers[0:3]):
        # for i, layer in enumerate(self.layers):
            feat, x = layer(x, H //4 //(2 ** i), W //4 //(2 ** i))
            outs.append(feat)

        return outs

    def flops(self):
        flops = 0
        excluded_flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            layer_flops, layer_excluded_flops = layer.flops()
            flops += layer_flops
            excluded_flops += layer_excluded_flops
        return flops, excluded_flops





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

        x1 = torch.permute(x1, (0, 3, 1, 2))
        x2 = torch.permute(x2, (0, 3, 1, 2))        
        x3 = torch.permute(x3, (0, 3, 1, 2))
        x4 = torch.permute(x4, (0, 3, 1, 2))

        return x1, x2, x3, x4



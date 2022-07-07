import torch.nn as nn
import torch
import torch.nn.functional as F
from .CTrans import ChannelTransformer
from .GT_UNet import *
from .GT_UNet import _make_bot_layer 
import numpy as np
from torch.nn import init
from torch.nn import Softmax
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d,NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable

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

# class SEAttention(nn.Module):

#     def __init__(self, channel=512,reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )


#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, skip_x_att , skip_x):
#         b, c, _, _ = skip_x_att.size()
#         y = self.avg_pool(skip_x_att).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return skip_x * y.expand_as(skip_x)

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
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

class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)

        return x 


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', kernel_size=3, padding=1):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation, kernel_size=kernel_size, padding=padding))

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

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class BatchNorm(nn.Module):
    """([BN] => ReLU)"""

    def __init__(self, in_channels, activation='ReLU'):
        super(BatchNorm, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.norm(x)
        return self.activation(out)

class Conv(nn.Module):
    """(convolution)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out

# class DownBlock_BN(nn.Module):
#     """Downscaling with maxpool convolution"""

#     def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', dilations=[2,3]):
#         super(DownBlock_BN, self).__init__()
#         self.maxpool = nn.MaxPool2d(2)
#         self.shortcut = _make_nConv(in_channels, out_channels, nb_Conv, activation)
#         self.mhsa = MHSA(in_channels=in_channels, heads=4, curr_h=8, curr_w=8,pos_enc_type='relative')
#         # self.bottleneck_dimension = out_channels // 2

#         # self.conv1 = Conv(in_channels, self.bottleneck_dimension ,kernel_size=3, padding=1, stride=1, dilation=1)
#         # self.conv2 = Conv(in_channels, self.bottleneck_dimension ,kernel_size=3, padding=dilations[0], stride=1, dilation=dilations[0])
#         # self.conv3 = Conv(in_channels, self.bottleneck_dimension ,kernel_size=3, padding=dilations[1], stride=1, dilation=dilations[1])

#         # self.conv4 = Conv(self.bottleneck_dimension*2, out_channels, kernel_size=1, padding=0, stride=1)
#         # self.conv5 = Conv(out_channels*2 ,out_channels , kernel_size=1, padding=0, stride=1)
#         # self.BN_1 = BatchNorm(in_channels=out_channels, activation='ReLU')
#         # self.BN_2 = BatchNorm(in_channels=out_channels, activation='ReLU')

#     def forward(self, x):
#         x = self.maxpool(x)
#         out = self.shortcut(x)

#         Q_h = Q_w = 8
#         N, C, H, W = out.shape
#         P_h, P_w = H // Q_h, W // Q_w

#         out = out.reshape(N * P_h * P_w, C, Q_h, Q_w)

#         out = self.mhsa(out)
#         out = out.permute(0, 3, 1, 2)  # back to pytorch dim order
#         N1, C1, H1, W1 = out.shape
#         out = out.reshape(N, C1, int(H), int(W))


#         # out = torch.cat((self.conv2(x),self.conv3(x)),dim=1)
#         # out = self.conv4(out)
#         # out = self.BN_1(out)

#         # out = torch.cat((out,shortcut),dim=1)
#         # out = self.conv5(out)
#         # out = self.BN_2(out)
#         # out = out + shortcut

#         return out

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()
        # self.chanel_in = in_dim
        self.gamma = nn.parameter.Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class Indentity(nn.Module):

    def __init__(self):
        super(Indentity, self).__init__()
    def forward(self, x):
        return x

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU',PA=True):
        super(UpBlock, self).__init__()
        self.up_1 = nn.Upsample(scale_factor=2)
        # self.up_2 = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        # self.gamma = nn.parameter.Parameter(torch.zeros(1))
        self.nConvs = _make_nConv(in_channels=in_channels, out_channels=out_channels, nb_Conv=2, activation='ReLU')
        # self.att = AttentionBlock(F_g=in_channels//2, F_l=in_channels//2, n_coefficients=in_channels//4)
        # self.se = SEAttention(channel=in_channels, reduction=8)
        # self.CA_skip = CAM_Module()
        # self.CA_x = CAM_Module()
        # self.PA = PA
        # if self.PA:
        #     self.PA = PAM_Module(in_dim=in_channels//2)

        # self.att = ParallelPolarizedSelfAttention(channel = in_channels//2)

        # self.nConvs_out = _make_nConv(in_channels=out_channels , out_channels=out_channels, nb_Conv=1, activation='ReLU')

    def forward(self, x, skip_x):
        out = self.up_1(x)
        # out = self.att(out)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        x = self.nConvs(x) 
        return x



class UNet(nn.Module):
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
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # Question here
        in_channels = 64

        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)

        # self.Conv1 = _make_bot_layer(ch_in=n_channels,ch_out=64)
        # self.Conv2 = _make_bot_layer(ch_in=64, ch_out=128)
        # self.Conv3 = _make_bot_layer(ch_in=128, ch_out=256)
        # self.Conv4 = _make_bot_layer(ch_in=256, ch_out=512)
        # self.Conv5 = _make_bot_layer(ch_in=512, ch_out=512) # original ch_out was 1024.



        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2, PA=False)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2, PA=False)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2, PA=False)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2, PA=False)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))


        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        up4 = self.up4(x5 , x4)
        up3 = self.up3(up4, x3)
        up2 = self.up2(up3, x2)
        up1 = self.up1(up2, x1)


        if self.last_activation is not None:
            logits = self.last_activation(self.outc(up1))
        else:
            logits = self.outc(up1)


        return logits




class RelPosSelfAttention(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative=True, fold_heads=False):
        super(RelPosSelfAttention, self).__init__()
        self.relative = relative
        self.fold_heads = fold_heads
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)

    def forward(self, q, k, v):
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits += self.relative_logits(q)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w = self.relative_logits_1d(q, self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension
        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim = q.shape
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _ = x.shape
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype).cuda()
        x = torch.cat([x, col_pad], dim=3)
        flat_x = torch.reshape(x, [bs, heads, -1]).cuda()
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype).cuda()
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :length, length - 1:]
        return final_x


class AbsPosSelfAttention(nn.Module):

    def __init__(self, W, H, dkh, absolute=True, fold_heads=False):
        super(AbsPosSelfAttention, self).__init__()
        self.absolute = absolute
        self.fold_heads = fold_heads

        self.emb_w = nn.Parameter(torch.Tensor(W, dkh))
        self.emb_h = nn.Parameter(torch.Tensor(H, dkh))
        nn.init.normal_(self.emb_w, dkh ** -0.5)
        nn.init.normal_(self.emb_h, dkh ** -0.5)

    def forward(self, q, k, v):
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        abs_logits = self.absolute_logits(q)
        if self.absolute:
            logits += abs_logits
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def absolute_logits(self, q):
        """Compute absolute position enc logits."""
        emb_h = self.emb_h[:, None, :]
        emb_w = self.emb_w[None, :, :]
        emb = emb_h + emb_w
        abs_logits = torch.einsum('bhxyd,pqd->bhxypq', q, emb)
        return abs_logits


class GroupPointWise(nn.Module):
    """"""

    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(torch.Tensor(in_channels, heads, proj_channels // heads))

        nn.init.normal_(self.w, std=0.01)

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        input = input.permute(0, 2, 3, 1).float()
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out = torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out


class MHSA(nn.Module):


    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type='relative', use_pos=True):
        super(MHSA, self).__init__()
        self.q_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.k_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.v_proj = GroupPointWise(in_channels, heads, proj_factor=1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type == 'relative':
            self.self_attention = RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads=True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        o = self.self_attention(q=q, k=k, v=v)
        return o












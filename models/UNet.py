import torch.nn as nn
import torch
import torchvision
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
import torchvision.ops
from torch.nn.modules import conv
from torch.nn.modules.conv import Conv2d
from einops import rearrange
from collections import OrderedDict


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

class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:

            self.convs.append(GhostModule(channel, channel))

            # self.convs.append(
            #     nn.Sequential(OrderedDict([
            #         ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
            #         ('bn',nn.BatchNorm2d(channel)),
            #         ('relu',nn.ReLU())
            #     ]))
            # )
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

    def forward(self, decoder, encoder):
        b, c, _, _ = decoder.size()
        y = self.avg_pool(decoder).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return encoder * y.expand_as(encoder)

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

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', kernel_size=3, padding=1, ghost=False):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation, kernel_size=kernel_size, padding=padding, ghost=ghost))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=8, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1, ghost=False):
        super(ConvBatchNorm, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if ghost:
            self.op = GhostModule(in_channels, out_channels)
        else:
            self.op = nn.Sequential(nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(out_channels), get_activation(activation))

    def forward(self, x):
        return self.op(x)

class DConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1):
        super(DConvBatchNorm, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = DeformableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', ghost=False):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, ghost=ghost)

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

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', ghost=False):
        super(UpBlock, self).__init__()
        self.up_1 = nn.Upsample(scale_factor=2)
        self.nConvs = _make_nConv(in_channels=in_channels, out_channels=out_channels, nb_Conv=2, activation='ReLU', kernel_size=1, padding=0, ghost=ghost)

    def forward(self, x, skip_x):
        out = self.up_1(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        x = self.nConvs(x) 
        return x


###################################################

def conv_bn(inp,oup,kernel_size=3,stride=1):
    return nn.Sequential(
        nn.Conv2d(inp,oup,kernel_size=kernel_size,stride=stride,padding=kernel_size//2),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.ln=nn.LayerNorm(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.ln(x),**kwargs)

class FeedForward(nn.Module):
    def __init__(self,dim,mlp_dim,dropout) :
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,heads,head_dim,dropout):
        super().__init__()
        inner_dim=heads*head_dim
        project_out=not(heads==1 and head_dim==dim)

        self.heads=heads
        self.scale=head_dim**-0.5

        self.attend=nn.Softmax(dim=-1)
        self.to_qkv=nn.Linear(dim,inner_dim*3,bias=False)
        
        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,x):
        qkv=self.to_qkv(x).chunk(3,dim=-1)
        q,k,v=map(lambda t:rearrange(t,'b p n (h d) -> b p h n d',h=self.heads),qkv)
        dots=torch.matmul(q,k.transpose(-1,-2))*self.scale
        attn=self.attend(dots)
        out=torch.matmul(attn,v)
        out=rearrange(out,'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,head_dim,mlp_dim,dropout=0.):
        super().__init__()
        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,Attention(dim,heads,head_dim,dropout)),
                PreNorm(dim,FeedForward(dim,mlp_dim,dropout))
            ]))


    def forward(self,x):
        out=x
        for att,ffn in self.layers:
            out=out+att(out)
            out=out+ffn(out)
        return out

class MobileViTAttention(nn.Module):
    def __init__(self,in_channel=3,dim=512,kernel_size=3,patch_size=1,depth=3,mlp_dim=1024):
        super().__init__()
        self.ph,self.pw=patch_size,patch_size
        self.conv1=GhostModule(inp=in_channel, oup=in_channel, kernel_size=3, ratio=8, dw_size=3, stride=1, relu=True)
        self.conv2=nn.Conv2d(in_channel,dim,kernel_size=1)

        self.trans=Transformer(dim=dim,depth=depth,heads=8,head_dim=64,mlp_dim=mlp_dim)

        self.conv3=nn.Conv2d(dim,in_channel,kernel_size=1)
        self.conv4=GhostModule(inp=2*in_channel, oup=in_channel, kernel_size=3, ratio=8, dw_size=3, stride=1, relu=True)

    def forward(self,x):
        y=x.clone() #bs,c,h,w

        ## Local Representation
        y=self.conv2(self.conv1(x)) #bs,dim,h,w

        ## Global Representation
        _,_,h,w=y.shape
        y=rearrange(y,'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim',ph=self.ph,pw=self.pw) #bs,h,w,dim
        y=self.trans(y)
        y=rearrange(y,'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)',ph=self.ph,pw=self.pw,nh=h//self.ph,nw=w//self.pw) #bs,dim,h,w

        ## Fusion
        y=self.conv3(y) #bs,dim,h,w
        y=torch.cat([x,y],1) #bs,2*dim,h,w
        y=self.conv4(y) #bs,c,h,w

        return y

###################################################

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=False):
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

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
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

        nb_Conv = 2
        ghost=True
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=nb_Conv, ghost=ghost)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=nb_Conv, ghost=ghost)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=nb_Conv, ghost=ghost)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=nb_Conv, ghost=ghost)

        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2, ghost=ghost)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2, ghost=ghost)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2, ghost=ghost)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2, ghost=ghost)

        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        

        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        b, c, h, w = x.shape
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

        logits = self.outc(up1)

        return logits




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet18
# import torchvision

# class UNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2):
#         super(UNet, self).__init__()
#         self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)

#     def forward(self, x):
#         # input_size = x.shape[-2:]
#         # # x = torch.cat([x, x, x], dim=1)  # 扩充为3通道
#         # x = self.conv1(x)
#         # x = self.bn1(x)
#         # x = self.relu(x)
#         # x = self.maxpool(x)

#         # x = self.layer1(x)
#         # x = self.layer2(x)
#         # x = self.layer3(x)
#         # x = self.layer4(x)
#         # x = self.last_conv(x)
#         # x = self.up(x)
#         return self.model(x)['out']






import torch
import torch.nn as nn
from functools import partial
from torchvision import models as resnet_model
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










# class FAMBlock(nn.Module):
#     def __init__(self, channels):
#         super(FAMBlock, self).__init__()

#         self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
#         self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

#         self.relu3 = nn.ReLU(inplace=True)
#         self.relu1 = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x3 = self.conv3(x)
#         x3 = self.relu3(x3)
#         x1 = self.conv1(x)
#         x1 = self.relu1(x1)
#         out = x3 + x1

#         return out


# class DecoderBottleneckLayer(nn.Module):
#     def __init__(self, in_channels, n_filters, use_transpose=True):
#         super(DecoderBottleneckLayer, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         self.relu1 = nn.ReLU(inplace=True)

#         if use_transpose:
#             self.up = nn.Sequential(
#                 nn.ConvTranspose2d(
#                     in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
#                 ),
#                 nn.BatchNorm2d(in_channels // 4),
#                 nn.ReLU(inplace=True)
#             )
#         else:
#             self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

#         self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
#         self.norm3 = nn.BatchNorm2d(n_filters)
#         self.relu3 = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)
#         x = self.up(x)
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.relu3(x)
#         return x


# class SEBlock(nn.Module):
#     def __init__(self, channel, r=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // r, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // r, channel, bias=False),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         # Squeeze
#         y = self.avg_pool(x).view(b, c)
#         # Excitation
#         y = self.fc(y).view(b, c, 1, 1)
#         # Fusion
#         y = torch.mul(x, y)
#         return y


# class UNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2):
#         super(UNet, self).__init__()

#         # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
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






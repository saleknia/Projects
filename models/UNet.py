# import torch.nn as nn
# import torch
# import torchvision
# import torch.nn.functional as F
# from .CTrans import ChannelTransformer
# from .GT_UNet import *
# from .GT_UNet import _make_bot_layer 
# import numpy as np
# from torch.nn import init
# from torch.nn import Softmax
# import math
# from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d,NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
# from torch.nn import functional as F
# from torch.autograd import Variable
# import torchvision.ops

# class DeformableConv2d(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=1,
#                  bias=False):

#         super(DeformableConv2d, self).__init__()
        
#         assert type(kernel_size) == tuple or type(kernel_size) == int

#         kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
#         self.stride = stride if type(stride) == tuple else (stride, stride)
#         self.padding = padding
        
#         self.offset_conv = nn.Conv2d(in_channels, 
#                                      2 * kernel_size[0] * kernel_size[1],
#                                      kernel_size=kernel_size, 
#                                      stride=stride,
#                                      padding=self.padding, 
#                                      bias=True)

#         nn.init.constant_(self.offset_conv.weight, 0.)
#         nn.init.constant_(self.offset_conv.bias, 0.)
        
#         self.modulator_conv = nn.Conv2d(in_channels, 
#                                      1 * kernel_size[0] * kernel_size[1],
#                                      kernel_size=kernel_size, 
#                                      stride=stride,
#                                      padding=self.padding, 
#                                      bias=True)

#         nn.init.constant_(self.modulator_conv.weight, 0.)
#         nn.init.constant_(self.modulator_conv.bias, 0.)
        
#         self.regular_conv = nn.Conv2d(in_channels=in_channels,
#                                       out_channels=out_channels,
#                                       kernel_size=kernel_size,
#                                       stride=stride,
#                                       padding=self.padding,
#                                       bias=bias)

#     def forward(self, x):
#         #h, w = x.shape[2:]
#         #max_offset = max(h, w)/4.

#         offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
#         modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
#         x = torchvision.ops.deform_conv2d(input=x, 
#                                           offset=offset, 
#                                           weight=self.regular_conv.weight, 
#                                           bias=self.regular_conv.bias, 
#                                           padding=self.padding,
#                                           mask=modulator,
#                                           stride=self.stride,
#                                           )
#         return x


# class ParallelPolarizedSelfAttention(nn.Module):

#     def __init__(self, channel=512):
#         super().__init__()
#         self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
#         self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
#         self.softmax_channel=nn.Softmax(1)
#         self.softmax_spatial=nn.Softmax(-1)
#         self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
#         self.ln=nn.LayerNorm(channel)
#         self.sigmoid=nn.Sigmoid()
#         self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
#         self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
#         self.agp=nn.AdaptiveAvgPool2d((1,1))

#     def forward(self, x):
#         b, c, h, w = x.size()

#         #Channel-only Self-Attention
#         channel_wv=self.ch_wv(x) #bs,c//2,h,w
#         channel_wq=self.ch_wq(x) #bs,1,h,w
#         channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
#         channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
#         channel_wq=self.softmax_channel(channel_wq)
#         channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
#         channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
#         channel_out=channel_weight*x

#         #Spatial-only Self-Attention
#         spatial_wv=self.sp_wv(x) #bs,c//2,h,w
#         spatial_wq=self.sp_wq(x) #bs,c//2,h,w
#         spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
#         spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
#         spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
#         spatial_wq=self.softmax_spatial(spatial_wq)
#         spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
#         spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
#         spatial_out=spatial_weight*x
#         out=spatial_out+channel_out
#         return out

# # class SEAttention(nn.Module):

# #     def __init__(self, channel=512,reduction=16):
# #         super().__init__()
# #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
# #         self.fc = nn.Sequential(
# #             nn.Linear(channel, channel // reduction, bias=False),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(channel // reduction, channel, bias=False),
# #             nn.Sigmoid()
# #         )


# #     def init_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 init.kaiming_normal_(m.weight, mode='fan_out')
# #                 if m.bias is not None:
# #                     init.constant_(m.bias, 0)
# #             elif isinstance(m, nn.BatchNorm2d):
# #                 init.constant_(m.weight, 1)
# #                 init.constant_(m.bias, 0)
# #             elif isinstance(m, nn.Linear):
# #                 init.normal_(m.weight, std=0.001)
# #                 if m.bias is not None:
# #                     init.constant_(m.bias, 0)

# #     def forward(self, skip_x_att , skip_x):
# #         b, c, _, _ = skip_x_att.size()
# #         y = self.avg_pool(skip_x_att).view(b, c)
# #         y = self.fc(y).view(b, c, 1, 1)
# #         return skip_x * y.expand_as(skip_x)

# class SEAttention(nn.Module):

#     def __init__(self, channel=512,reduction=8):
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

#     def forward(self, decoder, encoder):
#         b, c, _, _ = decoder.size()
#         y = self.avg_pool(decoder).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return encoder * y.expand_as(encoder)

# class UpConv(nn.Module):

#     def __init__(self, in_channels, out_channels):
#         super(UpConv, self).__init__()

#         self.up = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x

# class AttentionBlock(nn.Module):
#     """Attention block with learnable parameters"""

#     def __init__(self, F_g, F_l, n_coefficients):
#         """
#         :param F_g: number of feature maps (channels) in previous layer
#         :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
#         :param n_coefficients: number of learnable multi-dimensional attention coefficients
#         """
#         super(AttentionBlock, self).__init__()

#         self.W_gate = nn.Sequential(
#             nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(n_coefficients)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(n_coefficients)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, gate, skip_connection):
#         """
#         :param gate: gating signal from previous layer
#         :param skip_connection: activation from corresponding encoder layer
#         :return: output activations
#         """
#         g1 = self.W_gate(gate)
#         x1 = self.W_x(skip_connection)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         out = skip_connection * psi
#         return out

# class SpatialGroupEnhance(nn.Module):

#     def __init__(self, groups):
#         super().__init__()
#         self.groups=groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
#         self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
#         self.sig=nn.Sigmoid()
#         self.init_weights()


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

#     def forward(self, x):
#         b, c, h,w=x.shape
#         x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
#         xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
#         xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
#         t=xn.view(b*self.groups,-1) #bs*g,h*w

#         t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
#         std=t.std(dim=1,keepdim=True)+1e-5
#         t=t/std #bs*g,h*w
#         t=t.view(b,self.groups,h,w) #bs,g,h*w
        
#         t=t*self.weight+self.bias #bs,g,h*w
#         t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
#         x=x*self.sig(t)
#         x=x.view(b,c,h,w)

#         return x 


# def get_activation(activation_type):
#     activation_type = activation_type.lower()
#     if hasattr(nn, activation_type):
#         return getattr(nn, activation_type)()
#     else:
#         return nn.ReLU()

# def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', kernel_size=3, padding=1 ):
#     layers = []
#     layers.append(ConvBatchNorm(in_channels, out_channels, activation, kernel_size=kernel_size, padding=padding))

#     for _ in range(nb_Conv - 1):
#         layers.append(ConvBatchNorm(out_channels, out_channels, activation))
#     return nn.Sequential(*layers)

# class ConvBatchNorm(nn.Module):
#     """(convolution => [BN] => ReLU)"""

#     def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1):
#         super(ConvBatchNorm, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size=kernel_size, padding=padding)
#         self.norm = nn.BatchNorm2d(out_channels)
#         self.activation = get_activation(activation)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         return self.activation(out)

# class DConvBatchNorm(nn.Module):
#     """(convolution => [BN] => ReLU)"""

#     def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1):
#         super(DConvBatchNorm, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv = DeformableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
#         self.norm = nn.BatchNorm2d(out_channels)
#         self.activation = get_activation(activation)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         return self.activation(out)

# class DownBlock(nn.Module):
#     """Downscaling with maxpool convolution"""

#     def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
#         super(DownBlock, self).__init__()
#         self.maxpool = nn.MaxPool2d(2)
#         self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

#     def forward(self, x):
#         out = self.maxpool(x)
#         return self.nConvs(out)

# class BatchNorm(nn.Module):
#     """([BN] => ReLU)"""

#     def __init__(self, in_channels, activation='ReLU'):
#         super(BatchNorm, self).__init__()
#         self.norm = nn.BatchNorm2d(in_channels)
#         self.activation = get_activation(activation)

#     def forward(self, x):
#         out = self.norm(x)
#         return self.activation(out)

# class Conv(nn.Module):
#     """(convolution)"""
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)

#     def forward(self, x):
#         out = self.conv(x)
#         return out

# # class DownBlock_BN(nn.Module):
# #     """Downscaling with maxpool convolution"""

# #     def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', dilations=[2,3]):
# #         super(DownBlock_BN, self).__init__()
# #         self.maxpool = nn.MaxPool2d(2)
# #         self.shortcut = _make_nConv(in_channels, out_channels, nb_Conv, activation)
# #         self.mhsa = MHSA(in_channels=in_channels, heads=4, curr_h=8, curr_w=8,pos_enc_type='relative')
# #         # self.bottleneck_dimension = out_channels // 2

# #         # self.conv1 = Conv(in_channels, self.bottleneck_dimension ,kernel_size=3, padding=1, stride=1, dilation=1)
# #         # self.conv2 = Conv(in_channels, self.bottleneck_dimension ,kernel_size=3, padding=dilations[0], stride=1, dilation=dilations[0])
# #         # self.conv3 = Conv(in_channels, self.bottleneck_dimension ,kernel_size=3, padding=dilations[1], stride=1, dilation=dilations[1])

# #         # self.conv4 = Conv(self.bottleneck_dimension*2, out_channels, kernel_size=1, padding=0, stride=1)
# #         # self.conv5 = Conv(out_channels*2 ,out_channels , kernel_size=1, padding=0, stride=1)
# #         # self.BN_1 = BatchNorm(in_channels=out_channels, activation='ReLU')
# #         # self.BN_2 = BatchNorm(in_channels=out_channels, activation='ReLU')

# #     def forward(self, x):
# #         x = self.maxpool(x)
# #         out = self.shortcut(x)

# #         Q_h = Q_w = 8
# #         N, C, H, W = out.shape
# #         P_h, P_w = H // Q_h, W // Q_w

# #         out = out.reshape(N * P_h * P_w, C, Q_h, Q_w)

# #         out = self.mhsa(out)
# #         out = out.permute(0, 3, 1, 2)  # back to pytorch dim order
# #         N1, C1, H1, W1 = out.shape
# #         out = out.reshape(N, C1, int(H), int(W))


# #         # out = torch.cat((self.conv2(x),self.conv3(x)),dim=1)
# #         # out = self.conv4(out)
# #         # out = self.BN_1(out)

# #         # out = torch.cat((out,shortcut),dim=1)
# #         # out = self.conv5(out)
# #         # out = self.BN_2(out)
# #         # out = out + shortcut

# #         return out

# class PAM_Module(Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim

#         self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = Parameter(torch.zeros(1))

#         self.softmax = Softmax(dim=-1)
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)

#         out = self.gamma*out + x
#         return out

# class CAM_Module(nn.Module):
#     """ Channel attention module"""
#     def __init__(self):
#         super(CAM_Module, self).__init__()
#         # self.chanel_in = in_dim
#         self.gamma = nn.parameter.Parameter(torch.zeros(1))
#         self.softmax  = Softmax(dim=-1)

#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = x.view(m_batchsize, C, -1)
#         proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
#         energy = torch.bmm(proj_query, proj_key)
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         attention = self.softmax(energy_new)
#         proj_value = x.view(m_batchsize, C, -1)

#         out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, C, height, width)

#         out = self.gamma*out + x
#         return out

# class Indentity(nn.Module):

#     def __init__(self):
#         super(Indentity, self).__init__()
#     def forward(self, x):
#         return x


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

# class UpBlock(nn.Module):
#     """Upscaling then conv"""

#     def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU',PA=True):
#         super(UpBlock, self).__init__()
#         self.up_1 = nn.Upsample(scale_factor=2)
#         # self.up_2 = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
#         # self.gamma = nn.parameter.Parameter(torch.zeros(1))
#         self.nConvs = _make_nConv(in_channels=in_channels, out_channels=out_channels, nb_Conv=2, activation='ReLU')
#         # self.att = AttentionBlock(F_g=in_channels//2, F_l=in_channels//2, n_coefficients=in_channels//4)
#         # self.se = SEAttention(channel=in_channels//2, reduction=8)
#         # self.CA_skip = CAM_Module()
#         # self.CA_x = CAM_Module()
#         # self.PA = PA
#         # if self.PA:
#         #     self.PA = PAM_Module(in_dim=in_channels//2)

#         # self.att = ParallelPolarizedSelfAttention(channel = in_channels//2)

#         # self.nConvs_out = _make_nConv(in_channels=out_channels , out_channels=out_channels, nb_Conv=1, activation='ReLU')
#         # self.deconve = DConvBatchNorm(in_channels=in_channels//2, out_channels=in_channels//2)
#         # self.FAM = FAMBlock(channels=in_channels//2)

#     def forward(self, x, skip_x):
#         # skip_x = self.deconve(skip_x)  
#         # skip_x = self.FAM(skip_x)
#         out = self.up_1(x)
#         # skip_x = self.se(decoder=out, encoder=skip_x)
#         # out = self.att(out)
#         x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
#         x = self.nConvs(x) 
#         return x

# # class seg_head(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.scale_4 = nn.Upsample(scale_factor=8)
# #         self.scale_3 = nn.Upsample(scale_factor=4)
# #         self.scale_2 = nn.Upsample(scale_factor=2)
# #         self.conv_4 =  nn.Conv2d(64, 2, kernel_size=(1,1), stride=(1,1))
# #         self.conv_3 =  nn.Conv2d(32, 2, kernel_size=(1,1), stride=(1,1))
# #         self.conv_2 =  nn.Conv2d(16, 2, kernel_size=(1,1), stride=(1,1))
# #         self.conv_2 =  nn.Conv2d(16, 2, kernel_size=(1,1), stride=(1,1))

# #     def forward(self, up4, up3, up2, up1):
# #         # up2 = torchvision.ops.stochastic_depth(input=up2, p=0.5, mode='batch')
# #         # up3 = torchvision.ops.stochastic_depth(input=up3, p=0.5, mode='batch')
# #         # up4 = torchvision.ops.stochastic_depth(input=up4, p=0.5, mode='batch')
# #         up4 = self.scale_4(self.conv_4(up4))
# #         up3 = up3 + up4
# #         up3 = self.scale_3(self.conv_3(up3))
# #         up2 = up3 + up2
# #         up2 = self.scale_2(self.conv_2(up2))
# #         up = up2 + up1
        
# #         up = self.conv(up)
# #         up = self.BN_out(up)
# #         up = self.RELU6_out(up)
# #         up = self.out(up)

# #         return up


# class seg_head(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.scale_4 = nn.Upsample(scale_factor=8)
#         self.scale_3 = nn.Upsample(scale_factor=4)
#         self.scale_2 = nn.Upsample(scale_factor=2)
#         self.conv_4 =  nn.Conv2d(64, 2, kernel_size=(1,1), stride=(1,1))
#         self.conv_3 =  nn.Conv2d(32, 2, kernel_size=(1,1), stride=(1,1))
#         self.conv_2 =  nn.Conv2d(16, 2, kernel_size=(1,1), stride=(1,1))
#         self.conv_1 =  nn.Conv2d(16, 2, kernel_size=(1,1), stride=(1,1))

#     def forward(self, up4, up3, up2, up1):
#         up4 = self.scale_4(self.conv_4(up4))
#         up3 = self.scale_3(self.conv_3(up3))
#         up2 = self.scale_2(self.conv_2(up2))
#         up1 = self.conv_1(up1)
#         up = up4 + up3 + up2 + up1
#         return up

# class UNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=9):
#         '''
#         n_channels : number of channels of the input.
#                         By default 3, because we have RGB images
#         n_labels : number of channels of the ouput.
#                       By default 3 (2 labels + 1 for the background)
#         '''
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
#         # Question here
#         in_channels = 16

#         self.inc = ConvBatchNorm(n_channels, in_channels)
#         self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=4)
#         self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=4)
#         self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=4)
#         self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=4)

#         self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2, PA=False)
#         self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2, PA=False)
#         self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2, PA=False)
#         self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2, PA=False)
#         self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
#         self.seg_head = seg_head()

#         if n_classes == 1:
#             self.last_activation = nn.Sigmoid()
#         else:
#             self.last_activation = None

#     def forward(self, x):
#         # Question here
#         x = x.float()
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)


#         up4 = self.up4(x5 , x4)
#         up3 = self.up3(up4, x3)
#         up2 = self.up2(up3, x2)
#         up1 = self.up1(up2, x1)

#         # logits = self.seg_head(up4, up3, up2, up1)
#         logits = self.outc(up1)
#         return logits


#         # if self.last_activation is not None:
#         #     logits = self.last_activation(self.outc(up1))
#         # else:
#         #     logits = self.outc(up1)

#         # if self.training:
#         #     return logits, up4, up3, up2, up1
#         # else:
#         #     return logits


# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torchvision.models import resnet18
# # import torchvision

# # class UNet(nn.Module):
# #     def __init__(self, n_channels=3, n_classes=2):
# #         super(UNet, self).__init__()
# #         self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)

# #     def forward(self, x):
# #         # input_size = x.shape[-2:]
# #         # # x = torch.cat([x, x, x], dim=1)  # 扩充为3通道
# #         # x = self.conv1(x)
# #         # x = self.bn1(x)
# #         # x = self.relu(x)
# #         # x = self.maxpool(x)

# #         # x = self.layer1(x)
# #         # x = self.layer2(x)
# #         # x = self.layer3(x)
# #         # x = self.layer4(x)
# #         # x = self.last_conv(x)
# #         # x = self.up(x)
# #         return self.model(x)['out']





# timm

import torch
from torchvision import models as resnet_model
from torch import nn


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
        super(UNet, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        transformer = torch.load('/content/drive/MyDrive/FATNet/transformer.pth')
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)


    def forward(self, x):
        b, c, h, w = x.shape

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        emb = self.patch_embed(x)
        for i in range(12):
            emb = self.transformers[i](emb)
        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 192, 14, 14)
        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out






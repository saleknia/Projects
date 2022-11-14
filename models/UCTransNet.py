import torch.nn as nn
import torch
import timm
import torchvision
import torch.nn.functional as F
from .CTrans import ChannelTransformer
from torchvision import models as resnet_model
import numpy as np
from torch.nn import init
from torch.nn import Softmax

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

    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        # self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        # skip_x_att = self.coatt(g=up, x=skip_x)
        skip_x_att = skip_x
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class UCTransNet(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=256,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        
        # resnet = resnet_model.resnet50(pretrained=True)
        resnet = torchvision.models.segmentation.deeplabv3_resnet50(pretrain=True).backbone
        resnet.conv1.stride = (1, 1)
        resnet.backbone.layer3[0].downsample[0].stride = (2, 2)
        resnet.backbone.layer4[0].downsample[0].stride = (2, 2)
        resnet.backbone.layer3[1].conv2.dilation = (1, 1)
        resnet.backbone.layer3[1].conv2.padding = (1, 1)
        resnet.backbone.layer3[2].conv2.dilation = (1, 1)
        resnet.backbone.layer3[2].conv2.padding = (1, 1)
        resnet.backbone.layer3[3].conv2.dilation = (1, 1)
        resnet.backbone.layer3[3].conv2.padding = (1, 1)
        resnet.backbone.layer3[4].conv2.dilation = (1, 1)
        resnet.backbone.layer3[4].conv2.padding = (1, 1)
        resnet.backbone.layer3[5].conv2.dilation = (1, 1)
        resnet.backbone.layer3[5].conv2.padding = (1, 1)
        resnet.backbone.layer4[0].conv2.dilation = (1, 1)
        resnet.backbone.layer4[0].conv2.padding = (1, 1)
        resnet.backbone.layer4[1].conv2.dilation = (1, 1)
        resnet.backbone.layer4[1].conv2.padding = (1, 1)
        resnet.backbone.layer4[2].conv2.dilation = (1, 1)
        resnet.backbone.layer4[2].conv2.padding = (1, 1)
        self.inc = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        # (224,224) , 64

        self.down1 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        )
        # (112,112) , 64

        self.down2 = resnet.layer2 
        # (56,56) , 128

        self.down3 = resnet.layer3 # 256
        # (28,28) , 256

        self.down4 = resnet.layer4 # 512
        # (14,14) , 512

        self.reduce_3 = ConvBatchNorm(in_channels=512 , out_channels=128 , activation='ReLU', kernel_size=1, padding=0)
        self.reduce_4 = ConvBatchNorm(in_channels=1024, out_channels=128 , activation='ReLU', kernel_size=1, padding=0)
        self.reduce_5 = ConvBatchNorm(in_channels=2048, out_channels=128 , activation='ReLU', kernel_size=1, padding=0)

        self.fam3 = ConvBatchNorm(in_channels=256, out_channels=128, activation='ReLU', kernel_size=3, padding=1)
        self.fam4 = ConvBatchNorm(in_channels=256, out_channels=128, activation='ReLU', kernel_size=3, padding=1)
        self.fam5 = ConvBatchNorm(in_channels=256, out_channels=128, activation='ReLU', kernel_size=3, padding=1)

        self.pam3 = ConvBatchNorm(in_channels=256, out_channels=128, activation='ReLU', kernel_size=3, padding=1)
        self.pam4 = ConvBatchNorm(in_channels=256, out_channels=128, activation='ReLU', kernel_size=3, padding=1)

        self.mtc = ChannelTransformer(config, vis, img_size,channel_num=[128, 128, 128],patchSize=config.patch_sizes)

        self.up_5 = nn.Upsample(scale_factor=2)
        self.up_4 = nn.Upsample(scale_factor=2)
        self.up_3 = nn.Upsample(scale_factor=4)

        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))


    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x3 = self.reduce_3(x3)
        x4 = self.reduce_4(x4)
        x5 = self.reduce_5(x5)

        t3, t4, t5, att_weights = self.mtc(x3,x4,x5)

        t5 = torch.cat([x5, t5], dim=1)
        t5 = self.fam5(t5) 

        t4 = torch.cat([x4, t4], dim=1)
        t4 = self.fam4(t4) 

        t3 = torch.cat([x3, t3], dim=1)
        t3 = self.fam3(t3) 

        t5 = self.up_5(t5)
        t4 = torch.cat([t4, t5], dim=1)
        t4 = self.pam4(t4) 

        t4 = self.up_4(t4)
        t3 = torch.cat([t3, t4], dim=1)
        t3 = self.pam3(t3) 

        logits = self.up_3(self.outc(t3))

        return logits



# class UCTransNet(nn.Module):
#     def __init__(self, config,n_channels=3, n_classes=1,img_size=256,vis=False):
#         super().__init__()
#         self.vis = vis
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         in_channels = config.base_channel

#         model = torchvision.models.segmentation.deeplabv3_resnet50(pretrain=True).backbone

#         self.inc = nn.Sequential(
#                                 model.conv1  ,
#                                 model.bn1    ,
#                                 model.relu   ,
#                                 model.maxpool,
#                             )
#         # torch.Size([8, 64, 56, 56])

#         self.layer1 = model.layer1
#         # torch.Size([8, 256, 56, 56])

#         self.layer2 = model.layer2
#         # torch.Size([8, 512, 28, 28])

#         self.layer3 = model.layer3
#         # torch.Size([8, 1024, 28, 28])

#         self.layer4 = model.layer4
#         # torch.Size([8, 2048, 28, 28])   


#         self.reduce_1 = ConvBatchNorm(in_channels=256  , out_channels=128 , activation='ReLU', kernel_size=1, padding=0)

#         self.reduce_2 = ConvBatchNorm(in_channels=512  , out_channels=128 , activation='ReLU', kernel_size=1, padding=0)
#         self.reduce_3 = ConvBatchNorm(in_channels=1024 , out_channels=128 , activation='ReLU', kernel_size=1, padding=0)
#         self.reduce_4 = ConvBatchNorm(in_channels=2048 , out_channels=128 , activation='ReLU', kernel_size=1, padding=0)

#         self.mtc = ChannelTransformer(config, vis, img_size,channel_num=[in_channels, in_channels, in_channels],patchSize=config.patch_sizes)

#         self.fam2 = ConvBatchNorm(in_channels=256, out_channels=128, activation='ReLU', kernel_size=3, padding=1)
#         self.fam3 = ConvBatchNorm(in_channels=256, out_channels=128, activation='ReLU', kernel_size=3, padding=1)
#         self.fam4 = ConvBatchNorm(in_channels=256, out_channels=128, activation='ReLU', kernel_size=3, padding=1)

#         self.fusion_1 = ConvBatchNorm(in_channels=384, out_channels=128, activation='ReLU', kernel_size=3, padding=1)
#         self.fusion_2 = ConvBatchNorm(in_channels=256, out_channels=128, activation='ReLU', kernel_size=3, padding=1)      
        
#         self.outc = nn.Conv2d(in_channels*2, n_classes, kernel_size=(1,1), stride=(1,1))

#         self.up_int = nn.Upsample(scale_factor=2)
#         self.up_out = nn.Upsample(scale_factor=4)


#     def forward(self, x):
#         x = x.float()

#         inc = self.inc(x)
#         layer1 = self.layer1(inc)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)

#         layer1 = self.reduce_1(layer1)

#         layer2 = self.reduce_2(layer2)
#         layer3 = self.reduce_3(layer3)
#         layer4 = self.reduce_4(layer4)

#         t2, t3, t4, att_weights = self.mtc(layer2, layer3, layer4)

#         t4 = torch.cat([layer4, t4], dim=1)
#         t4 = self.fam4(t4) 

#         t3 = torch.cat([layer3, t3], dim=1)
#         t3 = self.fam3(t3) 

#         t2 = torch.cat([layer2, t2], dim=1)
#         t2 = self.fam2(t2) 

#         x = torch.cat([t4, t3, t2], dim=1)
#         x = self.fusion_1(x) 
#         x = self.up_int(x)

#         x = torch.cat([x, layer1], dim=1)

#         x = self.outc(x)
#         x = self.up_out(x)

#         return x



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
        n = int(nOut / 5)  # K=5,
        n1 = nOut - 4 * n  # (N-(K-1)INT(N/K)) for dilation rate of 2^0, for producing an output feature map of channel=nOut
        self.c1 = C(nIn, n, 1, 1)  # the point-wise convolutions with 1x1 help in reducing the computation, channel=c

        # K=5, dilation rate: 2^{k-1},k={1,2,3,...,K}
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n , 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n , 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n , 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n , 3, 1, 16)  # dilation rate of 2^4
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
        d8 = self.d8(output1)
        d16 = self.d8(output1)

        # Using hierarchical feature fusion (HFF) to ease the gridding artifacts which is introduced
        # by the large effective receptive filed of the ESP module
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output




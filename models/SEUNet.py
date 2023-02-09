from torchvision import models as resnet_model
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).to('cuda').unsqueeze(0).repeat(B*W,1,1)

class CCA(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels):
        super(CCA, self).__init__()
        self.CCA_1 = CrissCrossAttention(in_channels)
        self.CCA_2 = CrissCrossAttention(in_channels)

    def forward(self, x):
        x = self.CCA_1(x)
        x = self.CCA_2(x)
        return x

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return (out_H + out_W) + x


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', reduce=False, reduction_rate=1):
    layers = []
    if reduce:
        layers.append(ConvBatchNorm_r(in_channels, out_channels, activation, reduction_rate))
    else:
        layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        if reduce:
            layers.append(ConvBatchNorm_r(out_channels, out_channels, activation, reduction_rate))
        else:
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

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', reduce=False, reduction_rate=1):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, reduce=reduce, reduction_rate=reduction_rate)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', reduce=False, reduction_rate=1):
        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, reduce=reduce, reduction_rate=reduction_rate)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

from .CTrans import ChannelTransformer
import ml_collections
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 448  # KV_size = Q1 + Q2 + Q3 
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate  = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [4,2,1]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config
import timm
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

        # resnet = resnet_model.resnet34(pretrained=True)

        # self.firstconv = resnet.conv1
        # self.firstbn   = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.maxpool   = resnet.maxpool 
        # self.encoder1  = resnet.layer1
        # self.encoder2  = resnet.layer2
        # self.encoder3  = resnet.layer3
        # self.encoder4  = resnet.layer4

        channel = 18

        self.encoder = timm.create_model('hrnet_w18', pretrained=True, features_only=True)
        self.encoder.incre_modules = None

        self.up3 = UpBlock(channel*8, channel*4, nb_Conv=2)
        self.up2 = UpBlock(channel*4, channel*2, nb_Conv=2)
        self.up1 = UpBlock(channel*2, channel*1, nb_Conv=2)

        # self.up3 = UpBlock(in_channels=512, out_channels=256, nb_Conv=2)
        # self.up2 = UpBlock(in_channels=256, out_channels=128, nb_Conv=2)
        # self.up1 = UpBlock(in_channels=128, out_channels=64 , nb_Conv=2)


        # self.final_conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        # self.final_relu1 = nn.ReLU(inplace=True)
        # self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.final_relu2 = nn.ReLU(inplace=True)
        # self.final_conv3 = nn.ConvTranspose2d(32, n_classes, kernel_size=2, stride=2)

        self.final_conv1 = nn.ConvTranspose2d(18, 18, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(18, 18, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.ConvTranspose2d(18, n_classes, kernel_size=2, stride=2)

    def forward(self, x):
        b, c, h, w = x.shape

        x = torch.cat([x, x, x], dim=1)

        # e0 = self.firstconv(x)
        # e0 = self.firstbn(e0)
        # e0 = self.firstrelu(e0)
        # e0 = self.maxpool(e0)

        # e1 = self.encoder1(e0)
        # e2 = self.encoder2(e1)
        # e3 = self.encoder3(e2)
        # e4 = self.encoder4(e3)

        x = self.encoder.conv1(x)
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
        yl = self.encoder.stage4(xl)

        e1, e2, e3, e4 = yl[0], yl[1], yl[2], yl[3]

        e = self.up3(e4, e3)
        e = self.up2(e , e2)
        e = self.up1(e , e1)

        e = self.final_conv1(e)
        e = self.final_relu1(e)
        e = self.final_conv2(e)
        e = self.final_relu2(e)
        e = self.final_conv3(e)

        return e



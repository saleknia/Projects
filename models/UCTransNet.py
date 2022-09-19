import torch.nn as nn
import torch
import torch.nn.functional as F
from .CTrans import ChannelTransformer

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

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
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
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        # self.outc_1 = nn.Conv2d(64 , n_classes, kernel_size=1, stride=1, padding=0)
        # self.outc_2 = nn.Conv2d(64 , n_classes, kernel_size=1, stride=1, padding=0)
        # self.outc_3 = nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0)
        # self.outc_4 = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)
        # self.att_2 = ParallelPolarizedSelfAttention(channel = 64)
        # self.att_3 = ParallelPolarizedSelfAttention(channel = 128)
        self.last_activation = nn.Sigmoid() # if using BCELoss

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1,x2,x3,x4,att_weights, probs1, probs2, probs3, probs4 = self.mtc(x1,x2,x3,x4)

        # x = self.up4(x5, x4)
        # x = self.up3(x, x3)
        # x = self.up2(x, x2)
        # x = self.up1(x, x1)

        up4 = self.up4(x5, x4)
        up3 = self.up3(up4, x3)
        up2 = self.up2(up3, x2)
        up1 = self.up1(up2, x1)

        logits = self.outc(up1)
        # logits = self.outc_1(up1)
        # # logits = logits + self.att_2(F.interpolate(self.outc_2(up2), x.size()[2:], mode='bilinear', align_corners=False))
        # # logits = logits + self.att_3(F.interpolate(self.outc_3(up3), x.size()[2:], mode='bilinear', align_corners=False)) 
        # logits = logits + self.outc_2(self.att_2(F.interpolate(up2, x.size()[2:], mode='bilinear', align_corners=False)))
        # logits = logits + self.outc_3(self.att_3(F.interpolate(up3, x.size()[2:], mode='bilinear', align_corners=False))) 
        # # logits = logits + F.interpolate(self.outc_4(up4), x.size()[2:], mode='bilinear', align_corners=False) 

        return logits
        # if self.training:
        #     # return logits, probs1, probs2, probs3, probs4, up4, up3, up2, up1
        #     return logits, up4, up3, up2, up1, x5

        # else:
        #     return logits

        # if self.n_classes ==1:
        #     logits = self.last_activation(self.outc(x))
        # else:
        #     logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1
            
        # return logits

        # if self.training:
        #     if self.n_classes ==1:
        #         logits = self.last_activation(self.outc(up1))
        #     else:
        #         logits = self.outc(up1) # if nusing BCEWithLogitsLoss or class>1
                
        #     if self.vis: # visualize the attention maps
        #         return logits, x4, x3, x2, x1
        #     else:
        #         return logits, up4, up3, up2, up1

        # else:
        #     if self.n_classes ==1:
        #         logits = self.last_activation(self.outc(up1))
        #     else:
        #         logits = self.outc(up1) # if nusing BCEWithLogitsLoss or class>1
        #     if self.vis: # visualize the attention maps
        #         return logits
        #     else:
        #         return logits

        # if self.training:
        #     if self.n_classes ==1:
        #         logits = self.last_activation(self.outc(up1))
        #     else:
        #         logits = self.outc(up1) # if nusing BCEWithLogitsLoss or class>1

        #     if self.vis: # visualize the attention maps
        #         return logits, up4, up3, up2, up1
        #     else:
        #         return logits, up4, up3, up2, up1

        # else:
        #     if self.n_classes ==1:
        #         logits = self.last_activation(self.outc(up1))
        #     else:
        #         logits = self.outc(up1) # if nusing BCEWithLogitsLoss or class>1
        #     if self.vis: # visualize the attention maps
        #         return logits
        #     else:
        #         return logits

        # if self.training:
        #     if self.n_classes ==1:
        #         logits = self.last_activation(self.outc(x))
        #     else:
        #         logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1

        #     if self.vis: # visualize the attention maps
        #         return logits, x
        #     else:
        #         return logits, x

        # else:
        #     if self.n_classes ==1:
        #         logits = self.last_activation(self.outc(x))
        #     else:
        #         logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1
        #     if self.vis: # visualize the attention maps
        #         return logits
        #     else:
        #         return logits




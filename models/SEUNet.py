from torchvision import models as resnet_model
import torch.nn as nn
import torch

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

class ConvBatchNorm_r(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', reduction_rate=1):
        super(ConvBatchNorm_r, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels//reduction_rate,kernel_size=3, padding=1)
        self.norm_1 = nn.BatchNorm2d(out_channels//reduction_rate)
        self.activation_1 = get_activation(activation)

        self.conv_2 = nn.Conv2d(out_channels//reduction_rate, out_channels, kernel_size=1, padding=0)
        self.norm_2 = nn.BatchNorm2d(out_channels)
        self.activation_2 = get_activation(activation)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.activation_2(x)
        return x

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', reduce=False, reduction_rate=1):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, reduce=reduce, reduction_rate=reduction_rate)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class SKAttention(nn.Module):

    def __init__(self, channel=512, reduction=4, group=1):
        super().__init__()
        self.d=channel//reduction
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(2):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x, y):
        bs, c, _, _ = x.size()
        conv_outs=[x, y]
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
        attention_weights=torch.stack(weights,0)  #k,bs,channel,1,1
        attention_weights=torch.sigmoid(attention_weights)#k,bs,channel,1,1

        ### fuse
        V=(attention_weights*feats)
        V=torch.cat([feats[0], feats[1]], dim=1)
        return V

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', reduce=False, reduction_rate=1):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, reduce=reduce, reduction_rate=reduction_rate)
        self.att = SKAttention(in_channels//2)

    def forward(self, x, skip_x):
        out = self.up(x)
        # x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        x = self.att(out, skip_x)
        return self.nConvs(x)

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

        self.up3 = UpBlock(in_channels=512, out_channels=256, nb_Conv=2)
        self.up2 = UpBlock(in_channels=256, out_channels=128, nb_Conv=2)
        self.up1 = UpBlock(in_channels=128, out_channels=64 , nb_Conv=2)

        self.final_conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.ConvTranspose2d(32, n_classes, kernel_size=2, stride=2)

    def forward(self, x):
        b, c, h, w = x.shape

        x = torch.cat([x, x, x], dim=1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e0 = self.maxpool(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e = self.up3(e4, e3)
        e = self.up2(e , e2)
        e = self.up1(e , e1)

        e = self.final_conv1(e)
        e = self.final_relu1(e)
        e = self.final_conv2(e)
        e = self.final_relu2(e)
        e = self.final_conv3(e)

        return e









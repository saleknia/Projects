import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b5, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import random
from torch.nn import init
from .Mobile_netV2_loss import Mobile_netV2_loss

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import timm
from .wideresnet import *
from .wideresnet import recursion_change_bn
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.models import ModelBuilder
from mit_semseg.models import ModelBuilder, SegmentationModule

class SEUNet(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(SEUNet, self).__init__()

        # model_0 = models.__dict__['resnet50'](num_classes=365)

        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # model_0.load_state_dict(state_dict)

        # # model_1 = models.__dict__['resnet18'](num_classes=365)

        # # checkpoint = torch.load('/content/resnet18_places365.pth.tar', map_location='cpu')
        # # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # # model_1.load_state_dict(state_dict)

        # # model_2 = models.__dict__['densenet161'](num_classes=365)

        # # checkpoint = torch.load('/content/densenet161_places365.pth.tar', map_location='cpu')
        # # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # # state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
        # # state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}
        # # model_2.load_state_dict(state_dict)

        # for param in model_0.parameters():
        #     param.requires_grad = False

        # # for param in model_1.parameters():
        # #     param.requires_grad = False

        # # for param in model_2.parameters():
        # #     param.requires_grad = False


        # # for param in model_0.layer4.parameters():
        # #     param.requires_grad = True

        # # for param in model_1.layer4.parameters():
        # #     param.requires_grad = True

        # # for param in model_2.features.denseblock4.parameters():
        # #     param.requires_grad = True


        # self.conv1   = model_0.conv1
        # self.bn1     = model_0.bn1
        # self.relu    = model_0.relu 
        # self.maxpool = model_0.maxpool

        # self.layer1 = model_0.layer1
        # self.layer2 = model_0.layer2
        # self.layer3 = model_0.layer3
        # self.layer4 = model_0.layer4

        # # self.layer40 = model_0.layer4
        # # self.layer41 = model_1.layer4
        # # self.layer42 = model_2.features.denseblock4

        # self.avgpool = model_0.avgpool

        # self.fc_0 = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=2048, out_features=67, bias=True))

        # self.fc_1 = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=2048, out_features=67, bias=True))

        # self.fc_2 = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=2048, out_features=67, bias=True))

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_res_50_mit/SEUNet_MIT-67_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # self.load_state_dict(pretrained_teacher)

        # self.net_encoder = ModelBuilder.build_encoder(arch='resnet50' ,fc_dim=2048, weights='/content/encoder_epoch_30.pth')
        # for param in self.net_encoder.parameters():
        #     param.requires_grad = False

        net_encoder = ModelBuilder.build_encoder(
            arch='resnet50dilated',
            fc_dim=2048,
            weights='/content/encoder_epoch_20.pth')
        net_decoder = ModelBuilder.build_decoder(
            arch='ppm_deepsup',
            fc_dim=2048,
            num_class=150,
            weights='/content/decoder_epoch_20.pth',
            use_softmax=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).eval().cuda()

        for param in self.segmentation_module.parameters():
            param.requires_grad = False

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        self.features = model.features
        self.features[0][0].in_channels = 4
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=67, bias=True))

    def forward(self, x0):
        b, c, w, h = x0.shape

        # x = self.conv1(x0)
        # x = self.bn1(x)   
        # x = self.relu(x)  
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # seg = self.net_encoder(x0)[0]

        # # x0 = self.layer40(x)
        # # x0 = self.avgpool(x0)
        # # x0 = x0.view(x0.size(0), -1)
        # # x0 = self.fc_0(x0)

        # # x1 = self.res18_adapter(x)
        # # x1 = self.layer41(x1)
        # # x1 = self.avgpool(x1)
        # # x1 = x1.view(x1.size(0), -1)
        # # x1 = self.fc_1(x1)

        # # x = self.avgpool(x)
        # # x = x.view(x.size(0), -1)
        # # x = self.fc_0(x)

        # # x = self.avgpool(x)
        # # x = x.view(x.size(0), -1)
        # # x = self.fc_1(x)

        # x = self.avgpool(x + seg)
        # x = x.view(x.size(0), -1)
        # x = self.fc_2(x)

        with torch.no_grad():
            singleton_batch = {'img_data': x0}
            y = self.segmentation_module(singleton_batch, segSize=(224,224))
            predictions = torch.argmax(input=y,dim=1).long() / 150.0
            predictions = torch.cat([predictions.unsqueeze(dim=1), x0], dim=1)

        x = self.features(predictions)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # x = self.avgpool(x3)

        # x = self.classifier(x)


        return x





def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

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







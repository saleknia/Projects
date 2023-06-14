import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights
import torchvision
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import ttach as tta


import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

from .wideresnet import *
from .wideresnet import recursion_change_bn

from mit_semseg.models import ModelBuilder



class Mobile_netV2_loss(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_loss, self).__init__()
        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.b_0 = Mobile_netV2_0()
        loaded_data_b_0 = torch.load('/content/drive/MyDrive/checkpoint_B0_81_68/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_0 = loaded_data_b_0['net']

        a = pretrained_b_0.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_0.pop(key)

        self.b_0.load_state_dict(pretrained_b_0)
        self.b_0 = self.b_0.eval()

        self.b_1 = Mobile_netV2_1()
        loaded_data_b_1 = torch.load('/content/drive/MyDrive/checkpoint_B1_82_72/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_1 = loaded_data_b_1['net']

        a = pretrained_b_1.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_1.pop(key)

        self.b_1.load_state_dict(pretrained_b_1)
        self.b_1 = self.b_1.eval()
        
        self.b_2 = Mobile_netV2_2()
        loaded_data_b_2 = torch.load('/content/drive/MyDrive/checkpoint_B2_83_02/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_2 = loaded_data_b_2['net']

        a = pretrained_b_2.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_2.pop(key)

        self.b_2.load_state_dict(pretrained_b_2)
        self.b_2 = self.b_2.eval()

        self.b_3 = Mobile_netV2_3()
        loaded_data_b_3 = torch.load('/content/drive/MyDrive/checkpoint_B3_83_84/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_3 = loaded_data_b_3['net']

        a = pretrained_b_3.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_3.pop(key)

        self.b_3.load_state_dict(pretrained_b_3)
        self.b_3 = self.b_3.eval()

        self.res_18 = Mobile_netV2_res_18()
        loaded_data_res_18 = torch.load('/content/drive/MyDrive/checkpoint_res_18_81_97/Mobile_NetV2_MIT-67_best.pth', map_location='cpu')
        pretrained_res_18 = loaded_data_res_18['net']

        self.res_18.load_state_dict(pretrained_res_18)
        self.res_18 = self.res_18.eval()


        self.res_50 = Mobile_netV2_res_50()
        loaded_data_res_50 = torch.load('/content/drive/MyDrive/checkpoint_res_50_84_14/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_res_50 = loaded_data_res_50['net']

        self.res_50.load_state_dict(pretrained_res_50)
        self.res_50 = self.res_50.eval()


        self.dense = Mobile_netV2_dense()
        loaded_data_dense = torch.load('/content/drive/MyDrive/checkpoint_dense_85_04/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_dense = loaded_data_dense['net']

        self.dense.load_state_dict(pretrained_dense)
        self.dense = self.dense.eval()

        # self.b_0 = tta.ClassificationTTAWrapper(self.b_0, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')
        # self.b_1 = tta.ClassificationTTAWrapper(self.b_1, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')
        # self.b_2 = tta.ClassificationTTAWrapper(self.b_2, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')

        # self.res_18 = tta.ClassificationTTAWrapper(self.res_18, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')
        # self.res_50 = tta.ClassificationTTAWrapper(self.res_50, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')
        # self.dense = tta.ClassificationTTAWrapper(self.dense, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')

        # self.seg = tta.ClassificationTTAWrapper(self.seg, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')


    def forward(self, x):
        b, c, w, h = x.shape

        x0 = self.b_0(x)
        x1 = self.b_1(x) 
        x2 = self.b_2(x)
        x3 = self.b_3(x)


        # x3 = self.res_18(x)
        # x4 = self.res_50(x)
        # x5 = self.dense(x)

        # x3 = self.b_4(x)
        # x4 = self.b_5(x)

        x_18 = self.res_18(x)
        x_50 = self.res_50(x)
        x_d  = self.dense(x)

        # x_s  = self.seg(x)

        # x = (torch.softmax(x0, dim=1) + torch.softmax(x1, dim=1) + torch.softmax(x2, dim=1)) / 3.0

        # x = (torch.softmax(x_18, dim=1) + torch.softmax(x_50, dim=1)) / 3.0

        # x =  ((x0 + x1 + x2) / 3.0) + x3 + x4 

        # x = (x0 + x1 + x2) / 3.0 + (x_18 + x_50 + x_d) / 3.0
        # x = (x0 + x1 + x2 + (x0 + x1) / 2.0 + (x0 + x2) / 2.0 + (x1 + x2) / 2.0 + (x0 + x1 + x2) / 3.0) 
        # x = (((x2 + x_18) / 2.0) + ((x1 + x_d) / 2.0) + ((x0 + x_50) / 2.0)) / 3.0


        # x  = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15 + c16 + c17 + c18 + c19 + c20 
        # x  = x + c21 + c22 + c23 + c24 + c25 + c26 + c27 + c28 + c29 + c30 + c31 + c32 + c33 + c34 + c35 + c36 + c37 + c38 + c39 + c40 + c41 + c42 + c43 + c44 + c45 + c46 + c47 + c48 + c49 + c50 + c51 + c52 + c53 + c54 + c55 + c56 + c57 + c58 

        # x = ((x0 + x1 + x2) / 3.0) + x_d
        # x = ((x_d + x_50 + x_18) / 3.0) + ((x0 + x1 + x2) / 3.0)
        # x = ((x0 + x1 + x2) / 3.0) + x_50

        # x = ((x0 + x1 + x2) / 3.0)

        # y = (x + x_18) / 2.0
        # z = (x + x_50) / 2.0
        # w = (x + x_d)  / 2.0

        # x = ((x0 + x1 + x2) / 3.0)
        # y = (() / 3.0)
        # z = x_s 

        # return  + 2.0 * torch.softmax((x_18 + x_50 + x_d) / 3.0, dim=1)

        return torch.softmax((x0 + x1 + x2 + x3) / 4.0, dim=1) + x_18 + x_50 + x_d

        # return x_18

        # return x_s

        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b5
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
import random


class Mobile_netV2_0(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_0, self).__init__()

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)

        # model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool = model.avgpool

        for param in self.features[0:9].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=67, bias=True))


    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x # torch.softmax(x, dim=1)

class Mobile_netV2_1(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_1, self).__init__()

        model = efficientnet_b1(weights=EfficientNet_B1_Weights)
        # model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)

        # model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool = model.avgpool

        for param in self.features[0:9].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=67, bias=True))

    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x # torch.softmax(x, dim=1)


class Mobile_netV2_2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_2, self).__init__()

        model = efficientnet_b2(weights=EfficientNet_B2_Weights)
        # model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights)

        # model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool = model.avgpool

        for param in self.features[0:9].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1408, out_features=67, bias=True))


    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x # torch.softmax(x, dim=1)

class Mobile_netV2_3(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_3, self).__init__()

        model = efficientnet_b3(weights=EfficientNet_B3_Weights)
        # model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights)

        # model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool = model.avgpool

        for param in self.features[0:9].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1536, out_features=67, bias=True))


    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x # torch.softmax(x, dim=1)

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

class Mobile_netV2_res_18(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_res_18, self).__init__()


        model = models.__dict__['resnet18'](num_classes=365)
        # checkpoint = torch.load('/content/resnet18_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # model.load_state_dict(state_dict)

        self.model = model

        self.model.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=512, out_features=67, bias=True))
        # self.model.conv1.stride = (1, 1)

        self.avgpool = self.model.avgpool

        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        return torch.softmax(x, dim=1)


class Mobile_netV2_res_50(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_res_50, self).__init__()


        model = models.__dict__['resnet50'](num_classes=365)
        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # model.load_state_dict(state_dict)

        self.model = model

        self.model.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2048, out_features=67, bias=True))
        # self.model.conv1.stride = (1, 1)

        self.avgpool = self.model.avgpool

        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        return torch.softmax(x, dim=1)



class Mobile_netV2_dense(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_dense, self).__init__()

        model = models.__dict__['densenet161'](num_classes=365)
        # checkpoint = torch.load('/content/densenet161_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        # state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
        # state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}

        # model.load_state_dict(state_dict)

        self.model = model

        self.model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2208, out_features=67, bias=True))
        # self.model.features[0].stride = (1, 1)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        return torch.softmax(x, dim=1)




from mit_semseg.models import ModelBuilder

class Mobile_netV2_seg(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_seg, self).__init__()

        model =  ModelBuilder.build_encoder(arch='resnet50', fc_dim=2048, weights='/content/encoder_epoch_30.pth')

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2048, out_features=67, bias=True))


    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)[0]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return torch.softmax(x, dim=1)










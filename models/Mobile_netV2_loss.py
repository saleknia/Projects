import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights
import torchvision
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import ttach as tta

class Mobile_netV2_loss(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_loss, self).__init__()
        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.b_0 = Mobile_netV2_0()
        loaded_data_b_0 = torch.load('/content/drive/MyDrive/checkpoint_B0_81_23/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_0 = loaded_data_b_0['net']

        a = pretrained_b_0.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_0.pop(key)

        self.b_0.load_state_dict(pretrained_b_0)
        self.b_0 = self.b_0.eval()

        self.b_1 = Mobile_netV2_1()
        loaded_data_b_1 = torch.load('/content/drive/MyDrive/checkpoint_B1_82_80/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_1 = loaded_data_b_1['net']

        a = pretrained_b_1.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_1.pop(key)

        self.b_1.load_state_dict(pretrained_b_1)
        self.b_1 = self.b_1.eval()
        
        self.b_2 = Mobile_netV2_2()
        loaded_data_b_2 = torch.load('/content/drive/MyDrive/checkpoint_B2_84_07/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_2 = loaded_data_b_2['net']

        a = pretrained_b_2.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_2.pop(key)

        self.b_2.load_state_dict(pretrained_b_2)
        self.b_2 = self.b_2.eval()


        self.b_4 = Mobile_netV2_0()
        loaded_data_b_4 = torch.load('/content/drive/MyDrive/checkpoint_B0_82_87/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_4 = loaded_data_b_4['net']

        a = pretrained_b_4.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_4.pop(key)

        self.b_4.load_state_dict(pretrained_b_4)
        self.b_4 = self.b_4.eval()

        self.b_5 = Mobile_netV2_2()
        loaded_data_b_5 = torch.load('/content/drive/MyDrive/checkpoint_B2_84_52/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_5 = loaded_data_b_5['net']

        a = pretrained_b_5.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_5.pop(key)

        self.b_5.load_state_dict(pretrained_b_5)
        self.b_5 = self.b_5.eval()


        self.res_18 = Mobile_netV2_res_18()
        loaded_data_res_18 = torch.load('/content/drive/MyDrive/checkpoint_res_18/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_res_18 = loaded_data_res_18['net']

        self.res_18.load_state_dict(pretrained_res_18)
        self.res_18 = self.res_18.eval()


        self.res_50 = Mobile_netV2_res_50()
        loaded_data_res_50 = torch.load('/content/drive/MyDrive/checkpoint_res_50/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_res_50 = loaded_data_res_50['net']

        self.res_50.load_state_dict(pretrained_res_50)
        self.res_50 = self.res_50.eval()


        self.dense = Mobile_netV2_dense()
        loaded_data_dense = torch.load('/content/drive/MyDrive/checkpoint_dense/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_dense = loaded_data_dense['net']

        self.dense.load_state_dict(pretrained_dense)
        self.dense = self.dense.eval()

        # self.b_0 = tta.ClassificationTTAWrapper(self.b_0, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')
        # self.b_1 = tta.ClassificationTTAWrapper(self.b_1, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')
        # self.b_2 = tta.ClassificationTTAWrapper(self.b_2, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')

        # self.res_18 = tta.ClassificationTTAWrapper(self.res_18, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')
        # self.res_50 = tta.ClassificationTTAWrapper(self.res_50, tta.aliases.ten_crop_transform(224, 224), merge_mode='mean')


    def forward(self, x):
        b, c, w, h = x.shape

        x0 = self.b_0(x)
        x1 = self.b_1(x) 
        x2 = self.b_2(x)

        x3 = self.res_18(x)
        x4 = self.res_50(x)
        x5 = self.dense(x)

        # x3 = self.b_4(x)
        # x4 = self.b_5(x)

        # x_18 = self.res_18(x)
        # x_50 = self.res_50(x)
        # x_d  = self.dense(x)
        # x = (torch.softmax(x0, dim=1) + torch.softmax(x1, dim=1) + torch.softmax(x2, dim=1)) / 3.0

        # x = (torch.softmax(x_18, dim=1) + torch.softmax(x_50, dim=1)) / 3.0

        # x =  ((x0 + x1 + x2) / 3.0) + x3 + x4 

        # x = (x0 + x1 + x2) / 3.0 + (x_18 + x_50 + x_d) / 3.0
        # x = (x0 + x1 + x2 + (x0 + x1) / 2.0 + (x0 + x2) / 2.0 + (x1 + x2) / 2.0 + (x0 + x1 + x2) / 3.0) 
        # x = (((x2 + x_18) / 2.0) + ((x1 + x_d) / 2.0) + ((x0 + x_50) / 2.0)) / 3.0

        c1 = torch.softmax(x0, dim=1)
        c2 = torch.softmax(x1, dim=1)
        c3 = torch.softmax(x2, dim=1)
        c4 = torch.softmax(x3, dim=1)
        c5 = torch.softmax(x4, dim=1)
        c6 = torch.softmax(x5, dim=1)

        c4  = torch.softmax((x0 + x1) / 2.0, dim=1)
        c5  = torch.softmax((x0 + x2) / 2.0, dim=1)
        c6  = torch.softmax((x0 + x3) / 2.0, dim=1)
        c7  = torch.softmax((x0 + x4) / 2.0, dim=1)
        c8  = torch.softmax((x0 + x5) / 2.0, dim=1)

        c9  = torch.softmax((x1 + x2) / 2.0, dim=1)
        c10 = torch.softmax((x1 + x3) / 2.0, dim=1)
        c11 = torch.softmax((x1 + x4) / 2.0, dim=1)
        c12 = torch.softmax((x1 + x5) / 2.0, dim=1)

        c13 = torch.softmax((x2 + x3) / 2.0, dim=1)
        c14 = torch.softmax((x2 + x4) / 2.0, dim=1)
        c15 = torch.softmax((x2 + x5) / 2.0, dim=1)

        c16 = torch.softmax((x3 + x4) / 2.0, dim=1)
        c17 = torch.softmax((x3 + x5) / 2.0, dim=1)

        c18 = torch.softmax((x4 + x5) / 2.0, dim=1)

        c19 = torch.softmax((x0 + x1 + x2) / 3.0, dim=1)
        c20 = torch.softmax((x0 + x1 + x3) / 3.0, dim=1)
        c21 = torch.softmax((x0 + x1 + x4) / 3.0, dim=1)
        c22 = torch.softmax((x0 + x2 + x3) / 3.0, dim=1)
        c23 = torch.softmax((x0 + x2 + x4) / 3.0, dim=1)
        c24 = torch.softmax((x0 + x2 + x5) / 3.0, dim=1)
        c25 = torch.softmax((x0 + x3 + x4) / 3.0, dim=1)
        c26 = torch.softmax((x0 + x3 + x5) / 3.0, dim=1)
        c27 = torch.softmax((x0 + x4 + x5) / 3.0, dim=1)

        c28 = torch.softmax((x1 + x2 + x3) / 3.0, dim=1)
        c29 = torch.softmax((x1 + x2 + x4) / 3.0, dim=1)
        c30 = torch.softmax((x1 + x2 + x5) / 3.0, dim=1)
        c31 = torch.softmax((x1 + x3 + x4) / 3.0, dim=1)
        c32 = torch.softmax((x1 + x3 + x5) / 3.0, dim=1)
        c33 = torch.softmax((x1 + x4 + x5) / 3.0, dim=1)

        c34 = torch.softmax((x2 + x3 + x4) / 3.0, dim=1)
        c35 = torch.softmax((x2 + x3 + x5) / 3.0, dim=1)
        c36 = torch.softmax((x2 + x4 + x5) / 3.0, dim=1)

        c37 = torch.softmax((x3 + x4 + x5) / 3.0, dim=1)

        c38 = torch.softmax((x0 + x1 + x2 + x3) / 4.0, dim=1)
        c39 = torch.softmax((x0 + x1 + x2 + x4) / 4.0, dim=1)
        c40 = torch.softmax((x0 + x1 + x2 + x5) / 4.0, dim=1)
        c41 = torch.softmax((x0 + x1 + x3 + x4) / 4.0, dim=1)
        c42 = torch.softmax((x0 + x1 + x3 + x5) / 4.0, dim=1)
        c43 = torch.softmax((x0 + x1 + x4 + x5) / 4.0, dim=1)
        c44 = torch.softmax((x0 + x2 + x3 + x4) / 4.0, dim=1)
        c45 = torch.softmax((x0 + x2 + x3 + x5) / 4.0, dim=1)
        c46 = torch.softmax((x0 + x2 + x4 + x5) / 4.0, dim=1)
        c47 = torch.softmax((x0 + x3 + x4 + x5) / 4.0, dim=1)

        c48 = torch.softmax((x1 + x2 + x3 + x4) / 4.0, dim=1)
        c49 = torch.softmax((x1 + x2 + x3 + x5) / 4.0, dim=1)
        c50 = torch.softmax((x1 + x2 + x4 + x5) / 4.0, dim=1)
        c51 = torch.softmax((x1 + x3 + x4 + x5) / 4.0, dim=1)

        c52 = torch.softmax((x2 + x3 + x4 + x5) / 4.0, dim=1)

        c52 = torch.softmax((x0 + x1 + x2 + x3 + x4) / 5.0, dim=1)
        c53 = torch.softmax((x0 + x1 + x2 + x3 + x5) / 5.0, dim=1)
        c54 = torch.softmax((x0 + x1 + x2 + x4 + x5) / 5.0, dim=1)
        c55 = torch.softmax((x0 + x1 + x3 + x4 + x5) / 5.0, dim=1)
        c56 = torch.softmax((x0 + x2 + x3 + x4 + x5) / 5.0, dim=1)

        c57 = torch.softmax((x1 + x2 + x3 + x4 + x5) / 5.0, dim=1)

        c58 = torch.softmax((x0 + x1 + x2 + x3 + x4 + x5) / 6.0, dim=1)


        x  = (c4 + c5 + c6 + c7) / 4.0

        # x = c7

        return x

        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b5
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
import random


class Mobile_netV2_0(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_0, self).__init__()

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)

        model.features[0][0].stride = (1, 1)

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

        return x

class Mobile_netV2_1(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_1, self).__init__()

        model = efficientnet_b1(weights=EfficientNet_B1_Weights)
        # model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)

        model.features[0][0].stride = (1, 1)

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

        return x

class Mobile_netV2_2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_2, self).__init__()

        model = efficientnet_b2(weights=EfficientNet_B2_Weights)
        # model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights)

        model.features[0][0].stride = (1, 1)

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

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b5, efficientnet_v2_s, EfficientNet_V2_S_Weights
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


        teacher = models.__dict__['resnet18'](num_classes=365)
        # checkpoint = torch.load('/content/resnet18_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # teacher.load_state_dict(state_dict)

        self.teacher = teacher

        for param in self.teacher.parameters():
            param.requires_grad = False

        for param in self.teacher.layer4[-1].parameters():
            param.requires_grad = True

        self.teacher.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=512, out_features=67, bias=True))
        self.teacher.conv1.stride = (1, 1)

        self.avgpool = self.teacher.avgpool

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape


        x = self.teacher(x0)

        return x

class Mobile_netV2_res_50(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_res_50, self).__init__()


        teacher = models.__dict__['resnet50'](num_classes=365)
        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # teacher.load_state_dict(state_dict)

        self.teacher = teacher

        for param in self.teacher.parameters():
            param.requires_grad = False

        for param in self.teacher.layer4[-1].parameters():
            param.requires_grad = True

        self.teacher.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2048, out_features=67, bias=True))
        self.teacher.conv1.stride = (1, 1)

        self.avgpool = self.teacher.avgpool

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.teacher(x0)

        return x


class Mobile_netV2_dense(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_dense, self).__init__()

        teacher = models.__dict__['densenet161'](num_classes=365)
        # checkpoint = torch.load('/content/densenet161_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        # state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
        # state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}

        # teacher.load_state_dict(state_dict)

        self.teacher = teacher

        self.teacher.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2208, out_features=67, bias=True))
        self.teacher.features[0].stride = (1, 1)

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.teacher(x0)

        return x






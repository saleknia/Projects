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


        self.b_3 = Mobile_netV2_3()
        loaded_data_b_3 = torch.load('/content/drive/MyDrive/checkpoint_B3_84_07/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        pretrained_b_3 = loaded_data_b_3['net']

        a = pretrained_b_3.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_3.pop(key)

        self.b_3.load_state_dict(pretrained_b_3)
        self.b_3 = self.b_3.eval()


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
        x3 = self.b_3(x)

        x_18 = self.res_18(x)
        x_50 = self.res_50(x)

        # x = (torch.softmax(x0, dim=1) + torch.softmax(x1, dim=1) + torch.softmax(x2, dim=1)) / 3.0

        # x = (torch.softmax(x_18, dim=1) + torch.softmax(x_50, dim=1)) / 3.0

        x = ((x0 + x1 + x2) / 3.0) + x_18 + x_50

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
        checkpoint = torch.load('/content/resnet18_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        teacher.load_state_dict(state_dict)

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
        checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        teacher.load_state_dict(state_dict)

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







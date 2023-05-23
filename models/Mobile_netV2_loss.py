import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights
import torchvision
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class Mobile_netV2_loss(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_loss, self).__init__()
        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.b_0 = Mobile_netV2_0()
        loaded_data_b_0 = torch.load('/content/drive/MyDrive/checkpoint_VS_94_75/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_0 = loaded_data_b_0['net']

        a = pretrained_b_0.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_0.pop(key)

        self.b_0.load_state_dict(pretrained_b_0)
        self.b_0 = self.b_0.eval()


        self.b_1 = Mobile_netV2_1()
        loaded_data_b_1 = torch.load('/content/drive/MyDrive/checkpoint_VM_95_54/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_1 = loaded_data_b_1['net']

        a = pretrained_b_1.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_1.pop(key)

        self.b_1.load_state_dict(pretrained_b_1)
        self.b_1 = self.b_1.eval()

        self.b_2 = Mobile_netV2_2()
        loaded_data_b_2 = torch.load('/content/drive/MyDrive/checkpoint_VL_96_97/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_2 = loaded_data_b_2['net']

        a = pretrained_b_2.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_2.pop(key)

        self.b_2.load_state_dict(pretrained_b_2)
        self.b_2 = self.b_2.eval()


    def forward(self, x):
        b, c, w, h = x.shape

        x0 = self.b_0(x)
        x1 = self.b_1(x) 
        x2 = self.b_2(x)


        x = (x0 + x1 + x2*2) / 4.0
        

        return torch.softmax(x, dim=1)

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

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)

        model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool = model.avgpool

        for param in self.features[0:9].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=40, bias=True))


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

        # model = efficientnet_b1(weights=EfficientNet_B1_Weights)
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)

        model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool = model.avgpool

        for param in self.features[0:9].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=40, bias=True))

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

        # model = efficientnet_b2(weights=EfficientNet_B2_Weights)
        model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights)

        model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool = model.avgpool

        for param in self.features[0:9].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=40, bias=True))


    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x



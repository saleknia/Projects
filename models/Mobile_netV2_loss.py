import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights
import torchvision

class Mobile_netV2_loss(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_loss, self).__init__()
        model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.b_0 = Mobile_netV2_0()
        loaded_data_b_0 = torch.load('/content/drive/MyDrive/checkpoint_B0_83_92/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_0 = loaded_data_b_0['net']
        self.b_0.load_state_dict(pretrained_b_0)

        self.b_1 = Mobile_netV2_1()
        loaded_data_b_1 = torch.load('/content/drive/MyDrive/checkpoint_B1_84_51/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_1 = loaded_data_b_1['net']
        self.b_1.load_state_dict(pretrained_b_1)        
        
        self.b_2 = Mobile_netV2_2()
        loaded_data_b_2 = torch.load('/content/drive/MyDrive/checkpoint_B2_86_01/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_2 = loaded_data_b_2['net']
        self.b_2.load_state_dict(pretrained_b_2)

        self.b_3 = Mobile_netV2_3()
        loaded_data_b_3 = torch.load('/content/drive/MyDrive/checkpoint_B3_86_82/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_3 = loaded_data_b_3['net']
        self.b_3.load_state_dict(pretrained_b_3)

        self.combine = nn.Sequential(
            nn.Conv2d(1376, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True)
        )

        for param in self.b_0.parameters():
            param.requires_grad = False

        for param in self.b_1.parameters():
            param.requires_grad = False

        for param in self.b_2.parameters():
            param.requires_grad = False

        for param in self.b_3.parameters():
            param.requires_grad = False

        self.avgpool = model.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1536, out_features=40, bias=True),
        )

    def forward(self, x):
        b, c, w, h = x.shape

        x0 = self.b_0(x)
        x1 = self.b_1(x)
        x2 = self.b_2(x)
        x3 = self.b_3(x)

        # x = 1.0 * x0 + 1.45 * x1 + 1.67 * x2 + 2.0 * x3
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.combine(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        
        if self.training:
            return x
        else:
            return torch.softmax(x, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b5
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
import random


class Mobile_netV2_3(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_3, self).__init__()

        model = efficientnet_b3(weights=EfficientNet_B3_Weights)
        self.features = model.features
        self.avgpool = model.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1536, out_features=40, bias=True),
        )


    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        return x
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)

class Mobile_netV2_2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_2, self).__init__()

        model = efficientnet_b2(weights=EfficientNet_B2_Weights)

        self.features = model.features
        self.avgpool = model.avgpool


        self.classifier = nn.Sequential(
            nn.Linear(in_features=1408, out_features=40, bias=True),
        )


    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        return x
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)


class Mobile_netV2_1(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_1, self).__init__()

        model = efficientnet_b1(weights=EfficientNet_B1_Weights)

        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=40, bias=True),
        )

    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        return x
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)

class Mobile_netV2_0(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_0, self).__init__()

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.features = model.features
        self.avgpool = model.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=40, bias=True),
        )

    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        return x
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)





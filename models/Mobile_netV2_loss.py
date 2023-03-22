import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights
import torchvision

class Mobile_netV2_loss(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_loss, self).__init__()
        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.b_0 = Mobile_netV2_0()
        loaded_data_b_0 = torch.load('/content/drive/MyDrive/checkpoint_B0_90_00/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_0 = loaded_data_b_0['net']

        a = pretrained_b_0.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_0.pop(key)

        self.b_0.load_state_dict(pretrained_b_0)
        self.b_0 = self.b_0.eval()


        self.b_1 = Mobile_netV2_1()
        loaded_data_b_1 = torch.load('/content/drive/MyDrive/checkpoint_B1_91_66/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_1 = loaded_data_b_1['net']

        a = pretrained_b_1.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_1.pop(key)

        self.b_1.load_state_dict(pretrained_b_1)
        self.b_1 = self.b_1.eval()

        self.b_2 = Mobile_netV2_2()
        loaded_data_b_2 = torch.load('/content/drive/MyDrive/checkpoint_B2_92_21/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b_2 = loaded_data_b_2['net']

        a = pretrained_b_2.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_b_2.pop(key)

        self.b_2.load_state_dict(pretrained_b_2)
        self.b_2 = self.b_2.eval()

        # self.b_3 = Mobile_netV2_3()
        # loaded_data_b_3 = torch.load('/content/drive/MyDrive/checkpoint_B3_89_50/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        # pretrained_b_3 = loaded_data_b_3['net']

        # a = pretrained_b_3.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_b_3.pop(key)

        # self.b_3.load_state_dict(pretrained_b_3)
        # self.b_3 = self.b_3.eval()

        # for param in self.b_0.parameters():
        #     param.requires_grad = False

        # for param in self.b_1.parameters():
        #     param.requires_grad = False

        # for param in self.b_2.parameters():
        #     param.requires_grad = False

        # for param in self.b_3.parameters():
        #     param.requires_grad = False

        # net = sum(p.numel() for p in self.parameters())
        # self.w1  = sum(p.numel() for p in self.b_1.parameters()) / net
        # self.w2  = sum(p.numel() for p in self.b_2.parameters()) / net
        # self.w3  = sum(p.numel() for p in self.b_3.parameters()) / net

    def forward(self, x):
        b, c, w, h = x.shape

        x0 = self.b_0(x)
        x1 = self.b_1(x) 
        x2 = self.b_2(x)
        # x3 = self.b_3(x)

        # x = 1.0 * x0 + 1.45 * x1 + 1.67 * x2 + 2.0 * x3
        # x = 1.0 * x0 + 1.47 * x1 + 1.67 * x2 + 2.0 * x3
        # x = self.w1 * x1 + self.w2 * x2 + self.w3 * x3
        # x = 1.0 * x1 + 1.4 * x2 + 2.0 * x3
        # x = x0 + x1 + x2 + x3

        # x = (x0 + x1 + x2 + x3) / 4.0
        x = (x0 + x1 + x2) / 3.0
        

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


class Mobile_netV2_3(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_3, self).__init__()

        # self.teacher = Mobile_netV2_teacher()
        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_teacher/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # self.teacher.load_state_dict(pretrained_teacher)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model.features[0][0].stride = (1, 1)
        # self.features = model.features
        # self.avgpool = model.avgpool

        # self.segmentation = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights)
        # self.segmentation = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights)

        # for param in self.segmentation.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        model = efficientnet_b3(weights=EfficientNet_B3_Weights)

        # model.features[0][0].stride = (1, 1)
        # model.features[0][0].in_channels = 4

        self.features = model.features
        self.features[0][0].stride = (1, 1)
        self.avgpool = model.avgpool

        # for param in self.features[0:8].parameters():
        #     param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1536, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=512 , out_features=256, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=256 , out_features=40, bias=True),
        # )

    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
        # return torch.softmax(x, dim=1)
        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)

class Mobile_netV2_2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_2, self).__init__()

        # self.teacher = Mobile_netV2_teacher()
        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_teacher/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # self.teacher.load_state_dict(pretrained_teacher)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model.features[0][0].stride = (1, 1)
        # self.features = model.features
        # self.avgpool = model.avgpool

        # self.segmentation = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights)
        # self.segmentation = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights)

        # for param in self.segmentation.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        model = efficientnet_b2(weights=EfficientNet_B2_Weights)

        # model.features[0][0].stride = (1, 1)
        # model.features[0][0].in_channels = 4

        self.features = model.features
        self.features[0][0].stride = (1, 1)
        self.avgpool = model.avgpool

        # for param in self.features[0:8].parameters():
        #     param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1408, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=512 , out_features=256, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=256 , out_features=40, bias=True),
        # )

    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
        # return torch.softmax(x, dim=1)
        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)


class Mobile_netV2_1(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_1, self).__init__()

        # self.teacher = Mobile_netV2_teacher()
        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_teacher/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # self.teacher.load_state_dict(pretrained_teacher)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model.features[0][0].stride = (1, 1)
        # self.features = model.features
        # self.avgpool = model.avgpool

        # self.segmentation = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights)
        # self.segmentation = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights)

        # for param in self.segmentation.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        model = efficientnet_b1(weights=EfficientNet_B1_Weights)

        # model.features[0][0].stride = (1, 1)
        # model.features[0][0].in_channels = 4

        self.features = model.features
        self.features[0][0].stride = (1, 1)
        self.avgpool = model.avgpool

        # for param in self.features[0:8].parameters():
        #     param.requires_grad = False

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=1280, out_features=40, bias=True),
        # )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )


        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=512 , out_features=256, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=256 , out_features=40, bias=True),
        # )

    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
        # return torch.softmax(x, dim=1)
        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)

class Mobile_netV2_0(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_0, self).__init__()

        # self.teacher = Mobile_netV2_teacher()
        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_teacher/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # self.teacher.load_state_dict(pretrained_teacher)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model.features[0][0].stride = (1, 1)
        # self.features = model.features
        # self.avgpool = model.avgpool

        # self.segmentation = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights)
        # self.segmentation = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights)

        # for param in self.segmentation.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        # model.features[0][0].stride = (1, 1)
        # model.features[0][0].in_channels = 4

        self.features = model.features
        self.features[0][0].stride = (1, 1)
        self.avgpool = model.avgpool

        # for param in self.features[0:8].parameters():
        #     param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=512 , out_features=256, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=256 , out_features=40, bias=True),
        # )

    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
        # return torch.softmax(x, dim=1)
        # if self.training:
        #     return x
        # else:
        #     return torch.softmax(x, dim=1)




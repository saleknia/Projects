import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights, EfficientNet_B6_Weights, efficientnet_b6
import torchvision
import random


class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2, self).__init__()

        self.teacher = Mobile_netV2_teacher()
        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_teacher/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_teacher = loaded_data_teacher['net']
        self.teacher.load_state_dict(pretrained_teacher)

        for param in self.encoder_teacher.parameters():
            param.requires_grad = False

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        model.features[0][0].stride = (1, 1)
        self.features = model.features
        self.avgpool = model.avgpool


        self.drop_1  = nn.Dropout(p=0.5, inplace=True)
        self.dense_1 = nn.Linear(in_features=1280, out_features=512, bias=True)
        self.drop_2  = nn.Dropout(p=0.5, inplace=True)
        self.dense_2 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.drop_3  = nn.Dropout(p=0.5, inplace=True)
        self.dense_3 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.drop_4  = nn.Dropout(p=0.5, inplace=True)
        self.dense_4 = nn.Linear(in_features=128, out_features=num_classes, bias=True)

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=256, out_features=128, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=128, out_features=num_classes, bias=True),
        # )
        
    def forward(self, x):
        b, c, w, h = x.shape

        x4_teacher = self.teacher(x)

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        x1 = self.drop_1(x)
        x1 = self.dense_1(x1)

        x2 = self.drop_2(x1)
        x2 = self.dense_2(x2)        
        
        x3 = self.drop_3(x2)
        x3 = self.dense_3(x3)

        x4 = self.drop_4(x3)
        x4 = self.dense_4(x4)

        if self.training:
            return x4, x4_teacher
        else:
            return x4

class Mobile_netV2_teacher(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(Mobile_netV2_teacher, self).__init__()

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        model.features[0][0].stride = (1, 1)
        self.features = model.features
        self.avgpool = model.avgpool


        self.drop_1  = nn.Dropout(p=0.5, inplace=True)
        self.dense_1 = nn.Linear(in_features=1280, out_features=512, bias=True)
        self.drop_2  = nn.Dropout(p=0.5, inplace=True)
        self.dense_2 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.drop_3  = nn.Dropout(p=0.5, inplace=True)
        self.dense_3 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.drop_4  = nn.Dropout(p=0.5, inplace=True)
        self.dense_4 = nn.Linear(in_features=128, out_features=num_classes, bias=True)

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=256, out_features=128, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=128, out_features=num_classes, bias=True),
        # )
        
    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        x1 = self.drop_1(x)
        x1 = self.dense_1(x1)

        x2 = self.drop_2(x1)
        x2 = self.dense_2(x2)        
        
        x3 = self.drop_3(x2)
        x3 = self.dense_3(x3)

        x4 = self.drop_4(x3)
        x4 = self.dense_4(x4)

        return x4



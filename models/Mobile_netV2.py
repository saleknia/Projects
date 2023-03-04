import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b5
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
import random

class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2, self).__init__()

        # self.teacher = Mobile_netV2_teacher()
        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_B3_86_82/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # self.teacher.load_state_dict(pretrained_teacher)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        model = efficientnet_b3(weights=EfficientNet_B3_Weights)

        # model.features[0][0].stride = (1, 1)
        model.features[6][0].block[1][0].stride = (1, 1)

        self.features = model.features
        self.avgpool = model.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1536, out_features=40, bias=True),
        )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512 , out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256 , out_features=40, bias=True),
        # )

    def forward(self, x0):
        b, c, w, h = x0.shape

        # a = self.teacher(x0)

        x = self.features(x0)

        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.training:
            return x
        else:
            return torch.softmax(x, dim=1)


class Mobile_netV2_teacher(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_teacher, self).__init__()

        model = efficientnet_b3(weights=EfficientNet_B3_Weights)

        # model = efficientnet_b3(weights=EfficientNet_B3_Weights)

        # model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool = model.avgpool

        # for param in self.features[0:8].parameters():
        #     param.requires_grad = False


        self.classifier = nn.Sequential(
            nn.Linear(in_features=1536, out_features=40, bias=True),
        )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512 , out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256 , out_features=40, bias=True),
        # )

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.features(x0)

        x = self.avgpool(x) 
        
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        if self.training:
            return x 
        else:
            return torch.softmax(x, dim=1)




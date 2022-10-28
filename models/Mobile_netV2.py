import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights
import torchvision


class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2, self).__init__()

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )
        # model = resnet18(pretrained)
        # model = resnet50(pretrained)

        # take pretrained resnet, except AvgPool and FC
        # self.conv1 = model.conv1
        # self.bn1 = model.bn1
        # self.relu = model.relu
        # self.maxpool = model.maxpool
        # self.layer1 = model.layer1
        # self.layer2 = model.layer2
        # self.layer3 = model.layer3
        # self.layer4 = model.layer4
        # self.avgpool = model.avgpool
        # self.fc = nn.Linear(in_features=1280, out_features=40)
        # self.fc = nn.Linear(in_features=2048, out_features=40)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# class Mobile_netV2(nn.Module):
#     def __init__(self, num_classes=40, pretrained=True):
#         super(Mobile_netV2, self).__init__()

#         # take pretrained resnet, except AvgPool and FC
        
#         # model = resnet18(pretrained)
#         model = resnet50(pretrained)

#         self.conv1 = model.conv1
#         self.bn1 = model.bn1
#         self.relu = model.relu
#         self.maxpool = model.maxpool
#         self.layer1 = model.layer1
#         self.layer2 = model.layer2
#         self.layer3 = model.layer3
#         self.layer4 = model.layer4
#         self.avgpool = model.avgpool

#         self.avgpool = model.avgpool
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.4, inplace=True),
#             nn.Linear(in_features=2048, out_features=512, bias=True),
#             nn.Dropout(p=0.4, inplace=True),
#             nn.Linear(in_features=512, out_features=256, bias=True),
#             nn.Dropout(p=0.4, inplace=True),
#             nn.Linear(in_features=256, out_features=40, bias=True),
#         )


#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x



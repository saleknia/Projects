import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights
import torchvision

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2, self).__init__()

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # for param in model.parameters():
        #     param.requires_grad = False

        # self.features = model.features
        # self.SE = SEBlock(channel=1280)
        # self.avgpool = model.avgpool
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256, out_features=40, bias=True),
        # )

        model = resnet50(pretrained)
        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        for param in model.parameters():
            param.requires_grad = False

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )
        self.SE_2 = SEBlock(channel=512 )
        self.SE_3 = SEBlock(channel=1024)
        self.SE_4 = SEBlock(channel=2048)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.SE_2(x)
        x = self.layer3(x)
        x = self.SE_3(x)
        x = self.layer4(x)
        x = self.SE_4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.features(x)
        # x = self.SE(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
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



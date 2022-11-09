import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights, EfficientNet_B6_Weights, efficientnet_b6
import torchvision
import random

class SEBlock(nn.Module):
    def __init__(self, channel, r=8):
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

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model = efficientnet_b6(weights=EfficientNet_B6_Weights)

        self.features = model.features
        self.features[0][0].stride = (1, 1)
        self.avgpool = model.avgpool
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=2304, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256, out_features=40, bias=True),
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )
        self.aspp = ASPP(in_channels=1280)

        # model = resnet50(pretrained)
        # # # model = resnet18(pretrained)

        # # take pretrained resnet, except AvgPool and FC
        # self.conv1 = model.conv1
        # self.bn1 = model.bn1
        # self.relu = model.relu
        # self.maxpool = model.maxpool
        # self.layer1 = model.layer1
        # self.layer2 = model.layer2
        # self.layer3 = model.layer3
        # self.layer4 = model.layer4

        # self.avgpool = model.avgpool
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=2048, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256, out_features=40, bias=True),
        # )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # layer1 = self.layer1(x)
        # layer2 = self.layer2(layer1)
        # layer3 = self.layer3(layer2)
        # layer4 = self.layer4(layer3)
        # x = self.avgpool(layer4)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        x = self.features(x)
        x = self.aspp(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class ASPP(nn.Module):

    def __init__(self, in_channels):
        super(ASPP, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = in_channels // 5
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.aspp1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.aspp2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=(3,3), stride=(1,1), padding=(2,2), dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.aspp3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=(3,3), stride=(1,1), padding=(4,4), dilation=4, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.mid_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.aspp4 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=(3,3), stride=(1,1), padding=(8,8), dilation=8, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=self.mid_channels)
        self.relu4 = nn.ReLU(inplace=True)

        self.aspp5 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=self.mid_channels)
        self.relu5 = nn.ReLU(inplace=True)


    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.aspp2(x)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x3 = self.aspp3(x)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        x4 = self.aspp4(x)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)

        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)

        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)

        return x






















import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights
import torchvision

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
        
        self.features = model.features
        self.attention = SEBlock(channel=1280)
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )

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
        x = self.attention(x)
        if self.training:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        else:
            x1 = self.avgpool(x[:, :, 0:4, 0:4])
            x1 = x1.view(x1.size(0), -1)
            x1 = self.classifier(x1)

            x2 = self.avgpool(x[:, :, 4: , 0:4])
            x2 = x2.view(x2.size(0), -1)
            x2 = self.classifier(x2)

            x3 = self.avgpool(x[:, :, 0:4, 4: ])
            x3 = x3.view(x3.size(0), -1)
            x3 = self.classifier(x3)

            x4 = self.avgpool(x[:, :, 4: , 4: ])
            x4 = x4.view(x4.size(0), -1)
            x4 = self.classifier(x4)

            return (x1 + x2 + x3 + x4) / 4.0
























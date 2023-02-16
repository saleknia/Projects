import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights
import torchvision

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=3, padding=1, dilation=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class Mobile_netV2_loss(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_loss, self).__init__()
        model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.encoder_group_1 = Mobile_netV2(num_classes=3)
        loaded_data_group_1 = torch.load('/content/drive/MyDrive/checkpoint_group_1/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_group_1 = loaded_data_group_1['net']
        self.encoder_group_1.load_state_dict(pretrained_group_1)

        for param in self.encoder_group_1.parameters():
            param.requires_grad = False

        self.encoder_group_2 = Mobile_netV2(num_classes=4)
        loaded_data_group_2 = torch.load('/content/drive/MyDrive/checkpoint_group_2/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_group_2 = loaded_data_group_2['net']
        self.encoder_group_2.load_state_dict(pretrained_group_2)

        for param in self.encoder_group_2.parameters():
            param.requires_grad = False

        self.encoder_classifier = Mobile_netV2_classifier(num_classes=2)
        loaded_data_classifier = torch.load('/content/drive/MyDrive/checkpoint_classifier/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_classifier = loaded_data_classifier['net']
        self.encoder_classifier.load_state_dict(pretrained_classifier)

        for param in self.encoder_classifier.parameters():
            param.requires_grad = False

        self.avgpool = model.avgpool

        self.drop_1  = nn.Dropout(p=0.5, inplace=True)
        self.dense_1 = nn.Linear(in_features=1280, out_features=512, bias=True)
        self.drop_2  = nn.Dropout(p=0.5, inplace=True)
        self.dense_2 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.drop_3  = nn.Dropout(p=0.5, inplace=True)
        self.dense_3 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.drop_4  = nn.Dropout(p=0.5, inplace=True)
        self.dense_4 = nn.Linear(in_features=128, out_features=num_classes, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape

        x_group_1 = self.encoder_group_1(x)
        x_group_2 = self.encoder_group_1(x)

        alpha, beta = self.encoder_classifier(x)


        # x_fuse = torch.cat([alpha.expand_as(x_group_1)* x_group_1, beta.expand_as(x_group_2) * x_group_2], dim=1)

        x_fuse = (alpha.expand_as(x_group_1)* x_group_1) + (beta.expand_as(x_group_2) * x_group_2)

        x = self.avgpool(x_fuse)
        x = x.view(x.size(0), -1)

        x1 = self.drop_1(x)
        x1 = self.dense_1(x1)

        x2 = self.drop_2(x1)
        x2 = self.dense_2(x2)        
        
        x3 = self.drop_3(x2)
        x3 = self.dense_3(x3)

        x4 = self.drop_4(x3)
        x4 = self.dense_4(x4)

        return x4


class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Mobile_netV2, self).__init__()

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

        
    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        # x1 = self.drop_1(x)
        # x1 = self.dense_1(x1)

        # x2 = self.drop_2(x1)
        # x2 = self.dense_2(x2)        
        
        # x3 = self.drop_3(x2)
        # x3 = self.dense_3(x3)

        # x4 = self.drop_4(x3)
        # x4 = self.dense_4(x4)

        # x4 = torch.softmax(x4, dim=1)[:, 1:]

        return x

class Mobile_netV2_classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Mobile_netV2_classifier, self).__init__()

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

    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x1 = self.drop_1(x)
        x1 = self.dense_1(x1)

        x2 = self.drop_2(x1)
        x2 = self.dense_2(x2)        
        
        x3 = self.drop_3(x2)
        x3 = self.dense_3(x3)

        x4 = self.drop_4(x3)
        x4 = self.dense_4(x4)

        x4 = torch.softmax(x4, dim=1)

        return x4[:, 0], x4[:, 1]



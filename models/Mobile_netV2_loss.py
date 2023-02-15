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
        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.encoder_angry = Mobile_netV2()
        loaded_data_angry = torch.load('/content/drive/MyDrive/checkpoint_angry/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_angry = loaded_data_angry['net']
        self.encoder_angry.load_state_dict(pretrained_angry)

        for param in self.encoder_angry.parameters():
            param.requires_grad = False

        self.encoder_disgust = Mobile_netV2()
        loaded_data_disgust = torch.load('/content/drive/MyDrive/checkpoint_disgust/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_disgust = loaded_data_disgust['net']
        self.encoder_disgust.load_state_dict(pretrained_disgust)

        for param in self.encoder_disgust.parameters():
            param.requires_grad = False

        self.encoder_fear = Mobile_netV2()
        loaded_data_fear = torch.load('/content/drive/MyDrive/checkpoint_fear/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_fear = loaded_data_fear['net']
        self.encoder_fear.load_state_dict(pretrained_fear)

        for param in self.encoder_fear.parameters():
            param.requires_grad = False

        self.encoder_happy = Mobile_netV2()
        loaded_data_happy = torch.load('/content/drive/MyDrive/checkpoint_happy/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_happy = loaded_data_happy['net']
        self.encoder_happy.load_state_dict(pretrained_happy)

        for param in self.encoder_happy.parameters():
            param.requires_grad = False       

        self.encoder_neutral = Mobile_netV2()
        loaded_data_neutral = torch.load('/content/drive/MyDrive/checkpoint_neutral/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_neutral = loaded_data_neutral['net']
        self.encoder_neutral.load_state_dict(pretrained_neutral)

        for param in self.encoder_neutral.parameters():
            param.requires_grad = False

        self.encoder_sad = Mobile_netV2()
        loaded_data_sad = torch.load('/content/drive/MyDrive/checkpoint_sad/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_sad = loaded_data_sad['net']
        self.encoder_sad.load_state_dict(pretrained_sad)

        for param in self.encoder_sad.parameters():
            param.requires_grad = False

        self.encoder_surprise = Mobile_netV2()
        loaded_data_surprise = torch.load('/content/drive/MyDrive/checkpoint_surprise/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        pretrained_surprise = loaded_data_surprise['net']
        self.encoder_surprise.load_state_dict(pretrained_surprise)

        for param in self.encoder_surprise.parameters():
            param.requires_grad = False

        # self.reduce_angry    = ConvBatchNorm(in_channels=1280, out_channels=128, activation='ReLU', kernel_size=1, padding=0, dilation=1)
        # self.reduce_disgust  = ConvBatchNorm(in_channels=1280, out_channels=128, activation='ReLU', kernel_size=1, padding=0, dilation=1)
        # self.reduce_fear     = ConvBatchNorm(in_channels=1280, out_channels=128, activation='ReLU', kernel_size=1, padding=0, dilation=1)
        # self.reduce_happy    = ConvBatchNorm(in_channels=1280, out_channels=128, activation='ReLU', kernel_size=1, padding=0, dilation=1)
        # self.reduce_neutral  = ConvBatchNorm(in_channels=1280, out_channels=128, activation='ReLU', kernel_size=1, padding=0, dilation=1)
        # self.reduce_sad      = ConvBatchNorm(in_channels=1280, out_channels=128, activation='ReLU', kernel_size=1, padding=0, dilation=1)
        # self.reduce_surprise = ConvBatchNorm(in_channels=1280, out_channels=128, activation='ReLU', kernel_size=1, padding=0, dilation=1)

        # self.extend = ConvBatchNorm(in_channels=896, out_channels=1280, activation='ReLU', kernel_size=1, padding=0, dilation=1)

        # self.avgpool = model.avgpool

        # self.drop_1  = nn.Dropout(p=0.5, inplace=True)
        # self.dense_1 = nn.Linear(in_features=1280, out_features=512, bias=True)
        # self.drop_2  = nn.Dropout(p=0.5, inplace=True)
        # self.dense_2 = nn.Linear(in_features=512, out_features=256, bias=True)
        # self.drop_3  = nn.Dropout(p=0.5, inplace=True)
        # self.dense_3 = nn.Linear(in_features=256, out_features=128, bias=True)
        # self.drop_4  = nn.Dropout(p=0.5, inplace=True)
        # self.dense_4 = nn.Linear(in_features=128, out_features=num_classes, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape

        x_angry    = self.encoder_angry(x)
        x_disgust  = self.encoder_disgust(x)
        x_fear     = self.encoder_fear(x)
        x_happy    = self.encoder_happy(x)
        x_neutral  = self.encoder_neutral(x)
        x_sad      = self.encoder_sad(x)
        x_surprise = self.encoder_surprise(x)

        # x_angry    = self.reduce_angry(x_angry)
        # x_disgust  = self.reduce_disgust(x_disgust)
        # x_fear     = self.reduce_fear(x_fear)
        # x_happy    = self.reduce_happy(x_happy)
        # x_neutral  = self.reduce_neutral(x_neutral)
        # x_sad      = self.reduce_sad(x_sad)
        # x_surprise = self.reduce_surprise(x_surprise)

        # x_fuse = self.extend(torch.cat([x_angry, x_disgust, x_fear, x_happy, x_neutral, x_sad, x_surprise], dim=1))

        x_fuse = torch.cat([x_angry, x_disgust, x_fear, x_happy, x_neutral, x_sad, x_surprise], dim=1)


        # x = self.avgpool(x_fuse)
        # x = x.view(x.size(0), -1)

        # x = self.drop_1(x)
        # x = self.dense_1(x)

        # x = self.drop_2(x)
        # x = self.dense_2(x)        
        
        # x = self.drop_3(x)
        # x = self.dense_3(x)

        # x = self.drop_4(x)
        # x = self.dense_4(x)

        return x_surprise


class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Mobile_netV2, self).__init__()

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model.features[0][0].stride = (1, 1)
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

        # x4 = torch.softmax(x4, dim=1)[:, 1:]

        return x4





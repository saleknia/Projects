import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b5, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import random
from torch.nn import init
from .Mobile_netV2_loss import Mobile_netV2_loss

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import timm
from .wideresnet import *
from .wideresnet import recursion_change_bn
from .Mobile_netV2 import Mobile_netV2, mvit_teacher, convnext_small, mvit_small, convnext_tiny, mvit_tiny, convnextv2_tiny, convnext_teacher
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.models import ModelBuilder
from mit_semseg.models import ModelBuilder, SegmentationModule
import ttach as tta


# class SEUNet(nn.Module):
#     def __init__(self, num_classes=40, pretrained=True):
#         super(SEUNet, self).__init__()

#         ###############################################################################################
#         ###############################################################################################
#         model_0 = models.__dict__['resnet50'](num_classes=365)

#         checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
#         state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
#         model_0.load_state_dict(state_dict)

#         for param in model_0.parameters():
#             param.requires_grad = False

#         for param in model_0.layer4[-1].parameters():
#             param.requires_grad = True

#         ###############################################################################################
#         ###############################################################################################
#         model_1 = models.__dict__['resnet50'](num_classes=365)

#         # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
#         # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
#         # model_1.load_state_dict(state_dict)

#         for param in model_1.parameters():
#             param.requires_grad = False

#         for param in model_1.layer4.parameters():
#             param.requires_grad = True  

#         ###############################################################################################
#         ###############################################################################################
#         model_dense = models.__dict__['densenet161'](num_classes=365)

#         checkpoint = torch.load('/content/densenet161_places365.pth.tar', map_location='cpu')
#         state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
#         state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
#         state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}
#         model_dense.load_state_dict(state_dict)
#         model_dense.classifier = nn.Identity()
#         self.dense = model_dense
#         for param in self.dense.parameters():
#             param.requires_grad = False

#         model_res = models.__dict__['resnet50'](num_classes=365)

#         checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
#         state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
#         model_res.load_state_dict(state_dict)
#         model_res.fc = nn.Identity()
#         self.res = model_res
#         for param in self.res.parameters():
#             param.requires_grad = False

#         ###############################################################################################
#         ###############################################################################################

#         self.conv1   = model_0.conv1
#         self.bn1     = model_0.bn1
#         self.relu    = model_0.relu 
#         self.maxpool = model_0.maxpool

#         self.layer10 = model_0.layer1
#         self.layer20 = model_0.layer2
#         self.layer30 = model_0.layer3

#         self.layer40 = model_0.layer4
#         self.layer41 = model_1.layer4

#         self.avgpool_0 = model_0.avgpool
#         self.avgpool_1 = model_1.avgpool

#         self.fc_0 = nn.Sequential(
#             nn.Dropout(p=0.5, inplace=True),
#             nn.Linear(in_features=2048, out_features=67, bias=True))

#         self.fc_1 = nn.Sequential(
#             nn.Dropout(p=0.5, inplace=True),
#             nn.Linear(in_features=2048, out_features=67, bias=True))

#         # checkpoint = torch.load('/content/drive/MyDrive/checkpoint/a_best.pth', map_location='cpu')
#         # self.load_state_dict(checkpoint['net'])

#         # checkpoint = torch.load('/content/drive/MyDrive/checkpoint/Mobile_NetV2_MIT-67_best.pth', map_location='cpu')
#         # self.mobile.load_state_dict(checkpoint['net'])
        
#     def forward(self, x0):
#         b, c, w, h = x0.shape

#         # x_m = self.mobile(x0)

#         x_dense = self.dense(x0)
#         # x_res   = self.res(x0)

#         x = self.conv1(x0)
#         x = self.bn1(x)   
#         x = self.relu(x)  
#         x = self.maxpool(x)

#         x = self.layer10(x)
#         x = self.layer20(x)
#         x = self.layer30(x)

#         # x00 = self.layer40(x)
#         # x01 = self.avgpool_0(x00)
#         # x02 = x01.view(x01.size(0), -1)
#         # x03 = self.fc_0(x02)

#         x10 = self.layer41(x)
#         x11 = self.avgpool_1(x10)
#         x12 = x11.view(x11.size(0), -1)
#         x13 = self.fc_1(x12)

#         # x20 = self.layer42(x)
#         # x21 = self.avgpool_2(x20)
#         # x22 = x21.view(x21.size(0), -1)
#         # x23 = self.fc_2(x22)

#         # print(x_dense.shape)
#         # print(x11.shape)

#         # return x03 + x13

#         if self.training:
#             return x13, x12, x_dense
#         else:
#             return x13

# class SEUNet(nn.Module):
#     def __init__(self, num_classes=40, pretrained=True):
#         super(SEUNet, self).__init__()

#         model_dense = models.__dict__['densenet161'](num_classes=365)

#         checkpoint = torch.load('/content/densenet161_places365.pth.tar', map_location='cpu')
#         state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
#         state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
#         state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}
#         model_dense.load_state_dict(state_dict)

#         self.dense = model_dense

#         for param in self.dense.parameters():
#             param.requires_grad = False

#         for i, module in enumerate(self.dense.features.denseblock4):
#             if 20 <= i: 
#                 for param in self.dense.features.denseblock4[module].parameters():
#                     param.requires_grad = True

#         self.dense.classifier = nn.Sequential(
#             nn.Dropout(p=0.5, inplace=True),
#             nn.Linear(in_features=2208, out_features=num_classes, bias=True))

#         # checkpoint = torch.load('/content/drive/MyDrive/checkpoint/a_best.pth', map_location='cpu')
#         # self.load_state_dict(checkpoint['net'])

#         # checkpoint = torch.load('/content/drive/MyDrive/checkpoint/Mobile_NetV2_MIT-67_best.pth', map_location='cpu')
#         # self.mobile.load_state_dict(checkpoint['net'])
        
#     def forward(self, x0):
#         b, c, w, h = x0.shape

#         x_dense = self.dense(x0)
        
#         return x_dense


class SEUNet(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(SEUNet, self).__init__()

        self.convnext = convnext_small()
        self.mvit     = mvit_small()
        # self.teacher = convnext_teacher()

        self.dense_model = dense_model()

        checkpoint = torch.load('/content/drive/MyDrive/checkpoint_dense_ensemble/18_best.pth', map_location='cpu')
        pretrained_teacher = checkpoint['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.dense_model.load_state_dict(pretrained_teacher)

        self.res_model_2 = res_model()

        checkpoint = torch.load('/content/drive/MyDrive/checkpoint_dense_ensemble/2_res_best.pth', map_location='cpu')
        pretrained_teacher = checkpoint['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.res_model_2.load_state_dict(pretrained_teacher)

        self.res_model_0 = res_model()

        checkpoint = torch.load('/content/drive/MyDrive/checkpoint_dense_ensemble/0_res_best.pth', map_location='cpu')
        pretrained_teacher = checkpoint['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.res_model_0.load_state_dict(pretrained_teacher)

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x_res_0, feature_0   = self.res_model_0(x0)  
        x_res_2, feature_2   = self.res_model_2(x0)  

        # x_dense = self.dense_model(x0)

        x_trans = self.mvit(x0)    
        # x_next  = self.convnext(x0)

        
        distance = torch.abs(self.cos(feature_0, feature_2))[0]

        # output_1  = torch.softmax(x_res + x_next, dim=1)
        # output_2  = torch.softmax(x_next , dim=1)

        # x_dense = torch.softmax(x_dense, dim=1)
        x_res   = torch.softmax(x_res_2, dim=1)
        x_trans = torch.softmax(x_trans, dim=1)
        # x_next  = torch.softmax(x_next , dim=1)

        # output_1  = torch.softmax(x_res + x_dense + x_next , dim=1)
        # output_2  = torch.softmax(x_res + x_dense + x_trans, dim=1)

        return (x_res * (1.0)) #+ (x_trans)

class dense_model_distillation(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(dense_model_distillation, self).__init__()

        model_dense = models.__dict__['densenet161'](num_classes=365)

        checkpoint = torch.load('/content/densenet161_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
        state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}
        model_dense.load_state_dict(state_dict)

        self.dense = model_dense

        for param in self.dense.parameters():
            param.requires_grad = False

        for i, module in enumerate(self.dense.features.denseblock4):
            if 18 <= i: 
                for param in self.dense.features.denseblock4[module].parameters():
                    param.requires_grad = True

        self.dense.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2208, out_features=num_classes, bias=True))

        # checkpoint = torch.load('/content/drive/MyDrive/checkpoint/a_best.pth', map_location='cpu')
        # self.load_state_dict(checkpoint['net'])

        # checkpoint = torch.load('/content/drive/MyDrive/checkpoint_dense_ensemble/18_best.pth', map_location='cpu')
        # self.load_state_dict(checkpoint['net'])

        self.teacher = teacher()

    def forward(self, x0):
        b, c, w, h = x0.shape

        x_dense = self.dense(x0)
        x_t     = self.teacher(x0)
        
        if self.training:
            return x_dense, x_t
        else:
            return torch.softmax(x_dense,dim=1) # x_dense


class dense_model(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(dense_model, self).__init__()

        model_dense = models.__dict__['densenet161'](num_classes=365)

        checkpoint = torch.load('/content/densenet161_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
        state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}
        model_dense.load_state_dict(state_dict)

        self.dense = model_dense

        for param in self.dense.parameters():
            param.requires_grad = False

        for i, module in enumerate(self.dense.features.denseblock4):
            if 18 <= i: 
                for param in self.dense.features.denseblock4[module].parameters():
                    param.requires_grad = True

        self.dense.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2208, out_features=num_classes, bias=True))

        checkpoint = torch.load('/content/drive/MyDrive/checkpoint_dense_ensemble/18_best.pth', map_location='cpu')
        self.load_state_dict(checkpoint['net'])

        # for param in self.dense.parameters():
        #     param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape

        x_dense = self.dense(x0)
        
        return x_dense


class res_model(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(res_model, self).__init__()

        ###############################################################################################
        ###############################################################################################
        model = models.__dict__['resnet50'](num_classes=365)

        checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

        for param in model.parameters():
            param.requires_grad = False

        # for param in model.layer4[-1].parameters():
        #     param.requires_grad = True

        for param in model.layer4[-1].conv2.parameters():
            param.requires_grad = True

        for param in model.layer4[-1].conv3.parameters():
            param.requires_grad = True
        ###############################################################################################
        ###############################################################################################

        self.conv1   = model.conv1
        self.bn1     = model.bn1
        self.relu    = model.relu 
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = model.avgpool

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True), 
            nn.Linear(in_features=2048, out_features=67, bias=True))

        # checkpoint = torch.load('/content/drive/MyDrive/checkpoint_dense_ensemble/res_50.pth', map_location='cpu')
        # pretrained_teacher = checkpoint['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape

        # x0 = transform_test(x0)

        x = self.conv1(x0)
        x = self.bn1(x)   
        x = self.relu(x)  
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)

        return y, x 

from torchvision import transforms
transform_test = transforms.Compose([transforms.Resize((448, 448))])

class res_model_distill(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(res_model_distill, self).__init__()

        ###############################################################################################
        ###############################################################################################
        model = models.__dict__['resnet50'](num_classes=365)

        checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

        for param in model.parameters():
            param.requires_grad = False

        # for param in model.layer4[-1].conv2.parameters():
        #     param.requires_grad = True

        # for param in model.layer4[-1].conv3.parameters():
        #     param.requires_grad = True

        # for param in model.layer3.parameters():
        #     param.requires_grad = True

        for param in model.layer4[-1].parameters():
            param.requires_grad = True

        ###############################################################################################
        ###############################################################################################

        self.conv1   = model.conv1
        self.bn1     = model.bn1
        self.relu    = model.relu 
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = model.avgpool

        self.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2048, out_features=67, bias=True))

        # checkpoint = torch.load('/content/drive/MyDrive/checkpoint_dense_ensemble/res_50_distilled.pth', map_location='cpu')
        # pretrained_teacher = checkpoint['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

        # for param in self.parameters():
        #     param.requires_grad = False

        self.teacher = teacher()

    def forward(self, x0):
        b, c, w, h = x0.shape
        # x0 = transform_test(x0)
        x = self.conv1(x0)
        x = self.bn1(x)   
        x = self.relu(x)  
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)

        x_t = self.teacher(x0)

        if self.training:
            return y, x, x_t
        else:
            return y
        

class teacher(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(teacher, self).__init__()
        self.dense = dense_model()
        # self.res_model = res_model()

    def forward(self, x0):
        b, c, w, h = x0.shape

        # output = (self.dense(x0) + self.res50(x0)) / 2.0

        # output = self.res_model(x0)
        output = self.res_model(x0)

        # output = (self.dense(x0) + self.res_model(x0)) / 2.0
        # output = torch.softmax(output, dim=1)

        return output

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






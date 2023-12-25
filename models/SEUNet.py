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
from .Mobile_netV2 import Mobile_netV2, mvit_teacher, convnext_small, mvit_small, convnext_tiny, mvit_tiny, convnextv2_tiny, convnext_teacher, efficientnet_teacher
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.models import ModelBuilder
from mit_semseg.models import ModelBuilder, SegmentationModule
import ttach as tta


class SEUNet(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(SEUNet, self).__init__()

        self.mvit_tiny = mvit_tiny()
        self.next_tiny = convnextv2_tiny()
        self.res_model = res_model()
        self.cnn       = B2()

    def forward(self, x0):
        b, c, w, h = x0.shape

        x_res = self.res_model(x0)  
        x_t   = self.mvit_tiny(x0)   
        x_n   = self.next_tiny(x0) 
        x_c   = self.cnn(x0)  

        output_1  = torch.softmax(x_res + x_t, dim=1)
        output_2  = torch.softmax(x_res + x_n, dim=1)
        output_3  = torch.softmax(x_res + x_c, dim=1)

        out = torch.softmax(output_1 + output_2 + output_3, dim=1)

        return out

class B2(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(B2, self).__init__()

        model = efficientnet_b2(weights=EfficientNet_B2_Weights)

        model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool  = model.avgpool

        for param in self.features[0:4].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1408, out_features=67, bias=True))

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/cnn.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        pretrained_teacher = {str.replace(k,'model.',''): v for k,v in pretrained_teacher.items()}
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape
        x = self.features(x0)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.softmax(x, dim=1)

class mvit_tiny(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(mvit_tiny, self).__init__()

        model = timm.create_model('mvitv2_tiny', pretrained=True)

        self.model = model 

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages[3].parameters():
            param.requires_grad = True

        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True))

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/mvit_tiny.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape
        x = self.model(x0)
        return torch.softmax(x, dim=1)

class convnextv2_tiny(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(convnextv2_tiny, self).__init__()


        model = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=True)

        self.model = model 

        self.model.head.fc     = nn.Sequential(nn.Linear(in_features=768, out_features=num_classes, bias=True))
        self.model.head.drop.p = 0.5

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages[3].parameters():
            param.requires_grad = True

        for param in self.model.head.parameters():
            param.requires_grad = True

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/next_tiny.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        return torch.softmax(x, dim=1)

class res_model(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(res_model, self).__init__()

        model = models.__dict__['resnet50'](num_classes=365)
        checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        model.load_state_dict(state_dict)

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4[-1].parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2048, out_features=num_classes, bias=True))

        checkpoint = torch.load('/content/drive/MyDrive/checkpoint/res.pth', map_location='cpu')
        pretrained_teacher = checkpoint['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape
        x = self.model(x0)
        return torch.softmax(x, dim=1)

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











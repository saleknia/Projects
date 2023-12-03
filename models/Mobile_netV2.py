import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b5, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import random
from torchvision.models import resnet50, efficientnet_b4, EfficientNet_B4_Weights
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

from mit_semseg.models import ModelBuilder

class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(Mobile_netV2, self).__init__()

        # self.teacher = Mobile_netV2_teacher(num_classes=num_classes)
        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_base_next_89_38/Mobile_NetV2_MIT-67_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.teacher.load_state_dict(pretrained_teacher)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False


        # model = torchvision.models.maxvit_t(weights='DEFAULT')

        # model = timm.create_model('maxvit_tiny_tf_224.in1k', pretrained=True, num_classes=67)

        # model = torchvision.models.regnet_y_400mf(weights='DEFAULT')

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        # model = models.__dict__['resnet50'](num_classes=365)
        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        # # state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
        # # state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}

        # model.load_state_dict(state_dict)

        # self.model = model

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.layer4[-1].parameters():
        #     param.requires_grad = True

        # self.model.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2048, out_features=num_classes, bias=True))

        # # self.teacher.conv1.stride = (1, 1)

        ############################################################
        ############################################################

        # model = torchvision.models.convnext_small(weights='DEFAULT')

        # self.model = model 

        # self.model.classifier[2] = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=768, out_features=num_classes, bias=True))

        # self.model.classifier[0] = nn.Identity()
        
        # for param in self.model.parameters():
        #     param.requires_grad = False


        # for param in self.model.features[5][-3].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[5][-2].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[5][-1].parameters():
        #     param.requires_grad = True

        # # for param in self.model.features[5][6:9].parameters():
        # #     param.requires_grad = True

        # # for param in self.model.features[5][18:27].parameters():
        # #     param.requires_grad = True

        # for param in self.model.features[6].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[7].parameters():
        #     param.requires_grad = True

        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        # # self.teacher = convnext_teacher()
        # # self.teacher = convnext_small()



        # model = torchvision.models.convnext_tiny(weights='DEFAULT')

        # self.model = model 

        # self.model.classifier[2] = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=768, out_features=num_classes, bias=True))

        # # self.model.classifier[0] = nn.Identity()
        
        # for param in self.model.parameters():
        #     param.requires_grad = False


        # for param in self.model.features[5][-3].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[5][-2].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[5][-1].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[5][6:9].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[5][18:27].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[6].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[7].parameters():
        #     param.requires_grad = True

        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        # self.teacher = convnext_teacher()
        # self.teacher = convnext_small()

        ############################################################
        ############################################################

        # model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True)

        # # model = timm.create_model('timm/convnext_nano_ols.d1h_in1k', pretrained=True)

        # self.model = model 

        # self.model.head.fc     = nn.Sequential(nn.Linear(in_features=768, out_features=num_classes, bias=True))

        # self.model.head.drop.p = 0.5

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.stages[3].parameters():
        #     param.requires_grad = True

        # # for param in self.model.stages[3].blocks[1:3].parameters():
        # #     param.requires_grad = False

        # # for param in self.model.stages[2].parameters():
        # #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        ############################################################
        ############################################################

        # model = efficientnet_b3(weights=EfficientNet_B3_Weights)
        # # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)

        # # model.features[0][0].stride = (1, 1)

        # self.features = model.features
        # self.avgpool = model.avgpool

        # for param in self.features[0:7].parameters():
        #     param.requires_grad = False

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1536, out_features=67, bias=True))

        ############################################################
        ############################################################

        # self.features[0][0].stride = (1, 1)

        # model = resnet18(num_classes=365)

        # model = models.__dict__['resnet50'](num_classes=365)

        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # model.load_state_dict(state_dict)
        # self.teacher = model 

        # model = models.__dict__['densenet161'](num_classes=365)

        # checkpoint = torch.load('/content/densenet161_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
        # state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}
        # model.load_state_dict(state_dict)
        
        # self.model_sce = model
        # self.model_seg = ModelBuilder.build_encoder(arch='resnet50', fc_dim=2048, weights='/content/encoder_epoch_30.pth')
        # self.model_obj = torchvision.models.resnet50(weights='DEFAULT')


        # for param in self.model_sce.parameters():
        #     param.requires_grad = False

        # for param in self.model_seg.parameters():
        #     param.requires_grad = False

        # for param in self.model_obj.parameters():
        #     param.requires_grad = False

        # for param in self.model_sce.layer4[-1].parameters():
        #     param.requires_grad = True

        # model = resnet18(num_classes=365)

        # checkpoint = torch.load('/content/wideresnet18_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # model.load_state_dict(state_dict)

        # hacky way to deal with the upgraded batchnorm2D and avgpool layers...

        # for i, (name, module) in enumerate(model._modules.items()):
        #     module = recursion_change_bn(model)

        # model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

        # print(model_seg)
        # print(model_place)

        # self.model   = model
        # self.model.conv1.stride = (1, 1)
        # self.avgpool = torch.nn.AvgPool2d(kernel_size=28, stride=1, padding=0)

        # self.model_place = model_place
        # self.model_seg   = model_seg
        # self.model_cls   = model_cls

        # print(model)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.features.denseblock4.parameters():
        #     param.requires_grad = True

        # for param in self.model_cls.parameters():
        #     param.requires_grad = False

        # for param in self.model.stages[3].parameters():
        #     param.requires_grad = True

        # for param in self.model.layer4.parameters():
        #     param.requires_grad = True

        # self.model.classifier[5] = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=512, out_features=num_classes, bias=True))

        # self.avgpool = model.avgpool

        # # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) 

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=num_classes, bias=True))

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=256, out_features=num_classes, bias=True),
        # )

        # state_dict = torch.load('/content/drive/MyDrive/checkpoint_1/Mobile_NetV2_MIT-67_best.pth', map_location='cpu')['net']
        # self.load_state_dict(state_dict)

        #################################################################################
        #################################################################################

        model = timm.create_model('mvitv2_tiny', pretrained=True)

        self.model = model 

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages[3].blocks[-1].parameters():
            param.requires_grad = True

        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True))

        # # self.teacher = mvit_small()
        
        # # self.teacher = mvit_tiny()

        # self.teacher = mvit_teacher()

        #################################################################################
        #################################################################################

        # model = timm.create_model('convnextv2_tiny', pretrained=True)
        
        # model = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=True)

        # self.model = model 

        # self.model.head.fc     = nn.Sequential(nn.Linear(in_features=768, out_features=num_classes, bias=True))
        # self.model.head.drop.p = 0.5

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.stages[3].parameters():
        #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        # for param in self.model.stages[3].blocks[0].parameters():
        #     param.requires_grad = True

        # for param in self.model.stages[3].downsample.parameters():
        #     param.requires_grad = True

        # for param in self.model.stages[2].blocks[6:9].parameters():
        #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        # self.convnext = convnext_small()
        # self.mvit = mvit_small()

        # self.teacher = efficientnet_teacher()
        
        # self.teacher = convnext_teacher()

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape

        # x = self.features(x0)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        # x1 = self.features[0:4](x0)
        # x2 = self.features[4:6](x1)
        # x3 = self.features[6:9](x2)

        # x_t = self.teacher(x0)

        # x = self.avgpool(x3)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        x = self.model(x0)

        # print(x.shape)

        # x  = self.model(x0)

        # x = self.model(x0)

        # x = self.classifier(emb_s)

        # x = self.convnext(x0)

        return x

        # if self.training:
        #     return x, x_t
        # else:
        #     return x

class mvit_base(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(mvit_base, self).__init__()

        model = timm.create_model('mvitv2_base', pretrained=True)

        self.model = model 

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages[3].parameters():
            param.requires_grad = True

        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True))

        for param in self.model.parameters():
            param.requires_grad = False

        # state_dict = torch.load('/content/drive/MyDrive/checkpoint/base_best.pth', map_location='cpu')['net']
        # self.load_state_dict(state_dict)

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/base_distilled_best.pth', map_location='cpu')
        # pretrained_teacher = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)
        # x = self.classifier(x)
        return x # torch.softmax(x, dim=1)



class mvit_small(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(mvit_small, self).__init__()

        model = timm.create_model('mvitv2_small', pretrained=True)

        self.model = model 

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages[3].parameters():
            param.requires_grad = True

        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True))

        for param in self.model.parameters():
            param.requires_grad = False

        # state_dict = torch.load('/content/drive/MyDrive/checkpoint/small_best.pth', map_location='cpu')['net']
        # self.load_state_dict(state_dict)

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/small_distilled_best.pth', map_location='cpu')
        # pretrained_teacher = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)
        # x = self.classifier(x)
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

        for param in self.model.parameters():
            param.requires_grad = False

        # state_dict = torch.load('/content/drive/MyDrive/checkpoint_tiny/MVITV2_tiny.pth', map_location='cpu')['net']
        # self.load_state_dict(state_dict)

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/tiny_distilled_best.pth', map_location='cpu')
        # pretrained_teacher = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)


    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        return torch.softmax(x, dim=1)

class mvit_teacher(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(mvit_teacher, self).__init__()

        self.base  = mvit_base()
        self.small = mvit_tiny()
        self.tiny  = mvit_tiny()

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/mvit_base.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.base.load_state_dict(pretrained_teacher)

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/mvit_tiny.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.small.load_state_dict(pretrained_teacher)

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/mvit_tiny_s.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.tiny.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        # x = (self.small(x0) + self.tiny(x0) + self.base(x0)) / 2.0

        x_t = self.tiny(x0)
        x_s = self.small(x0)

        return x_t, x_s


class convnext_small(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(convnext_small, self).__init__()

        model = torchvision.models.convnext_small(weights='DEFAULT')

        self.model = model 

        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True))

        # self.model.classifier[0] = nn.Identity()
        
        for param in self.model.parameters():
            param.requires_grad = False

        # for param in self.model.features[5][6:9].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[5][18:27].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[6].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[7][0].parameters():
        #     param.requires_grad = True

        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        # state_dict = torch.load('/content/drive/MyDrive/checkpoint/small_best.pth', map_location='cpu')['net']
        # self.load_state_dict(state_dict)

        # state_dict = torch.load('/content/drive/MyDrive/checkpoint_convnext/small_distilled_best.pth', map_location='cpu')['net']
        # self.load_state_dict(state_dict)

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/next_small.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)
        # x = self.classifier(x)
        return x # torch.softmax(x, dim=1)

class convnext_tiny(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(convnext_tiny, self).__init__()

        model = torchvision.models.convnext_tiny(weights='DEFAULT')

        self.model = model 

        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True))

        # self.model.classifier[0] = nn.Identity()
        
        for param in self.model.parameters():
            param.requires_grad = False

        # for param in self.model.features[5][6:9].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[5][18:27].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[6].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[7][0].parameters():
        #     param.requires_grad = True

        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True


        state_dict = torch.load('/content/drive/MyDrive/checkpoint/Mobile_NetV2_MIT-67_best.pth', map_location='cpu')['net']
        self.load_state_dict(state_dict)

        # state_dict = torch.load('/content/drive/MyDrive/checkpoint_mvitv2_small/MVITV2_small.pth', map_location='cpu')['net']
        # self.load_state_dict(state_dict)

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_convnext/tiny_best_0.pth', map_location='cpu')
        # pretrained_teacher = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)
        # x = self.classifier(x)
        return x

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

        # for param in self.model.parameters():
        #     param.requires_grad = False

        state_dict = torch.load('/content/drive/MyDrive/checkpoint_convnextv2/tiny_best_0.pth', map_location='cpu')['net']
        self.load_state_dict(state_dict)

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_convnextv2/tiny_distilled_best.pth', map_location='cpu')
        # pretrained_teacher = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)
        # x = self.classifier(x)
        return x

class convnext_teacher(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(convnext_teacher, self).__init__()

        self.small = convnext_small()
        self.tiny  = convnext_tiny()

    def forward(self, x0):
        b, c, w, h = x0.shape

        # x = (self.small(x0) + self.tiny(x0)) / 2.0

        x = self.small(x0)

        return x

from torchvision import transforms
transform_test = transforms.Compose([
    transforms.Resize((384, 384)),
])

class efficientnet_teacher(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(efficientnet_teacher, self).__init__()

        self.b2 = B2()
        self.b3 = B3()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape
        x = transform_test(x0)
        x = (self.b2(x0) + self.b3(x0)) / 2.0

        return x


class B3(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(B3, self).__init__()

        model = efficientnet_b3(weights=EfficientNet_B3_Weights)

        # model.features[0][0].stride = (1, 1)

        self.features = model.features

        self.avgpool = model.avgpool

        for param in self.features[0:4].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1536, out_features=num_classes, bias=True))

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_efficientnet/B3_best.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
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
        return x

class B2(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(B2, self).__init__()

        model = efficientnet_b2(weights=EfficientNet_B2_Weights)

        # model.features[0][0].stride = (1, 1)

        self.features = model.features

        self.avgpool = model.avgpool

        for param in self.features[0:4].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1408, out_features=num_classes, bias=True))

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_efficientnet/B2_best.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
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
        return x

# class Mobile_netV2(nn.Module):
#     def __init__(self, num_classes=40, pretrained=True):
#         super(Mobile_netV2, self).__init__()

#         # self.teacher = Mobile_netV2_teacher()
#         # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_VL_96_97/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
#         # pretrained_teacher = loaded_data_teacher['net']
#         # a = pretrained_teacher.copy()
#         # for key in a.keys():
#         #     if 'teacher' in key:
#         #         pretrained_teacher.pop(key)
#         # self.teacher.load_state_dict(pretrained_teacher)

#         # for param in self.teacher.parameters():
#         #     param.requires_grad = False

#         # self.teacher = Mobile_netV2_loss()
#         # self.teacher.eval()
#         # for param in self.teacher.parameters():
#         #     param.requires_grad = False

#         # model = efficientnet_b2(weights=EfficientNet_B2_Weights)

#         # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)

#         # model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)

#         # model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights)

#         model = torchvision.models.convnext_tiny(weights='DEFAULT')
#         model.features[0][0].stride = (2, 2)
        
#         # model.features[0][0].stride = (1, 1)

#         self.features = model.features

#         for param in self.features[0:6].parameters():
#             param.requires_grad = False

#         # for param in self.features[0:4].parameters():
#         #     param.requires_grad = False

#         self.avgpool = model.avgpool

#         # self.classifier = nn.Sequential(
#         #     nn.Dropout(p=0.4, inplace=True),
#         #     nn.Linear(in_features=1280, out_features=512, bias=True),
#         #     nn.Dropout(p=0.4, inplace=True),
#         #     nn.Linear(in_features=512, out_features=256, bias=True),
#         #     nn.Dropout(p=0.4, inplace=True),
#         #     nn.Linear(in_features=256, out_features=40, bias=True),
#         # )

#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5, inplace=True),
#             nn.Linear(in_features=1280, out_features=40, bias=True))
        
#     def forward(self, x0):
#         b, c, w, h = x0.shape

#         # x_t, x1_t, x2_t = self.teacher(x0)

#         x_t, x1_t, x2_t, x3_t = self.teacher(x0)

#         # print(x_t)

#         # x1 = self.features[0:7](x0)
#         # x2 = self.features[7:8](x1)
#         # x3 = self.features[8:9](x2)

#         x1 = self.features[0:4](x0)
#         x2 = self.features[4:6](x1)
#         x3 = self.features[6:9](x2)

#         # x0 = self.features[0:6](x0)
#         # x1 = self.features[6:7](x0)
#         # x2 = self.features[7:8](x1)
#         # x3 = self.features[8:9](x2)

#         # x3 = self.features(x0)

#         x = self.avgpool(x3)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)

#         # print(x1.shape)
#         # print(x2.shape)
#         # print(x3.shape)

#         # print(x1_t.shape)
#         # print(x2_t.shape)
#         # print(x3_t.shape)

#         if self.training:
#             return x, x_t, x1, x2, x3, x1_t, x2_t, x3_t
#         else:
#             return x

#         # return x

#         # if self.training:
#         #     return x#, x_t#, x1, x2, x_t, x1_t, x2_t
#         # else:
#         #     return self.avgpool(x3) # torch.softmax(x, dim=1)

class Mobile_netV2_teacher(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_teacher, self).__init__()

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # # model.features[0][0].stride = (1, 1)

        # self.features = model.features
        # self.avgpool = model.avgpool

        # for param in self.features[0:9].parameters():
        #     param.requires_grad = False

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=67, bias=True))

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=1536, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256, out_features=40, bias=True),
        # )

        # teacher = models.__dict__['resnet50'](num_classes=365)
        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # teacher.load_state_dict(state_dict)

        # self.teacher = teacher

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        # for param in self.teacher.layer4[-1].parameters():
        #     param.requires_grad = True

        # self.teacher.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2048, out_features=num_classes, bias=True))
        # self.teacher.conv1.stride = (1, 1)

        # self.avgpool = self.teacher.avgpool

        # teacher = models.__dict__['resnet18'](num_classes=365)
        # checkpoint = torch.load('/content/resnet18_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # teacher.load_state_dict(state_dict)

        # self.teacher = teacher

        # self.teacher.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2048, out_features=67, bias=True))
        # self.teacher.conv1.stride = (1, 1)

        # self.avgpool = self.teacher.avgpool

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        # model = timm.create_model('convnextv2_base', pretrained=True)

        # self.model = model 

        # self.model.head.fc = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1024, out_features=num_classes, bias=True))

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.stages[3].parameters():
        #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)

        model.features[0][0].stride = (1, 1)

        self.features = model.features

        self.avgpool = model.avgpool

        for param in self.features[0:6].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True))

    def forward(self, x0):
        b, c, w, h = x0.shape

        x1 = self.features[0:4](x0)
        x2 = self.features[4:6](x1)
        x3 = self.features[6:9](x2)

        # x_t = self.teacher(x0)

        x = self.avgpool(x3)
        x = x.view(x.size(0), -1)
        return x
        # x = self.classifier(x)

        # x0 = self.teacher.conv1(x0)
        # x0 = self.teacher.bn1(x0)
        # x0 = self.teacher.relu(x0)
        # x0 = self.teacher.maxpool(x0)
        # x1 = self.teacher.layer1(x0)
        # x2 = self.teacher.layer2(x1)
        # x3 = self.teacher.layer3(x2)
        # x4 = self.teacher.layer4(x3)

        # x = self.avgpool(x3) 
        # x = x.view(x.size(0), -1)
        # x = self.teacher.fc(x)

        # x = self.model(x0)

        # x_stem  = self.model.stem(x0)
        # x_stage = self.model.stages(x_stem)
        # x_norm  = self.model.norm_pre(x_stage)
        # x_head  = self.model.head(x_norm)

        # return torch.softmax(x, dim=1)

        # x_stem  = self.model.stem(x0)

        # x0 = self.model.stages[0](x_stem)
        # x1 = self.model.stages[1](x0)
        # x2 = self.model.stages[2](x1)
        # x3 = self.model.stages[3](x2)

        # x_norm  = self.model.norm_pre(x3)
        # x       = self.model.head(x_norm)


        # return x, x2, x3


# class Mobile_netV2(nn.Module):
#     def __init__(self, num_classes=40, pretrained=True):
#         super(Mobile_netV2, self).__init__()

#         self.teacher = Mobile_netV2_teacher()
#         loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_B3_86_80/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
#         pretrained_teacher = loaded_data_teacher['net']
#         a = pretrained_teacher.copy()
#         for key in a.keys():
#             if 'teacher' in key:
#                 pretrained_teacher.pop(key)
#         self.teacher.load_state_dict(pretrained_teacher)

#         for param in self.teacher.parameters():
#             param.requires_grad = False

#         # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)

#         model = efficientnet_b0(weights=EfficientNet_B0_Weights)

#         # model.features[0][0].stride = (1, 1)

#         for param in model.features[0:5].parameters():
#             param.requires_grad = False

#         self.features = model.features
#         self.avgpool = model.avgpool

#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5, inplace=True),
#             nn.Linear(in_features=1280, out_features=40, bias=True),
#         )

#         # self.classifier = nn.Sequential(
#         #     nn.Dropout(p=0.4, inplace=True),
#         #     nn.Linear(in_features=1280, out_features=512, bias=True),
#         #     nn.Dropout(p=0.4, inplace=True),
#         #     nn.Linear(in_features=512 , out_features=256, bias=True),
#         #     nn.Dropout(p=0.4, inplace=True),
#         #     nn.Linear(in_features=256 , out_features=40, bias=True),
#         # )

#     def forward(self, x0):
#         b, c, w, h = x0.shape

#         x1_t, x2_t = self.teacher(x0)

#         x1 = self.features[0:7](x0)
#         x2 = self.features[7:8](x1)
#         x3 = self.features[8:9](x2)

#         # x = self.features(x0)

#         x = self.avgpool(x3) 
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)

#         if self.training:
#             return x, x1, x2, x1_t, x2_t
#         else:
#             return torch.softmax(x, dim=1)











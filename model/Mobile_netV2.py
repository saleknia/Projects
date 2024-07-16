from re import S, X
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

from timm.layers import LayerNorm2d

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

from pyts.image import GramianAngularField as GAF
from pyts.preprocessing import MinMaxScaler as scaler
from pyts.image import MarkovTransitionField as MTF
from mit_semseg.models import ModelBuilder
from pyts.image import RecurrencePlot

from torchvision import transforms
transform_test = transforms.Compose([
    transforms.Resize((384, 384)),
])
from torchvision.transforms import FiveCrop, Lambda
from efficientvit.seg_model_zoo import create_seg_model

from efficientvit.cls_model_zoo import create_cls_model


class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(Mobile_netV2, self).__init__()

        # model = resnet18(num_classes=365)
        # checkpoint = torch.load('/content/wideresnet18_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # model.load_state_dict(state_dict)

        # self.model = model
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=512, out_features=num_classes, bias=True),
        # )
        
        ############################################################
        ############################################################

        # model = torchvision.models.convnext_small(weights='DEFAULT')

        # self.model = model 

        # self.model.classifier[2] = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=768, out_features=num_classes, bias=True))

        # # self.model.classifier[0] = nn.Identity()
        
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.features[6].parameters():
        #     param.requires_grad = True

        # for param in self.model.features[7].parameters():
        #     param.requires_grad = True

        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        # # self.teacher = convnext_teacher()
        # # self.teacher = convnext_small()

        ############################################################
        ############################################################

        # model = efficientnet_b2(weights=EfficientNet_B2_Weights)

        # model.features[0][0].stride = (1, 1)

        # self.model = model
        # # self.avgpool = model.avgpool

        # for param in self.model.features[0:5].parameters():
        #     param.requires_grad = False

        # self.model.classifier[0].p            = 0.5
        # self.model.classifier[1].out_features = 67

        ############################################################
        ############################################################

        # scene      = models.__dict__['resnet50'](num_classes=365)
        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        # scene.load_state_dict(state_dict)

        # self.scene = scene

        # for param in self.scene.parameters():
        #     param.requires_grad = False

        # self.scene.fc = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=2048, out_features=256, bias=True),
        # )

        # model = timm.create_model('timm/convnextv2_tiny.fcmae_ft_in1k', pretrained=True)

        # self.model = model 

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.model.head.fc = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=768, out_features=256, bias=True),
        # )

        # seg = create_seg_model(name="b2", dataset="ade20k", weight_url="/content/drive/MyDrive/b2.pt")

        # seg.input_stem.op_list[0].conv.stride  = (1, 1)
        # seg.input_stem.op_list[0].conv.padding = (0, 0)

        # # seg.head.output_ops[0].op_list[0] = nn.Identity()

        # self.seg = seg

        # for param in self.seg.parameters():
        #     param.requires_grad = False

        # for param in self.seg.head.parameters():
        #     param.requires_grad = True

        # for param in self.seg.backbone.stages[-1].parameters():
        #     param.requires_grad = True

        # self.avgpool = nn.AvgPool2d()
        # self.dropout = nn.Dropout(0.5)
        # self.fc_SEM  = nn.Sequential(nn.Linear(in_features=2400, out_features=num_classes, bias=True))

        #################################################################################
        #################################################################################

        # # model      = models.__dict__['resnet50'](num_classes=365)
        # # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        # # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        # model      = models.__dict__['resnet50'](num_classes=365)
        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        # model.load_state_dict(state_dict)

        # self.model = model

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.layer4[-1].parameters():
        #     param.requires_grad = True

        # self.model.fc = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=2048, out_features=67, bias=True),
        # )

        
        # #################################################################################
        # #################################################################################


        model      = models.__dict__['densenet161'](num_classes=365)
        checkpoint = torch.load('/content/densenet161_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
        state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}

        model.load_state_dict(state_dict)

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        # for param in self.model.features.denseblock4.parameters():
        #     param.requires_grad = True

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2208, out_features=67, bias=True),
        )

        # #################################################################################
        # #################################################################################

        # model = timm.create_model('mvitv2_tiny', pretrained=True, features_only=True)
        # head  = timm.create_model('mvitv2_tiny', pretrained=True).head

        # self.model = model

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.model.stages[2:].parameters():
        #     param.requires_grad = True

        # self.avgpool = nn.AvgPool2d(7, stride=1)

        # self.head = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=768, out_features=num_classes, bias=True))

        #################################################################################
        #################################################################################

        # model = create_seg_model(name="b3", dataset="ade20k", weight_url="/content/drive/MyDrive/b3.pt").backbone
        # # head  = create_cls_model(name="b2", weight_url="/content/drive/MyDrive/b2-r224.pt").head

        # model.input_stem.op_list[0].conv.stride  = (1, 1)
        # model.input_stem.op_list[0].conv.padding = (0, 0)

        # self.model = model
        # # self.head  = head

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # # self.head.op_list[3] = nn.Sequential(
        # #     nn.Dropout(p=0.5, inplace=True),
        # #     nn.Linear(in_features=2560, out_features=num_classes, bias=True),
        # # )

        # # for param in self.model.stages[-1].parameters():
        # #     param.requires_grad = True

        # # for param in self.model.stages[-2].parameters():
        # #     param.requires_grad = True

        # # for param in self.head.parameters():
        # #     param.requires_grad = True

        # self.dropout = nn.Dropout(0.5)
        # self.avgpool = nn.AvgPool2d(14, stride=1)
        # self.fc_SEM  = nn.Linear(512, num_classes)

        #################################################################################
        #################################################################################

        # model = create_cls_model(name="b2", weight_url="/content/drive/MyDrive/b2-r256.pt")

        # self.model = model

        # print(model)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.stages[-1].parameters():
        #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True
      
        # self.dropout = nn.Dropout(0.5)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.fc_SEM  = nn.Linear(384, num_classes)

        #################################################################################
        #################################################################################

        # model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True)
        # model = timm.create_model('tf_efficientnet_b0.in1k', pretrained=True)

        # self.model = model 

        # self.model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=1280, out_features=num_classes, bias=True))

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.blocks[4:].parameters():
        #     param.requires_grad = True

        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        #################################################################################
        #################################################################################

        # model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True)

        # self.model = model 

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.model.head.fc = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=768, out_features=num_classes, bias=True),
        # )

        # for param in self.model.stages[2:].parameters():
        #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        #################################################################################
        #################################################################################

        # model = timm.create_model('timm/maxvit_tiny_tf_224.in1k', pretrained=True)

        # self.model = model 

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.model.head.fc = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=512, out_features=num_classes, bias=True),
        # )

        # for param in self.model.stages[-1].parameters():
        #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        ##################################################################################
        ##################################################################################

        # self.model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True, features_only=True)
        
        # self.head  = timm.create_model('convnext_tiny.fb_in1k', pretrained=True).head 
        
        # self.head.fc = nn.Sequential(
        #             nn.Dropout(p=0.5, inplace=True),
        #             nn.Linear(in_features=768, out_features=num_classes, bias=True),
        #         )

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.stages_2.parameters():
        #     param.requires_grad = True

        # for param in self.model.stages_3.parameters():
        #     param.requires_grad = True

        # for param in self.head.parameters():
        #     param.requires_grad = True

        ##################################################################################
        ##################################################################################

        # model = timm.create_model('timm/efficientvit_b2.r224_in1k', pretrained=True)

        # self.model = model 

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.model.head.classifier[4] = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=2560, out_features=num_classes, bias=True),
        # )

        # for param in self.model.stages[-1].parameters():
        #     param.requires_grad = True

        # # for param in self.model.stages[-2].parameters():
        # #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        #################################################################################
        #################################################################################

        # classifier = timm.create_classifier('tf_efficientnet_b0', pretrained=True)

        # self.classifier = classifier 

        # for param in self.classifier.blocks[0:5].parameters():
        #     param.requires_grad = False

        # for param in self.classifier.conv_stem.parameters():
        #     param.requires_grad = False

        # self.classifier.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1280, out_features=num_classes, bias=True),
        # )

        # for param in self.model.blocks[5][10:15].parameters():
        #     param.requires_grad = True

        # self.avgpool = torch.nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

        # self.model = B0()

        # self.transform_0 = transforms.Compose([transforms.Resize((224, 224))])

        # self.transform = transforms.Compose([
        #     transforms.Resize((384, 384)),
        # ])

        # self.count = 0.0
        # self.batch = 0.0

        # self.transform = torchvision.transforms.Compose([FiveCrop(224), Lambda(lambda crops: torch.stack([crop for crop in crops]))])

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/tiny.pth', map_location='cpu')
        # pretrained_teacher  = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

        #################################
        #################################
        # self.inspector = None
        # self.inspector = femto()
        #################################
        #################################
 
        # self.b0 = maxvit()
        # self.b1 = mvit_tiny()
        # self.b2 = convnext_tiny()

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/best.pth', map_location='cpu')
        # pretrained_teacher  = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

        # self.model = teacher_ensemble()

        # self.count = 0.0
        # self.batch = 0.0
        # self.transform = torchvision.transforms.Compose([FiveCrop(224), Lambda(lambda crops: torch.stack([crop for crop in crops]))])

    def forward(self, x_in):

        # b, c, w, h = x_in.shape

        # x = self.model(x_in)
        # x = x['stage_final']
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # x = self.fc_SEM(x)

        # x0, x1, x2, x3, x4 = self.model(x_in)

        # x4 = self.avgpool(x4)
        # x4 = x4.view(x4.size(0), -1)

        # x = self.head(x4)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # x = self.fc_SEM(x)

        x = self.model(x_in)

        # if (not self.training):

        #     self.batch = self.batch + 1.0

        #     if self.batch == 1335:
        #         print(self.count)

        #     x0, x1 = x_in[0], x_in[1]
            
        #     x = self.model(x0)
        #     # x = self.head(x[4])
        #     x = torch.softmax(x, dim=1)

        #     if (x.max() < 0.9):

        #         y = self.transform(x1)
        #         ncrops, bs, c, h, w = y.size()
        #         x = self.model(y.view(-1, c, h, w))
        #         # x = self.head(x[4])
        #         # x = torch.softmax(x, dim=1).mean(0, keepdim=True)
        #         # x = x.mean(0, keepdim=True)

        #         x = torch.softmax(x, dim=1)
        #         a, b, c = torch.topk(x.max(dim=1).values, 3).indices
        #         x = ((x[a] + x[b] + x[c]) / 3.0).unsqueeze(dim=0)

        #         self.count = self.count + 1.0
        
        # else:
        #     b, c, w, h = x_in.shape
        #     x = self.model(x_in)


        return x

        # if self.training:
        #     return x, x2, x3, t2, t3
        # else:
        #     return x


class teacher_ensemble(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(teacher_ensemble, self).__init__()

        self.maxvit = maxvit_model()
        self.scene  = scene()
        self.seg    = seg()



    def forward(self, x0):

        # b, c, w, h = x0.shape

        a = self.maxvit(x0)
        b = self.scene(x0)
        c = self.seg(x0) 

        x = (a + b + c) / 3.0

        return x

class scene(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(scene, self).__init__()
       
        model      = models.__dict__['resnet50'](num_classes=365)
        checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        model.load_state_dict(state_dict)

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4[-1].parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2048, out_features=67, bias=True),
        )
        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/scene.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

        self.transform = torchvision.transforms.Compose([FiveCrop(224), Lambda(lambda crops: torch.stack([crop for crop in crops]))])

    def forward(self, x_in):
        # b, c, w, h = x0.shape

        # x = self.model(x0)

        if (not self.training):

            x0, x1 = x_in[0], x_in[1]
            
            x = self.model(x0)
            x = torch.softmax(x, dim=1)

            if (x.max() < 0.8):

                y = self.transform(x1)
                ncrops, bs, c, h, w = y.size()
                x = self.model(y.view(-1, c, h, w))
                # x = torch.softmax(x, dim=1).mean(0, keepdim=True)

                x = torch.softmax(x, dim=1)
                a, b, c = torch.topk(x.max(dim=1).values, 3).indices
                x = ((x[a] + x[b] + x[c]) / 3.0).unsqueeze(dim=0)
        
        else:
            b, c, w, h = x_in.shape
            x = self.model(x_in)

        return x # torch.softmax(x, dim=1)


class seg(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(seg, self).__init__()
       
        model = create_seg_model(name="b2", dataset="ade20k", weight_url="/content/drive/MyDrive/b2.pt").backbone

        model.input_stem.op_list[0].conv.stride  = (1, 1)
        model.input_stem.op_list[0].conv.padding = (0, 0)

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages[-1].parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AvgPool2d(14, stride=1)
        self.fc_SEM  = nn.Linear(384, num_classes)

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/seg.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

        self.transform = torchvision.transforms.Compose([FiveCrop(224), Lambda(lambda crops: torch.stack([crop for crop in crops]))])

    def forward(self, x_in):
        # b, c, w, h = x0.shape

        # x = self.model(x0)
        # x = x['stage_final']
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # x = self.fc_SEM(x)

        if (not self.training):

            x0, x1 = x_in[0], x_in[1]
                
            x = self.model(x0)
            x = x['stage_final']
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc_SEM(x)
            x = torch.softmax(x, dim=1)

            if (x.max() < 0.8):

                y = self.transform(x1)
                ncrops, bs, c, h, w = y.size()
                # x = self.model(y.view(-1, c, h, w))
                # x = torch.softmax(x, dim=1).mean(0, keepdim=True)

                x = self.model(y.view(-1, c, h, w))
                x = x['stage_final']
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.fc_SEM(x)
                # x = torch.softmax(x, dim=1).mean(0, keepdim=True)

                x = torch.softmax(x, dim=1)
                a, b, c = torch.topk(x.max(dim=1).values, 3).indices
                x = ((x[a] + x[b] + x[c]) / 3.0).unsqueeze(dim=0)
        
        else:
            b, c, w, h = x_in.shape
            x = self.model(x_in)

        return x # torch.softmax(x, dim=1)

class maxvit_model(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(maxvit_model, self).__init__()

        model = timm.create_model('timm/maxvit_tiny_tf_224.in1k', pretrained=True)

        self.model = model 

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=num_classes, bias=True),
        )

        for param in self.model.stages[-1].parameters():
            param.requires_grad = True

        for param in self.model.head.parameters():
            param.requires_grad = True

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/max.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

        self.transform = torchvision.transforms.Compose([FiveCrop(224), Lambda(lambda crops: torch.stack([crop for crop in crops]))])
        
    def forward(self, x_in):
        # b, c, w, h = x0.shape

        if (not self.training):

            x0, x1 = x_in[0], x_in[1]
            
            x = self.model(x0)
            x = torch.softmax(x, dim=1)

            if (x.max() < 0.8):

                y = self.transform(x1)
                ncrops, bs, c, h, w = y.size()
                x = self.model(y.view(-1, c, h, w))
                # x = torch.softmax(x, dim=1).mean(0, keepdim=True)

                x = torch.softmax(x, dim=1)
                a, b, c = torch.topk(x.max(dim=1).values, 3).indices
                x = ((x[a] + x[b] + x[c]) / 3.0).unsqueeze(dim=0)
        
        else:
            b, c, w, h = x_in.shape
            x = self.model(x_in)

        return x # torch.softmax(x, dim=1)


class mvit_tiny(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(mvit_tiny, self).__init__()

        model = timm.create_model('mvitv2_tiny', pretrained=True, features_only=True)
        head  = timm.create_model('mvitv2_tiny', pretrained=True).head

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.model.stages[2:].parameters():
            param.requires_grad = True

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True))

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/mvit.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)


    def forward(self, x0):

        b, c, w, h = x0.shape

        x0, x1, x2, x3 = self.model(x0)

        x3 = self.avgpool(x3)
        x3 = x3.view(x3.size(0), -1)
        x  = self.head(x3)

        return torch.softmax(x, dim=1)


class convnext_tiny(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(convnext_tiny, self).__init__()

        self.model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True, features_only=True)
        
        self.head  = timm.create_model('convnext_tiny.fb_in1k', pretrained=True).head 
        
        self.head.fc = nn.Sequential(
                    nn.Dropout(p=0.5, inplace=True),
                    nn.Linear(in_features=768, out_features=num_classes, bias=True),
                )

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages_2.parameters():
            param.requires_grad = True

        for param in self.model.stages_3.parameters():
            param.requires_grad = True

        for param in self.head.parameters():
            param.requires_grad = True

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/next.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x0, x1, x2, x3 = self.model(x0)

        x = self.head(x3)

        return torch.softmax(x, dim=1)

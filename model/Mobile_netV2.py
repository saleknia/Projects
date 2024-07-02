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


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()


        self.up = nn.ConvTranspose2d(768, 384, 3, stride=2, padding=1, output_padding=1)
            
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(Mobile_netV2, self).__init__()


        # scene = models.__dict__['resnet50'](num_classes=365)
        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')

        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        # # state_dict = {str.replace(k,'.1','1'): v for k,v in state_dict.items()}
        # # state_dict = {str.replace(k,'.2','2'): v for k,v in state_dict.items()}

        # scene.load_state_dict(state_dict)

        # self.scene = scene

        # for param in self.scene.parameters():
        #     param.requires_grad = False

        # for param in self.scene.layer4[-1].parameters():
        #     param.requires_grad = True

        # self.scene.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features=2048, out_features=num_classes, bias=True))

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

        self.model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True, features_only=True, out_indices=[0, 1, 2, 3])

        # self.model.stem_0.stride = (2, 2)
        
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

        # self.decode = DecoderBottleneckLayer(in_channels=768, n_filters=384, use_transpose=True)

        self.up = nn.ConvTranspose2d(768, 384, 3, stride=2, padding=1, output_padding=1)

        # self.head.norm = LayerNorm2d(384) 

        # seg = create_seg_model(name="b2", dataset="ade20k", weight_url="/content/drive/MyDrive/b2.pt").backbone

        # seg.input_stem.op_list[0].conv.stride  = (1, 1)
        # seg.input_stem.op_list[0].conv.padding = (0, 0)

        # # seg.head.output_ops[0].op_list[0] = nn.Identity()

        # # self.fusion = nn.Conv2d(150, , kernel_size=(1, 1), stride=(1, 1))

        # self.seg = seg

        # for param in self.seg.parameters():
        #     param.requires_grad = False

        # # for param in self.seg.head.parameters():
        # #     param.requires_grad = True

        # # for param in self.seg.stages[-1].op_list[-2:].parameters():
        # #     param.requires_grad = True

        # self.avgpool = nn.AvgPool2d(14, stride=14)
        # # self.dropout = nn.Dropout(0.5)
        # self.fc_SEM  = nn.Sequential(
        #     nn.Linear(in_features=384, out_features=256, bias=True),
        #     nn.Sigmoid()
        # )
        #################################################################################
        #################################################################################

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

        # model = timm.create_model('mvitv2_tiny', pretrained=True)

        # self.model = model

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.stages[3].parameters():
        #     param.requires_grad = True

        # self.model.head = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=768, out_features=num_classes, bias=True))

        #################################################################################
        #################################################################################

        # model = create_seg_model(name="b3", dataset="ade20k", weight_url="/content/drive/MyDrive/b3.pt").backbone

        # model.input_stem.op_list[0].conv.stride  = (1, 1)
        # model.input_stem.op_list[0].conv.padding = (0, 0)

        # self.model = model

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.stages[-1].op_list[6:10].parameters():
        #     param.requires_grad = True

        # self.dropout = nn.Dropout(0.5)
        # self.avgpool = nn.AvgPool2d(14, stride=1)
        # self.fc_SEM  = nn.Linear(512, 67)

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

        # for param in self.model.stages[-1].parameters():
        #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        #################################################################################
        #################################################################################

        # model = timm.create_model('timm/maxvit_tiny_tf_224.in1k', pretrained=True)

        # self.model = model 
        # self.head  = model.head

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.model.head = nn.Identity()

        # self.head.fc = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=512, out_features=num_classes, bias=True),
        # )

        # for param in self.model.stages[-1].parameters():
        #     param.requires_grad = True

        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        # #################################################################################
        # #################################################################################

        # model = timm.create_model('timm/efficientvit_b1.r224_in1k', pretrained=True)

        # self.model = model 

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.model.head.classifier[4] = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(in_features=1600, out_features=num_classes, bias=True),
        # )

        # for param in self.model.stages[-1].parameters():
        #     param.requires_grad = True

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

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/next.pth', map_location='cpu')
        # pretrained_teacher  = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

        # self.teacher = maxvit_model()


    def forward(self, x_in):

        b, c, w, h = x_in.shape

        # x = self.seg(x_in)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # x = self.fc_SEM(x)

        # x = self.seg(x_in)
        # x = x['stage_final']
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # x = self.fc_SEM(x)

        x0, x1, x2, x3 = self.model(x_in)

        x3 = self.up(x3)

        x  = torch.cat([x2, x3], dim=1)
        
        x = self.head(x)

        # x = self.head_1(x3)

        # x = x * y
        # x = self.fc(x)

        # if self.training:
        # t = self.teacher(x_in)

        # x = self.b0(x_in) + self.b1(x_in) + self.b2(x_in)
 
            # x = torch.softmax(self.model(x_in), dim=1) 

            # x = (self.tiny(x_in) + self.small(x_in) + self.base(x_in)) / 3.0

        # self.batch = self.batch + 1.0

        # if self.batch == 1335:
        #     print(self.count)

            # xt = self.tiny(x_in) 
            # x  = xt

            # if x.max() <= 0.8:
            #     self.count = self.count + 1.0

            # if (0.5 < x.max()) and (x.max() <= 0.8): 
            #     # self.count = self.count + 1.0
            #     xs = self.small(x_in)
            #     x  = (xt + xs) / 2.0

            # if x.max() <= 0.5: 
            #     # self.count = self.count + 1.0
            #     xs = self.small(x_in)
            #     xb = self.base(x_in)
            #     x  = (xt + xs + xb) / 3.0

        # else:
        #     x = self.model(x_in)


        return x

        # if self.training:
        #     return x, x2, x3, t2, t3
        # else:
        #     return x


class teacher_ensemble(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(teacher_ensemble, self).__init__()

        self.b0 = maxvit_model()
        self.b1 = mvit_tiny()
        self.b2 = convnext_tiny()

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=768, out_features=40, bias=True),
        )

    def forward(self, x0):
        b, c, w, h = x0.shape

        a = self.b0(x0)
        b = self.b1(x0)
        c = self.b2(x0) 

        # a = torch.softmax(a, dim=1)
        # b = torch.softmax(b, dim=1)
        # c = torch.softmax(c, dim=1)

        x = torch.cat([a, b, c], dim=1)

        x = self.fc(x)

        return x


class b3(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(b3, self).__init__()

        model = create_seg_model(name="b2", dataset="ade20k", weight_url="/content/drive/MyDrive/b2.pt").backbone

        model.input_stem.op_list[0].conv.stride  = (1, 1)
        model.input_stem.op_list[0].conv.padding = (0, 0)

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        # for param in self.model.stages[-1].op_list[8:10].parameters():
        #     param.requires_grad = True

        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AvgPool2d(14, stride=1)
        self.fc_SEM  = nn.Linear(384, 256)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/b3.pth', map_location='cpu')
        # pretrained_teacher  = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

        # self.fc_SEM = nn.Identity()

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)
        x = x['stage_final']
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_SEM(x)

        return x # torch.softmax(x, dim=1)


class maxvit_model(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
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

        for param in self.model.parameters():
            param.requires_grad = False

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/maxvit.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

        self.head       = self.model.head
        self.model.head = nn.Identity()

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        x0 = self.head(x[:,:,0:5,0:5]).softmax(dim=1)
        x1 = self.head(x[:,:,5:8,0:5]).softmax(dim=1)
        x2 = self.head(x[:,:,0:5,5:8]).softmax(dim=1)
        x3 = self.head(x[:,:,5:8,5:8]).softmax(dim=1)

        a = x0.max()
        b = x1.max()
        c = x2.max()
        d = x3.max()

        if a>b and a>c and a>d:
            x = x0

        if b>a and b>c and b>d:
            x = x1

        if c>b and c>a and c>d:
            x = x2

        if d>b and d>c and d>a:
            x = x3

        # x = (x0 + x1 + x2 + x3) / 4.0

        return x # torch.softmax(x, dim=1)


class small(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(small, self).__init__()

        model = timm.create_model('timm/convnext_small.fb_in1k', pretrained=True)

        self.model = model 

        self.model.head.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=256, bias=True),
        )

        for param in self.model.parameters():
            param.requires_grad = False

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/small.pth', map_location='cpu')
        # pretrained_teacher  = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

        # self.model.head.fc = nn.Identity()

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        return x # torch.softmax(x, dim=1)



class tiny(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(tiny, self).__init__()

        model = timm.create_model('timm/convnextv2_tiny.fcmae_ft_in1k', pretrained=True)

        self.model = model 

        self.model.head.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True),
        )

        for param in self.model.parameters():
            param.requires_grad = False

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/tiny.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        return torch.softmax(x, dim=1)

class nano(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(nano, self).__init__()

        model = timm.create_model('timm/convnextv2_nano.fcmae_ft_in1k', pretrained=True)

        self.model = model 

        self.model.head.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=640, out_features=num_classes, bias=True),
        )

        for param in self.model.parameters():
            param.requires_grad = False

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/nano.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        return torch.softmax(x, dim=1)


class pico(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(pico, self).__init__()

        model = timm.create_model('timm/convnextv2_pico.fcmae_ft_in1k', pretrained=True)

        self.model = model 

        self.model.head.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=512, out_features=num_classes, bias=True),
        )

        for param in self.model.parameters():
            param.requires_grad = False

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/pico.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        return torch.softmax(x, dim=1)


class s(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(s, self).__init__()

        scene = models.__dict__['resnet50'](num_classes=365)
        checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        scene.load_state_dict(state_dict)

        self.scene = scene

        for param in self.scene.parameters():
            param.requires_grad = False

        self.scene.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2048, out_features=256, bias=True),
        )



        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/scene.pth', map_location='cpu')
        # pretrained_teacher = loaded_data_teacher['net']
        # pretrained_teacher = {str.replace(k,'model.',''): v for k,v in pretrained_teacher.items()}
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

        # self.scene.fc = self.scene.fc[1]

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.scene(x0)

        return x # torch.softmax(x, dim=1)

class efficientnet_teacher(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(efficientnet_teacher, self).__init__()

        model = timm.create_model('tf_efficientnetv2_s', pretrained=True)

        self.model = model 

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True),
        )

        for param in self.model.parameters():
            param.requires_grad = False

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/enet_small.pth', map_location='cpu')
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


class B5(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(B5, self).__init__()

        model = efficientnet_b5(weights=EfficientNet_B5_Weights)

        model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool  = model.avgpool

        for param in self.features[0:5].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2048, out_features=67, bias=True))

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/cnn_B5.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        pretrained_teacher = {str.replace(k,'model.',''): v for k,v in pretrained_teacher.items()}
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x0):
        b, c, w, h = x0.shape
        # x = transform_test(x0)
        x = self.features(x0)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x # torch.softmax(x, dim=1)

class B0(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(B0, self).__init__()

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        # model.features[0][0].stride = (1, 1)

        self.features = model.features
        self.avgpool  = model.avgpool

        for param in self.features[0:6].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True))

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/cnn.pth', map_location='cpu')
        # pretrained_teacher = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape
        # x = transform_test(x0)
        x = self.features(x0)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if not self.training:
            x = torch.softmax(x, dim=1)
        return x

class mvit_base(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
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

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/base.pth', map_location='cpu')
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
        return torch.softmax(x, dim=1)

class mvit_small(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(mvit_small, self).__init__()

        model = timm.create_model('mvitv2_small', pretrained=True)

        self.model = model 

        for param in self.model.parameters():
            param.requires_grad = False

        # for param in self.model.stages[3].parameters():
        #     param.requires_grad = True

        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=768, out_features=256, bias=True))

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/small.pth', map_location='cpu')
        # pretrained_teacher = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()

        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.load_state_dict(pretrained_teacher)

        # self.model.head = nn.Identity()

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)
        # x = self.classifier(x)
        return x # torch.softmax(x, dim=1)


class mvit_tiny(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
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

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/mvit.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        a = pretrained_teacher.copy()

        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.load_state_dict(pretrained_teacher)

        self.head       = self.model.head
        self.model.head = nn.Identity()
        self.model.norm = nn.Identity()

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.model(x0)

        print(x.shape)
        # x = self.classifier(x)
        return x # torch.softmax(x, dim=1)


class mvit_teacher(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(mvit_teacher, self).__init__()

        # self.base  = mvit_base()
        self.small = mvit_tiny()
        self.tiny  = mvit_tiny()

        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/mvit_base.pth', map_location='cpu')
        # pretrained_teacher  = loaded_data_teacher['net']
        # a = pretrained_teacher.copy()
        # for key in a.keys():
        #     if 'teacher' in key:
        #         pretrained_teacher.pop(key)
        # self.base.load_state_dict(pretrained_teacher)

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/mvit_tiny_0.pth', map_location='cpu')
        pretrained_teacher  = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.small.load_state_dict(pretrained_teacher)

        loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint/mvit_tiny_1.pth', map_location='cpu')
        pretrained_teacher = loaded_data_teacher['net']
        a = pretrained_teacher.copy()
        for key in a.keys():
            if 'teacher' in key:
                pretrained_teacher.pop(key)
        self.tiny.load_state_dict(pretrained_teacher)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = (self.small(x0) + self.tiny(x0)) / 2.0

        # x_t = self.tiny(x0)
        # x_s = self.small(x0)

        return x


class convnext_small(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(convnext_small, self).__init__()

        model = timm.create_model('convnext_small.fb_in1k', pretrained=True)

        self.model = model 

        self.model.head.fc     = nn.Sequential(nn.Linear(in_features=768, out_features=num_classes, bias=True))

        self.model.head.drop.p = 0.0

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages[3].parameters():
            param.requires_grad = True

        for param in self.model.head.parameters():
            param.requires_grad = True

        for param in self.model.parameters():
            param.requires_grad = False

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
    def __init__(self, num_classes=40, pretrained=True):
        super(convnext_tiny, self).__init__()

        self.model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True, features_only=True, out_indices=[0, 1, 2, 3])

        self.model.stem_0.stride = (2, 2)
        
        self.head  = timm.create_model('convnext_tiny.fb_in1k', pretrained=True).head
        
        self.head.fc = nn.Sequential(
                    nn.Dropout(p=0.5, inplace=True),
                    nn.Linear(in_features=768, out_features=num_classes, bias=True),
                )

        for param in self.model.parameters():
            param.requires_grad = False

        # for param in self.model.stages_2.parameters():
        #     param.requires_grad = True

        # for param in self.model.stages_3.parameters():
        #     param.requires_grad = True

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

        # x = self.head(x3)

        return x2, x3 # torch.softmax(x, dim=1)

# teacher = convnext_tiny().cuda()

class convnext_teacher(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(convnext_teacher, self).__init__()

        self.small = convnext_small()
        self.tiny  = convnext_tiny()

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = (self.small(x0) + self.tiny(x0)) / 2.0

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



# class efficientnet_teacher(nn.Module):
#     def __init__(self, num_classes=67, pretrained=True):
#         super(efficientnet_teacher, self).__init__()

#         self.b2 = B2()
#         self.b3 = B3()

#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, x0):
#         b, c, w, h = x0.shape
#         x = transform_test(x0)
#         x = (self.b2(x0) + self.b3(x0)) / 2.0

#         return x





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










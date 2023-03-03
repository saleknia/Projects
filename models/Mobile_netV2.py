import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b5
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
import random


class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2, self).__init__()

        # self.teacher = Mobile_netV2_teacher()
        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_teacher/Mobile_NetV2_FER2013_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # self.teacher.load_state_dict(pretrained_teacher)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model.features[0][0].stride = (1, 1)
        # self.features = model.features
        # self.avgpool = model.avgpool

        # self.segmentation = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights)
        # self.segmentation = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights)

        # for param in self.segmentation.parameters():
        #     param.requires_grad = False

        # model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.PAM = CrissCrossAttention(320)

        # model.features[0][0].stride = (1, 1)
        # model.features[0][0].in_channels = 4

        self.features = model.features
        self.features[0][0].stride = (1, 1)
        self.avgpool = model.avgpool

        # for param in self.features[0:8].parameters():
        #     param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=40, bias=True),
        )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512 , out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256 , out_features=40, bias=True),
        # )

    def forward(self, x):
        b, c, w, h = x.shape

        x = self.features[0:8](x)
        x = self.PAM(x)
        x = self.features[8](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.training:
            return x
        else:
            return torch.softmax(x, dim=1)

class Mobile_netV2_teacher(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(Mobile_netV2_teacher, self).__init__()

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

        return x4



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).to('cuda').repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.i = 0


    def forward(self, x):
        self.i = self.i + 1
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())

        if i%50 == 0:
            print(self.gamma)
        
        return self.gamma*(out_H + out_W) + x



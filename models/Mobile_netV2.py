import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b5
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
import random
from torch.nn import init

class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2, self).__init__()

        # self.teacher = Mobile_netV2_teacher()
        # loaded_data_teacher = torch.load('/content/drive/MyDrive/checkpoint_B3_91_85/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        # pretrained_teacher = loaded_data_teacher['net']
        # self.teacher.load_state_dict(pretrained_teacher)

        # for param in self.teacher.parameters():
        #     param.requires_grad = False

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        # model = efficientnet_b5(weights=EfficientNet_B5_Weights)

        # model.features[0][0].stride = (1, 1)
        # model.features[6][0].block[1][0].stride = (1, 1)

        # for param in model.features[0:5].parameters():
        #     param.requires_grad = False

        self.features = model.features
        self.avgpool = model.avgpool

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

    def forward(self, x0):
        b, c, w, h = x0.shape

        x = self.features(x0)

        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.training:
            return x
        else:
            return torch.softmax(x, dim=1)

class Mobile_netV2_teacher(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_teacher, self).__init__()

        model = efficientnet_b3(weights=EfficientNet_B3_Weights)

        # model = efficientnet_b3(weights=EfficientNet_B3_Weights)

        model.features[0][0].stride = (1, 1)

        for param in model.features[0:5].parameters():
            param.requires_grad = False

        self.features = model.features
        self.avgpool = model.avgpool

        # for param in self.features[0:8].parameters():
        #     param.requires_grad = False


        self.classifier = nn.Sequential(
            nn.Linear(in_features=1536, out_features=40, bias=True),
        )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512 , out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256 , out_features=40, bias=True),
        # )

    # def forward(self, x0):
    #     b, c, w, h = x0.shape

    #     x = self.features(x0)

    #     x = self.avgpool(x) 
        
    #     x = x.view(x.size(0), -1)

    #     x = self.classifier(x)

    #     if self.training:
    #         return x 
    #     else:
    #         return torch.softmax(x, dim=1)

    def forward(self, x0):
        b, c, w, h = x0.shape

        x1 = self.features[0:7](x0)
        x2 = self.features[7:9](x1)

        return x1, x2


class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,act_layer=nn.GELU,drop=0.1):
        super().__init__()
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self, x) :
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class WeightedPermuteMLP(nn.Module):
    def __init__(self,dim,seg_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.seg_dim=seg_dim

        self.mlp_c=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_h=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_w=nn.Linear(dim,dim,bias=qkv_bias)

        self.reweighting=MLP(dim,dim//4,dim*3)

        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)
    
    def forward(self,x) :
        B,H,W,C=x.shape

        c_embed=self.mlp_c(x)

        S=C//self.seg_dim
        h_embed=x.reshape(B,H,W,self.seg_dim,S).permute(0,3,2,1,4).reshape(B,self.seg_dim,W,H*S)
        h_embed=self.mlp_h(h_embed).reshape(B,self.seg_dim,W,H,S).permute(0,3,2,1,4).reshape(B,H,W,C)

        w_embed=x.reshape(B,H,W,self.seg_dim,S).permute(0,3,1,2,4).reshape(B,self.seg_dim,H,W*S)
        w_embed=self.mlp_w(w_embed).reshape(B,self.seg_dim,H,W,S).permute(0,2,3,1,4).reshape(B,H,W,C)

        weight=(c_embed+h_embed+w_embed).permute(0,3,1,2).flatten(2).mean(2)
        weight=self.reweighting(weight).reshape(B,C,3).permute(2,0,1).softmax(0).unsqueeze(2).unsqueeze(2)

        x=c_embed*weight[0]+w_embed*weight[1]+h_embed*weight[2]

        x=self.proj_drop(self.proj(x))

        return x






import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b5, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
import random
from torch.nn import init



class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2, self).__init__()

        # self.model = deit_base_distilled_patch16_224(pretrained=True)
        # self.model.head = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=768, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256, out_features=40, bias=True),
        # )
        model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # model = efficientnet_b6(weights=EfficientNet_B6_Weights)
        # self.patch_embed = model.patch_embed
        # self.transformers = nn.ModuleList(
        #     [model.blocks[i] for i in range(12)]
        # )
        # self.norm = model.norm
        # self.avgpool = model.fc_norm
        # self.classifier = nn.Linear(in_features=192, out_features=40, bias=True)
        self.features = model.features
        # self.features[0][0].stride = (1, 1)
        self.avgpool = model.avgpool
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=2304, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256, out_features=40, bias=True),
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )
        # self.aspp = ASPP(in_channels=1280)

        # model = resnet50(pretrained)

        # # take pretrained resnet, except AvgPool and FC
        # self.conv1 = model.conv1
        # self.conv1.stride = (1, 1)
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
        b, c, w, h = x.shape
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # emb = self.patch_embed(x)
        # for i in range(12):
        #     emb = self.transformers[i](emb)
        # emb = self.norm(emb)
        # print(emb.shape)
        # emb = self.avgpool(emb)
        # print(emb.shape)
        # emb = self.classifier(emb)
        # out, _ = self.model(x)
        
        if self.training:
            return x
        else:
            return torch.softmax(x, dim=1)



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

# class Mobile_netV2_teacher(nn.Module):
#     def __init__(self, num_classes=40, pretrained=True):
#         super(Mobile_netV2_teacher, self).__init__()

#         # model = efficientnet_b1(weights=EfficientNet_B1_Weights)

#         # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)

#         model = efficientnet_b3(weights=EfficientNet_B3_Weights)

#         # model.features[0][0].stride = (1, 1)

#         for param in model.features[0:5].parameters():
#             param.requires_grad = False

#         self.features = model.features
#         self.avgpool = model.avgpool

#         # for param in self.features[0:8].parameters():
#         #     param.requires_grad = False


#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=1536, out_features=40, bias=True),
#         )

#         # self.classifier = nn.Sequential(
#         #     nn.Dropout(p=0.4, inplace=True),
#         #     nn.Linear(in_features=1280, out_features=512, bias=True),
#         #     nn.Dropout(p=0.4, inplace=True),
#         #     nn.Linear(in_features=512 , out_features=256, bias=True),
#         #     nn.Dropout(p=0.4, inplace=True),
#         #     nn.Linear(in_features=256 , out_features=40, bias=True),
#         # )

#     # def forward(self, x0):
#     #     b, c, w, h = x0.shape

#     #     x = self.features(x0)

#     #     x = self.avgpool(x) 
        
#     #     x = x.view(x.size(0), -1)

#     #     x = self.classifier(x)

#     #     if self.training:
#     #         return x 
#     #     else:
#     #         return torch.softmax(x, dim=1)

#     def forward(self, x0):
#         b, c, w, h = x0.shape


#         x1 = self.features[0:7](x0)
#         x2 = self.features[7:8](x1)
#         # x3 = self.features[8:9](x2)

#         return x1, x2










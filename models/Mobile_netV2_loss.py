import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b4, EfficientNet_B4_Weights
import torchvision

class enet(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(enet, self).__init__()

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




# # class Mobile_netV2(nn.Module):
# #     def __init__(self, num_classes=40, pretrained=True):
# #         super(Mobile_netV2, self).__init__()

# #         model = efficientnet_b1(weights=EfficientNet_B1_Weights)
        
# #         self.features = model.features
# #         self.avgpool = model.avgpool
# #         self.classifier = nn.Sequential(
# #             nn.Dropout(p=0.4),
# #             nn.Linear(in_features=1280, out_features=512, bias=True),
# #             nn.ReLU(),
# #             nn.Dropout(p=0.4),
# #             nn.Linear(in_features=512, out_features=256, bias=True),
# #             nn.ReLU(),
# #             nn.Dropout(p=0.4),
# #             nn.Linear(in_features=256, out_features=40, bias=True),
# #         )

# #     def forward(self, x):
# #         x = self.features(x)
# #         x = self.avgpool(x)
# #         x = x.view(x.size(0), -1)
# #         x = self.classifier(x)
# #         return x


# # class Mobile_netV2_loss(nn.Module):
# #     def __init__(self, num_classes=40, pretrained=True):
# #         super(Mobile_netV2_loss, self).__init__()
# #         model_a = enet()
# #         loaded_data_a = torch.load('/content/drive/MyDrive/checkpoint_a/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
# #         pretrained_a = loaded_data_a['net']
# #         model_a.load_state_dict(pretrained_a)

# #         for param in model_a.parameters():
# #             param.requires_grad = False

# #         model_b = enet()
# #         loaded_data_b = torch.load('/content/drive/MyDrive/checkpoint_b/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
# #         pretrained_b = loaded_data_b['net']
# #         model_b.load_state_dict(pretrained_b)

# #         for param in model_b.parameters():
# #             param.requires_grad = False

# #         self.features_a = model_a.features
# #         self.features_b = model_b.features
# #         self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
# #         self.classifier = nn.Sequential(
# #             nn.Dropout(p=0.4, inplace=True),
# #             nn.Linear(in_features=2560, out_features=512, bias=True),
# #             nn.Dropout(p=0.4, inplace=True),
# #             nn.Linear(in_features=512, out_features=256, bias=True),
# #             nn.Dropout(p=0.4, inplace=True),
# #             nn.Linear(in_features=256, out_features=40, bias=True),
# #         )

# #     def forward(self, x):
# #         x_a = self.features_a(x)
# #         x_b = self.features_b(x)
# #         x = torch.cat([x_a, x_b], dim=1)
# #         x = self.avgpool(x)
# #         x = x.view(x.size(0), -1)
# #         x = self.classifier(x)
# #         return x


class Mobile_netV2_loss(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(Mobile_netV2_loss, self).__init__()
        model_a = enet()
        loaded_data_a = torch.load('/content/drive/MyDrive/checkpoint_a/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_a = loaded_data_a['net']
        model_a.load_state_dict(pretrained_a)

        for param in model_a.parameters():
            param.requires_grad = False

        model_b = enet()
        loaded_data_b = torch.load('/content/drive/MyDrive/checkpoint_b/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        pretrained_b = loaded_data_b['net']
        model_b.load_state_dict(pretrained_b)

        for param in model_b.parameters():
            param.requires_grad = False

        model = efficientnet_b0(weights=EfficientNet_B0_Weights)

        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=2, bias=True),
            nn.Softmax(dim=1),
        )

        # model_c = enet()
        # loaded_data_c = torch.load('/content/drive/MyDrive/checkpoint_c_c/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        # pretrained_c = loaded_data_c['net']
        # model_c.load_state_dict(pretrained_c)

        # for param in model_c.parameters():
        #     param.requires_grad = False

        # model_d = enet()
        # loaded_data_d = torch.load('/content/drive/MyDrive/checkpoint_d_c/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        # pretrained_d = loaded_data_d['net']
        # model_d.load_state_dict(pretrained_d)

        # for param in model_d.parameters():
        #     param.requires_grad = False

        # model_e = enet()
        # loaded_data_e = torch.load('/content/drive/MyDrive/checkpoint_e/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        # pretrained_e = loaded_data_e['net']
        # model_e.load_state_dict(pretrained_e)

        # for param in model_e.parameters():
        #     param.requires_grad = False

        # model_f = Mobile_netV2()
        # loaded_data_f = torch.load('/content/drive/MyDrive/checkpoint_a_1/Mobile_NetV2_Standford40_best.pth', map_location='cuda')
        # pretrained_f = loaded_data_f['net']
        # model_f.load_state_dict(pretrained_f)

        # for param in model_f.parameters():
        #     param.requires_grad = False

        self.model_a = model_a
        self.model_b = model_b
        # self.model_c = model_c
        # self.model_d = model_d
        # self.model_e = model_e
        # self.model_f = model_f

        # self.features_a = model_a.features
        # self.features_b = model_b.features

        # self.avgpool_a = nn.AdaptiveAvgPool2d(output_size=1)
        # self.avgpool_b = nn.AdaptiveAvgPool2d(output_size=1)

        # self.classifier_a = nn.Sequential(
        #     # nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=1280, out_features=640, bias=True),
        # )

        # self.classifier_b = nn.Sequential(
        #     # nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=1280, out_features=640, bias=True),
        # )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=1280, out_features=512, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.Dropout(p=0.4, inplace=True),
        #     nn.Linear(in_features=256, out_features=40, bias=True),
        # )

    def forward(self, x):
        # x_a = self.features_a(x)
        # x_a = self.avgpool_a(x_a)
        # x_a = x_a.view(x_a.size(0), -1)
        # x_a = self.classifier_a(x_a)

        # x_b = self.features_b(x)
        # x_b = self.avgpool_b(x_b)
        # x_b = x_b.view(x_b.size(0), -1)
        # x_b = self.classifier_b(x_b)

        # x = torch.cat([x_a, x_b], dim=1)
        # x = self.classifier(x)

        y = self.features(x)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        weights = self.classifier(y)

        x = ((weights[0].expand(32,40) * self.model_a(x)) + (weights[1].expand(32,40) * self.model_b(x)))
        return x







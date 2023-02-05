import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import einops
import timm
from torchvision import models as resnet_model
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple

class Cross(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()

        self.encoder = timm.create_model('hrnet_w18_small', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)

    def forward(self, x):
        x_input = x.float()
        B, C, H, W = x.shape
        x = self.encoder(x_input)
        return x




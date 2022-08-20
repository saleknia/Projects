from operator import index
import sys
import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import weakref
from functools import wraps
from medpy import metric
import medpy
import math
import random
import pickle
import warnings
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
warnings.filterwarnings("ignore")
from torch.autograd import Variable
from typing import Optional, Sequence
from torch import Tensor
from torch import nn
from torch.nn import functional as F

def pw_cosine(x,y):
    x = torch.nn.functional.normalize(x)
    y = torch.nn.functional.normalize(y)
    cosine = torch.mm(x, y.T)
    cosine = cosine.fill_diagonal_(0.0)
    cosine = torch.nn.functional.relu(cosine-0.25)
    cosine = torch.mean(cosine)
    return cosine

def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))


def importance_maps_distillation(student, teacher, exp=4):
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    loss = 0.0
    for s,t in zip(student, teacher):
        if s.shape[2] != t.shape[2]:
            s = F.interpolate(s, t.size()[-2:], mode='bilinear')
        loss = loss + torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()
    return loss


# class disparity(nn.Module):
#     def __init__(self):
#         super(disparity, self).__init__()
#         self.epsilon = 1e-6

#         # ENet
#         # self.down_scales = [1.0,0.5,0.25,0.125,0.125]

#         # ESPNet
#         self.down_scales = [1.0,0.5,0.5,0.25,0.125]


#         num_class = 11

#         self.num_class = num_class
        

#         # ENet
#         # self.proto_0 = torch.zeros(num_class, 11 )
#         # self.proto_1 = torch.zeros(num_class, 16 )
#         # self.proto_2 = torch.zeros(num_class, 64 )
#         # self.proto_3 = torch.zeros(num_class, 128)
#         # self.proto_4 = torch.zeros(num_class, 128)

#         # ESPNet
#         self.proto_0 = torch.zeros(num_class, 11 )
#         self.proto_1 = torch.zeros(num_class, 16 )
#         self.proto_2 = torch.zeros(num_class, 11 )
#         self.proto_3 = torch.zeros(num_class, 64 )
#         self.proto_4 = torch.zeros(num_class, 128)

#         # SUNet
#         # self.proto_0 = torch.zeros(num_class, self.num_class +1)
#         # self.proto_1 = torch.zeros(num_class, 8 )
#         # self.proto_2 = torch.zeros(num_class, 16)
#         # self.proto_3 = torch.zeros(num_class, 32)
#         # self.proto_4 = torch.zeros(num_class, 64)


#         # Mobile_NetV2
#         # self.proto_0 = torch.zeros(num_class, 9  )
#         # self.proto_1 = torch.zeros(num_class, 9  )
#         # self.proto_2 = torch.zeros(num_class, 320)
#         # self.proto_3 = torch.zeros(num_class, 96 )
#         # self.proto_4 = torch.zeros(num_class, 64 )

#         # ResNet_18
#         # self.proto_0 = torch.zeros(num_class, num_class+1)
#         # self.proto_1 = torch.zeros(num_class, 64 )
#         # self.proto_2 = torch.zeros(num_class, 128)
#         # self.proto_3 = torch.zeros(num_class, 256)
#         # self.proto_4 = torch.zeros(num_class, 512)


#         # DABNet
#         # self.proto_1 = torch.zeros(num_class, 9  )
#         # self.proto_2 = torch.zeros(num_class, 64 )
#         # self.proto_3 = torch.zeros(num_class, 128)
#         # self.proto_4 = torch.zeros(num_class, 9  )

#         # self.proto_0 = torch.zeros(num_class, num_class+1)
#         # self.proto_1 = torch.zeros(num_class, 32 )
#         # self.proto_2 = torch.zeros(num_class, 32 )
#         # self.proto_3 = torch.zeros(num_class, 64 )
#         # self.proto_4 = torch.zeros(num_class, 128)

#         # self.protos = torch.load('/content/UNet_V2/protos_file.pth')
#         # self.protos = [self.proto_1, self.proto_2, self.proto_3, self.proto_4]

#         self.protos = [self.proto_0,self.proto_1, self.proto_2, self.proto_3, self.proto_4]
#         self.momentum = torch.tensor(0.0)
#         self.iteration = 0
#         # self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
#         self.momentum_schedule = cosine_scheduler(0.85, 1.0, 60.0, 61)
#         self.pixel_wise = CriterionPixelWise()


#     def forward(self, masks, t_masks, up4, up3, up2, up1, outputs):
#         loss = 0.0
#         # loss = loss + self.pixel_wise(preds_S=outputs, masks=masks)
#         up = [outputs, up1, up2, up3, up4]

#         # print(outputs.shape)
#         # print(up1.shape)
#         # print(up2.shape)
#         # print(up3.shape)
#         # print(up4.shape)

#         for k in range(5):
#             indexs = []
#             WP = []
#             B,C,H,W = up[k].shape
            
#             temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks = temp_masks.squeeze(dim=1)

#             temp_t_masks = nn.functional.interpolate(t_masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_t_masks = temp_t_masks.squeeze(dim=1)

#             mask_unique_value = torch.unique(temp_masks)
#             mask_unique_value = mask_unique_value[0:-1]
#             unique_num = len(mask_unique_value)
            
#             if unique_num<2:
#                 return 0

#             prototypes = torch.zeros(size=(unique_num,C))

#             for count,p in enumerate(mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])

#                 bin_mask_t = torch.tensor(temp_t_masks==p,dtype=torch.int8)
#                 bin_mask_t = bin_mask_t.unsqueeze(dim=1).expand_as(up[k])

#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         batch_counter = batch_counter + 1
#                 temp = temp / batch_counter
#                 prototypes[count] = temp
#                 # WP.append(torch.sum(bin_mask_t)/torch.sum(bin_mask))

#             # WP = torch.tensor(WP)
#             # WP = torch.diag(WP)
#             # WP = WP.detach()

#             indexs = [x.item()-1 for x in mask_unique_value]
#             indexs.sort()

#             l = 0.0
#             proto = self.protos[k][indexs].unsqueeze(dim=0)
#             prototypes = prototypes.unsqueeze(dim=0)
#             distances_c = torch.cdist(proto.clone().detach(), prototypes, p=2.0)
#             proto = self.protos[k][indexs].squeeze(dim=0)
#             prototypes = prototypes.squeeze(dim=0)
#             x = (torch.eye(distances_c[0].shape[0],distances_c[0].shape[1]))
#             diagonal = distances_c[0] * x

#             # weights = 1.0 / distances_c.clamp(min=self.epsilon)
#             # weights = weights / weights.max()
#             # weights = weights.detach()


#             proto = prototypes.unsqueeze(dim=0)
#             distances = torch.cdist(proto.clone().detach(), proto, p=2.0)
#             l = l + (1.0 / torch.mean(distances))

#             l = l + (1.0 / torch.mean((distances_c[0]-diagonal)))
#             l = l + (1.0 * torch.mean(diagonal))

#             # l = l + 0.5 * cosine_loss


#             loss = loss + l

#             self.update(prototypes, mask_unique_value, k)
#         self.iteration = self.iteration + 1

#         return loss


#     @torch.no_grad()
#     def update(self, prototypes, mask_unique_value, k):
#         for count, p in enumerate(mask_unique_value):
#             p = p.long().item()
#             self.momentum = self.momentum_schedule[self.iteration] 
#             self.protos[k][p] = self.protos[k][p] * self.momentum + prototypes[count] * (1 - self.momentum)

class disparity(nn.Module):
    def __init__(self):
        super(disparity, self).__init__()

        # ENet
        self.down_scales = [0.5,0.25,0.125,0.125]



        num_class = 10

        self.num_class = num_class
        

        # ENet
        self.proto_0 = torch.zeros(num_class, 11 )
        self.proto_1 = torch.zeros(num_class, 16 )
        self.proto_2 = torch.zeros(num_class, 64 )
        self.proto_3 = torch.zeros(num_class, 128)
        self.proto_4 = torch.zeros(num_class, 128)

        self.protos = [self.proto_0, self.proto_1, self.proto_2, self.proto_3, self.proto_4]
        self.momentum = torch.tensor(0.0)
        self.iteration = 0
        self.momentum_schedule = cosine_scheduler(0.85, 1.0, 60.0, 396)
        # self.momentum_schedule = cosine_scheduler(0.85, 1.0, 60.0, 368)

    def dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, masks, t_masks, up4, up3, up2, up1, outputs):
        loss = 0.0
        up = [outputs, up1, up2, up3, up4]

        for k in range(5):
            indexs = []
            WP = []
            B,C,H,W = up[k].shape
            
            temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
            temp_masks = temp_masks.squeeze(dim=1)

            temp_t_masks = nn.functional.interpolate(t_masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
            temp_t_masks = temp_t_masks.squeeze(dim=1)

            mask_unique_value = torch.unique(temp_masks)
            mask_unique_value = mask_unique_value[1:]
            unique_num = len(mask_unique_value)
            
            if unique_num<2:
                return 0

            prototypes = torch.zeros(size=(unique_num,C))

            for count,p in enumerate(mask_unique_value):
                p = p.long()
                bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
                bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])

                bin_mask_t = torch.tensor(temp_t_masks==p,dtype=torch.int8)
                bin_mask_t = bin_mask_t.unsqueeze(dim=1).expand_as(up[k])

                temp = 0.0
                batch_counter = 0
                for t in range(B):
                    if torch.sum(bin_mask[t])!=0:
                        v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
                        temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
                        batch_counter = batch_counter + 1
                temp = temp / batch_counter
                # wp = torch.sum(bin_mask_t)/torch.sum(bin_mask)
                # wp = self.dice_loss(bin_mask_t,bin_mask) 
                prototypes[count] = temp
            #     WP.append(wp)

            # WP = torch.tensor(WP)
            # WP = torch.diag(WP)
            # WP = WP.detach()


            indexs = [x.item()-1 for x in mask_unique_value]
            indexs.sort()

            l = 0.0
            proto = self.protos[k][indexs].unsqueeze(dim=0)
            prototypes = prototypes.unsqueeze(dim=0)
            distances_c = torch.cdist(proto.clone().detach(), prototypes, p=2.0)
            proto = self.protos[k][indexs].squeeze(dim=0)
            prototypes = prototypes.squeeze(dim=0)
            x = (torch.eye(distances_c[0].shape[0],distances_c[0].shape[1]))
            diagonal = distances_c[0] * x

            l = l + (1.0 / torch.mean((distances_c[0]-diagonal)))
            l = l + (1.0 * torch.mean(diagonal))

            loss = loss + l

            self.update(prototypes, mask_unique_value, k)
        self.iteration = self.iteration + 1

        return loss

    # @torch.no_grad()
    # def update(self, prototypes_t, mask_unique_value, k):
    #     for count, p in enumerate(mask_unique_value):
    #         p = p.long().item()
    #         self.protos[k][p-1] = self.protos[k][p-1] + prototypes_t[count]
    #         self.protos[k][p-1] = nn.functional.normalize(self.protos[k][p-1], p=2.0, dim=0, eps=1e-12, out=None)

    @torch.no_grad()
    def update(self, prototypes, mask_unique_value, k):
        for count, p in enumerate(mask_unique_value):
            p = p.long().item()
            self.momentum = self.momentum_schedule[self.iteration] 
            self.protos[k][p-1] = self.protos[k][p-1] * self.momentum + prototypes[count] * (1 - self.momentum)

# class disparity(nn.Module):
#     def __init__(self):
#         super(disparity, self).__init__()
#         # self.down_scales = [0.5, 0.5, 0.25, 0.125]
#         # self.down_scales = [1.0,0.5,0.25,0.125,0.125]
#         # self.down_scales = [1.0, 0.5, 0.25, 0.125, 0.125]
#         # self.down_scales = [1.0, 0.25, 0.125, 0.0625, 0.03125]
#         self.down_scales = [1.0, 1.0, 0.5, 0.25, 0.125]



#     def forward(self, masks, t_masks, outputs, up4, up3, up2, up1):
#         loss = 0.0
#         up = [outputs, up1, up2, up3, up4]
#         # up = [up1, up2, up3, up4]


#         for k in range(5):
#             B,C,H,W = up[k].shape
            
#             temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks = temp_masks.squeeze(dim=1)

#             temp_masks_t = nn.functional.interpolate(t_masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks_t = temp_masks_t.squeeze(dim=1)

#             mask_unique_value = torch.unique(temp_masks)
#             mask_unique_value = mask_unique_value[1:]
#             unique_num = len(mask_unique_value)

#             if unique_num<2:
#                 return 0

#             prototypes = torch.zeros(size=(unique_num,C),device='cuda')

#             for count,p in enumerate(mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])

#                 bin_mask_t = torch.tensor(temp_masks_t==p,dtype=torch.int8)
#                 bin_mask_t = bin_mask_t.unsqueeze(dim=1).expand_as(up[k])

#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         batch_counter = batch_counter + 1
#                 temp = temp / batch_counter
#                 prototypes[count] = temp

#             l = 0.0

#             distances = torch.cdist(prototypes.detach().clone(), prototypes, p=2.0)
#             # distances_t = torch.cdist(prototypes_t, prototypes, p=2.0)
#             # diagonal = distances_t * (torch.eye(distances_t.shape[0],distances_t.shape[1],device='cuda'))

#             # l = pw_cosine(prototypes, prototypes)

#             # l = l + torch.mean(diagonal)
#             l = 1.0 / (torch.mean(distances)) 

#             loss = loss + l

#         return loss



class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cuda',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.
    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl

SEED = 666

class color():
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

palate=[
        (0,0,0),
        (51,153,255),
        (255,0,0),
]
palate = np.array(palate,dtype=np.float32)/255.0

labels=['Background','Lung','Infection']

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    # bar = '█' * filled_length + ' ' * (bar_length - filled_length)
    bar = '■' * filled_length + '□' * (bar_length - filled_length)

    sys.stdout.write('\r%s |\033[34m%s\033[0m| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def hd95(masks, preds, num_class):
    NaN = np.nan
    masks = masks.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    metric_list = []
    for i in range(1,num_class):
        if np.sum(masks==i)>0 and np.sum(preds==i)>0:
            metric = medpy.metric.binary.hd95(result=(preds==i), reference=(masks==i))
        elif np.sum(masks==i)==0 and np.sum(preds==i)==0:
            metric = NaN
        else:
            metric = 0.0
        metric_list.append(metric)
    metric_list = np.array(metric_list)
    result = np.nanmean(metric_list)
    return result


class Save_Checkpoint(object):
    def __init__(self,filename,current_num_epoch,last_num_epoch,initial_best_acc,initial_best_epoch=1):
        # self.net=net
        # self.optimizer=optimizer
        # self.lr_scheduler=lr_scheduler
        self.best_acc = initial_best_acc
        self.best_epoch = 1
        self.initial_best_epoch = initial_best_epoch
        self.last_num_epoch = last_num_epoch
        self.current_num_epoch = current_num_epoch
        self.folder = 'checkpoint'
        self.filename=filename
        # self.best_path = os.path.join(os.path.abspath(self.folder), self.filename + '_best.pth')
        # self.last_path = os.path.join(os.path.abspath(self.folder), self.filename + '_last.pth')
        self.best_path = '/content/drive/MyDrive/checkpoint/' + self.filename + '_best.pth'
        self.last_path = '/content/drive/MyDrive/checkpoint/' + self.filename + '_last.pth'
        os.makedirs(self.folder, exist_ok=True)

    def save_best(self, acc, acc_per_class, epoch, net, optimizer, lr_scheduler):
        if self.best_acc < acc:
            print(color.BOLD+color.RED+'Saving best checkpoint...'+color.END)
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'acc': acc,
                'acc_per_class':acc_per_class,
                'best_epoch': epoch
            }
            self.best_epoch = epoch
            self.best_acc = acc
            torch.save(state, self.best_path)

    def save_last(self, acc, acc_per_class, epoch, net, optimizer, lr_scheduler):
        print(color.BOLD+color.RED+'Saving last checkpoint...'+color.END)
        if self.best_epoch==1:
            self.best_epoch = self.initial_best_epoch
        self.current_num_epoch = self.current_num_epoch + 1
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': lr_scheduler.state_dict(),
            'acc': acc,
            'best_acc':self.best_acc,
            'acc_per_class':acc_per_class,
            'best_epoch':self.best_epoch,
            'num_epoch': self.last_num_epoch + self.current_num_epoch
        }

        numpy_state = np.random.get_state()
        with open('/content/drive/MyDrive/checkpoint/numpy_state.pickle', 'wb') as f:
            pickle.dump(numpy_state, f)

        random_state = random.getstate()
        with open('/content/drive/MyDrive/checkpoint/random_state.pickle', 'wb') as f:
            pickle.dump(random_state, f)

        torch_state = torch.get_rng_state()
        torch.save(torch_state, '/content/drive/MyDrive/checkpoint/torch_state.pth')

        cuda_state = torch.cuda.get_rng_state()
        torch.save(cuda_state, '/content/drive/MyDrive/checkpoint/cuda_state.pth')
        torch.save(state, self.last_path)

    def early_stopping(self, epoch):
        if self.best_epoch==1:
            return epoch-self.initial_best_epoch
        else:
            return epoch-self.best_epoch
        
    def best_accuracy(self):
        return self.best_acc

               
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Evaluator(object):
    ''' For using this evaluator target and prediction
        dims should be [B,H,W] '''
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        Acc = torch.tensor(Acc)
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        Acc = torch.tensor(Acc)
        return Acc

    def Mean_Intersection_over_Union(self,per_class=False,show=False):
        numerator = np.diag(self.confusion_matrix) 
        denominator = (np.sum(self.confusion_matrix,axis=1) + np.sum(self.confusion_matrix, axis=0)-np.diag(self.confusion_matrix))
        if show:
            # print('Intersection Pixels: ',numerator)
            # print('Union Pixels: ',denominator)
            print('MIoU Per Class: ',numerator/denominator)
        class_MIoU = numerator/denominator
        class_MIoU = class_MIoU[1:]
        MIoU = np.nanmean(class_MIoU)
        MIoU = torch.tensor(MIoU)
        class_MIoU = torch.tensor(class_MIoU)
        if per_class:
            return MIoU,class_MIoU
        else:
            return MIoU

    def Dice(self,per_class=False,show=False):
        numerator = 2*np.diag(self.confusion_matrix) 
        denominator = (np.sum(self.confusion_matrix,axis=1) + np.sum(self.confusion_matrix, axis=0))
        if show:
            # print('Intersection Pixels: ',numerator)
            # print('Union Pixels: ',denominator)
            print('Dice Per Class: ',numerator/denominator)
        class_Dice = numerator/denominator
        class_Dice = class_Dice[1:]
        Dice = np.nanmean(class_Dice)
        Dice = torch.tensor(Dice)
        class_Dice = torch.tensor(class_Dice)
        if per_class:
            return Dice,class_Dice
        else:
            return Dice

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        FWIoU = torch.tensor(FWIoU)
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        gt_image=gt_image.int().detach().cpu().numpy()
        pre_image=pre_image.int().detach().cpu().numpy()
        assert gt_image.shape == pre_image.shape
        for lp, lt in zip(pre_image, gt_image):
            self.confusion_matrix += self._generate_matrix(lt.flatten(), lp.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


# class DiceLoss(nn.Module):
    
#     ''' For this loss function you should consider
#         background as a one specific class '''

#     ''' For this loss function score and target dims
#         should be [B, #classes, H, W] and [B, H, W] respectively '''

#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes

#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def _dice_loss(self, score, target):
#         ''' 
#         According to SAD paper notation
#         No ---> Number of overlapping pixels between predition and target.
#         Np ---> Number of predicted pixels.
#         Ng ---> Number of ground-truth pixels.
#         '''
#         target = target.float()
#         intersect = torch.sum(score * target)  # No 
#         y_sum = torch.sum(target * target)  # Ng
#         z_sum = torch.sum(score * score)  # Np
#         num = intersect 
#         den = y_sum + z_sum
#         w = 1.0 / (torch.sum(target * target))
#         num = num * w
#         den = den * w
#         return num, den

#     def forward(self, inputs, target, weight=None, softmax=False):
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#         assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#             # loss += dice * (1.0 / (torch.sum(target[:, i] * target[:, i]) + 1))
#         return loss / self.n_classes

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

# class atten_loss(nn.Module):

#     def __init__(self):
#         super(atten_loss, self).__init__()

#     def attention(self, features):
#         features = torch.abs(input=features)**2
#         atten = torch.sum(input=features,dim=1)
#         return atten
#     def flat(self, atten):
#         vectors = torch.flatten(input=atten,start_dim=1)
#         return vectors
#     def vector_norm(self, vectors):
#         for i in range(vectors.shape[0]):
#             norm = torch.linalg.norm(input=vectors[i].clone())
#             if norm!=0.0:
#                 vectors[i] = torch.div(input=vectors[i].clone(),other=norm)
#         return vectors

#     def attention_loss(self, features_1, features_2, scale_factor=2,mask=False):
#         if mask:
#             atten_features_1 = features_1.float()
#         else:
#             atten_features_1 = self.attention(features_1) 
#         atten_features_2 = self.attention(features_2)

#         if scale_factor==2:
#             atten_features_2 = torch.unsqueeze(input=atten_features_2,dim=1)
#             atten_features_2 = torch.nn.functional.interpolate(input=atten_features_2,scale_factor=scale_factor,mode='bilinear',align_corners=True)
#             atten_features_2 = torch.squeeze(input=atten_features_2,dim=1)
        
#         atten_features_1 = self.flat(atten_features_1) 
#         atten_features_2 = self.flat(atten_features_2)
#         atten_features_1 = self.vector_norm(atten_features_1) 
#         atten_features_2 = self.vector_norm(atten_features_2)

#         loss = 0.0
#         for i in range(atten_features_1.shape[0]):
#             loss = loss + torch.linalg.norm(atten_features_1[i]-atten_features_2[i])
#         return loss

#     def forward(self, masks, x1, x2, x3, x4):
#         loss = 0.0
#         loss = loss + 0.25 * self.attention_loss(features_1=masks,features_2=x1,mask=True,scale_factor=1)
#         loss = loss + self.attention_loss(features_1=x1,features_2=x2)
#         loss = loss + self.attention_loss(features_1=x2,features_2=x3)
#         return loss

# class prototype_loss(nn.Module):
#     def __init__(self):
#         super(prototype_loss, self).__init__()
#         self.cosine = torch.nn.CosineSimilarity(dim=0)
#         self.distance = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
#         self.down_scales = [1.0,0.5,0.25]
#     def forward(self, masks, up3, up2, up1):
#         loss = 0.0
#         up = [up1, up2, up3]

#         # mask_unique_value = torch.unique(masks)
#         # unique_num = len(mask_unique_value)

#         for k in range(3):
#             B,C,H,W = up[k].shape
            
#             temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks = temp_masks.squeeze(dim=1)

#             mask_unique_value = torch.unique(temp_masks)
#             mask_unique_value = mask_unique_value[1:]
#             unique_num = len(mask_unique_value)
            
#             if unique_num<2:
#                 return 0

#             prototypes = torch.zeros(size=(unique_num,C))
#             # similarities = torch.zeros(size=(unique_num,unique_num))

#             for count,p in enumerate(mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
#                 # temp = nn.functional.avg_pool2d(bin_mask*up[k], kernel_size=(H,W))
#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         batch_counter = batch_counter + 1
#                 # temp = temp.reshape(B,C)
#                 # prototypes[count] = torch.mean(temp, dim=0)
#                 temp = temp / batch_counter
#                 prototypes[count] = temp

#             l = 0.0
#             prototypes = prototypes.unsqueeze(dim=0)
#             distances = torch.cdist(prototypes, prototypes, p=2.0)
#             # similarities = 1.0 / distances
#             l = 1.0 / torch.mean(distances)

#             loss = loss + l

#         return loss


# class prototype_loss(nn.Module):
#     def __init__(self):
#         super(prototype_loss, self).__init__()
#         self.cosine = torch.nn.CosineSimilarity(dim=0)
#         self.distance = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
#         self.down_scales = [1.0,0.5,0.25]
#     def forward(self, masks, up3, up2, up1):
#         loss = 0.0
#         up = [up1, up2, up3]

#         # mask_unique_value = torch.unique(masks)
#         # unique_num = len(mask_unique_value)

#         for k in range(3):
#             B,C,H,W = up[k].shape
            
#             temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks = temp_masks.squeeze(dim=1)

#             mask_unique_value = torch.unique(temp_masks)
#             mask_unique_value = mask_unique_value[1:]
#             unique_num = len(mask_unique_value)
            
#             if unique_num<2:
#                 return 0

#             prototypes = torch.zeros(size=(unique_num,C))
#             similarities = torch.zeros(size=(unique_num,unique_num))

#             for count,p in enumerate(mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
#                 # temp = nn.functional.avg_pool2d(bin_mask*up[k], kernel_size=(H,W))
#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         batch_counter = batch_counter + 1
#                 # temp = temp.reshape(B,C)
#                 # prototypes[count] = torch.mean(temp, dim=0)
#                 temp = temp / batch_counter
#                 prototypes[count] = temp

#             l = 0.0
#             # prototypes = prototypes.unsqueeze(dim=0)
#             for i in range(unique_num):
#                 for j in range(unique_num):
#                     if i<j :
#                         similarities[i,j] = self.cosine(prototypes[i],prototypes[j])**2
#             # distances = torch.cdist(prototypes, prototypes, p=2.0)
#             # # similarities = 1.0 / distances
#             # l = 1.0 / torch.mean(distances)
#             l = torch.mean(similarities)
#             loss = loss + l

#         return loss


#####################################################################################################
#####################################################################################################
#####################################################################################################


# class prototype_loss_mean(nn.Module):
#     def __init__(self, num_class=9):
#         super(prototype_loss_mean, self).__init__()
#         self.num_class = num_class - 1
#         self.distance = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
#         self.down_scales = [1.0,0.5,0.25,0.125]
#         # self.proto_1 = torch.zeros(num_class,64,device='cuda',requires_grad=False)
#         # self.proto_2 = torch.zeros(num_class,128,device='cuda',requires_grad=False)
#         # self.proto_3 = torch.zeros(num_class,256,device='cuda',requires_grad=False)
#         # self.proto_4 = torch.zeros(num_class,512,device='cuda',requires_grad=False)
#         # self.proto_1 = torch.zeros(num_class,64 ,device='cuda',requires_grad=False)
#         # self.proto_2 = torch.zeros(num_class,64 ,device='cuda',requires_grad=False)
#         # self.proto_3 = torch.zeros(num_class,128,device='cuda',requires_grad=False)
#         # self.proto_4 = torch.zeros(num_class,256,device='cuda',requires_grad=False)
#         # self.protos = [self.proto_1, self.proto_2, self.proto_3, self.proto_4]
#         # self.momentum = torch.tensor(0.9,device='cuda',requires_grad=False)
#         # self.iteration = 0
#         # self.max_iteration = 368 * 30

#     def forward(self, masks, up4, up3, up2, up1):
#         loss = 0.0
#         # self.iteration = self.iteration + 1
#         up = [up1, up2, up3, up4]

#         for k in range(4):
#             B,C,H,W = up[k].shape
            
#             temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks = temp_masks.squeeze(dim=1)

#             mask_unique_value = torch.unique(temp_masks)
#             mask_unique_value = mask_unique_value[1:]
#             unique_num = len(mask_unique_value)
            
#             if unique_num<2:
#                 return 0

#             prototypes = torch.zeros(size=(unique_num,C),device='cuda')

#             for count,p in enumerate(mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         batch_counter = batch_counter + 1
#                 temp = temp / batch_counter
#                 prototypes[count] = temp
            
#             l = 0.0
#             indexs = [x.item()-1 for x in mask_unique_value]

#             # self.update(prototypes, mask_unique_value, k)

#             proto = self.protos[k][indexs].unsqueeze(dim=0)
#             prototypes = prototypes.unsqueeze(dim=0)
#             distances_c = torch.cdist(proto.to('cuda'), prototypes.to('cuda'), p=2.0)
#             distances = torch.cdist(prototypes.to('cuda'), prototypes.to('cuda'), p=2.0)
#             proto = self.protos[k][indexs].squeeze(dim=0)
#             prototypes = prototypes.squeeze(dim=0)

#             diagonal = distances_c[0] * (torch.eye(distances_c[0].shape[0],distances_c[0].shape[1],device='cuda'))
#             l = 1.0 / (torch.mean(distances)) #+ torch.mean(diagonal)
#             loss = loss + l
#             # self.update(prototypes, mask_unique_value, k)

#         return loss

#     @torch.no_grad()
#     def update(self, prototypes, mask_unique_value, k):
#         for count, p in enumerate(mask_unique_value):
#             p = p.long().item()
#             # self.momentum = self.iteration / self.max_iteration
#             self.protos[k][p-1] = self.protos[k][p-1] * self.momentum + prototypes[count] * (1 - self.momentum)
#             # self.protos[k][p-1] = nn.functional.normalize(self.protos[k][p-1], p=2.0, dim=0, eps=1e-12, out=None)

# class prototype_loss_mean(nn.Module):
#     def __init__(self):
#         super(prototype_loss_mean, self).__init__()
#         self.cosine = torch.nn.CosineSimilarity(dim=0)
#         self.distance = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
#         self.proto = torch.zeros(8, 64, device='cuda', requires_grad=False)
#         self.momentum = torch.tensor(0.1,device='cuda',requires_grad=False)
#     @torch.no_grad()
#     def update(self, prototypes, mask_unique_value):
#         prototypes = prototypes.to('cuda')
#         mask_unique_value = mask_unique_value.to('cuda')
#         for count, p in enumerate(mask_unique_value):
#             p = p.long().item()
#             self.proto[p-1] = self.proto[p-1] * self.momentum + prototypes[count] * (1 - self.momentum)


#     def forward(self, masks, outputs):
#         loss = 0.0
        
#         temp_masks = masks
#         B,C,H,W = outputs.shape

#         mask_unique_value = torch.unique(temp_masks)
#         mask_unique_value = mask_unique_value[1:]
#         unique_num = len(mask_unique_value)
        
#         if unique_num<2:
#             return 0

#         prototypes = torch.zeros(size=(unique_num,C))

#         for count,p in enumerate(mask_unique_value):
#             p = p.long()
#             bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#             bin_mask = bin_mask.unsqueeze(dim=1).expand_as(outputs)
#             temp = 0.0
#             batch_counter = 0
#             for t in range(B):
#                 if torch.sum(bin_mask[t])!=0:
#                     v = torch.sum(bin_mask[t]*outputs[t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                     temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                     batch_counter = batch_counter + 1
#             temp = temp / batch_counter
#             prototypes[count] = temp
#         indexs = [x.item()-1 for x in mask_unique_value]
        
#         # self.update(prototypes, mask_unique_value)

#         proto = self.proto[indexs].unsqueeze(dim=0)
#         distances = torch.cdist(proto.to('cuda'), prototypes.to('cuda'), p=2.0)

#         # distances = torch.cdist(proto.to('cuda'), proto.to('cuda'), p=2.0)

#         diagonal = distances[0] * (torch.eye(distances[0].shape[0],distances[0].shape[1],device='cuda'))
#         loss = 1.0 / (torch.mean(distances[0]-diagonal)) + torch.mean(diagonal)

#         # loss = 1.0 / (torch.mean(distances))

#         # loss = 1.0 / (torch.mean(distances[0]-diagonal))

#         self.update(prototypes, mask_unique_value)

#         return loss


#####################################################################################################
#####################################################################################################
#####################################################################################################


# class prototype_loss_mean(nn.Module):
#     def __init__(self):
#         super(prototype_loss_mean, self).__init__()

#     def forward(self, masks, outputs):
#         loss = 0.0
        
#         temp_masks = masks
#         B,C,H,W = outputs.shape

#         mask_unique_value = torch.unique(temp_masks)
#         mask_unique_value = mask_unique_value[1:]
#         unique_num = len(mask_unique_value)
        
#         if unique_num<2:
#             return 0
#         var = 0
#         for count,p in enumerate(mask_unique_value):
#             p = p.long()
#             bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#             bin_mask = bin_mask.unsqueeze(dim=1).expand_as(outputs)
#             temp = 0.0
#             batch_counter = 0
#             for t in range(B):
#                 if torch.sum(bin_mask[t])!=0:
#                     v = bin_mask[t]*outputs[t]
#                     v = v.sum(dim=0)
#                     v_shape_0, v_shape_1 = v.shape
#                     v = torch.nn.functional.normalize(v.reshape(-1), dim=0, p=2.0, eps=1e-12, out=None)
#                     v = v.reshape(v_shape_0, v_shape_1)
#                     temp = temp + torch.var(v)
#                     batch_counter = batch_counter + 1
#             temp = temp / batch_counter
#             var = var + temp
#         loss = var / unique_num
#         return loss


# class prototype_loss(nn.Module):
#     def __init__(self):
#         super(prototype_loss, self).__init__()
#         num_class = 12
#         self.cosine = torch.nn.CosineSimilarity(dim=0)
#         self.distance = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
#         self.down_scales = [1.0,0.5,0.25,0.125]
#         self.proto_1 = torch.zeros(num_class,64,device='cuda',requires_grad=False)
#         self.proto_2 = torch.zeros(num_class,128,device='cuda',requires_grad=False)
#         self.proto_3 = torch.zeros(num_class,256,device='cuda',requires_grad=False)
#         self.proto_4 = torch.zeros(num_class,512,device='cuda',requires_grad=False)
#         self.protos = [self.proto_1, self.proto_2, self.proto_3, self.proto_4]
#         self.momentum = torch.tensor(0.95,device='cuda',requires_grad=False)
#         self.iteration = 0
#         self.max_iteration = 30 * 368
#     def forward(self, masks, up4, up3, up2, up1):
#         self.iteration = self.iteration + 1
#         loss = 0.0
#         up = [up1, up2, up3, up4]

#         # mask_unique_value = torch.unique(masks)
#         # unique_num = len(mask_unique_value)

#         for k in range(4):
#             B,C,H,W = up[k].shape
            
#             temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks = temp_masks.squeeze(dim=1)

#             mask_unique_value = torch.unique(temp_masks)
#             mask_unique_value = mask_unique_value[1:]
#             unique_num = len(mask_unique_value)
            
#             if unique_num<2:
#                 return 0

#             prototypes = torch.zeros(size=(unique_num,C),device='cuda')
#             similarities = torch.zeros(size=(unique_num,unique_num))

#             for count,p in enumerate(mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
#                 # temp = nn.functional.avg_pool2d(bin_mask*up[k], kernel_size=(H,W))
#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         batch_counter = batch_counter + 1
#                 # temp = temp.reshape(B,C)
#                 # prototypes[count] = torch.mean(temp, dim=0)
#                 temp = temp / batch_counter
#                 prototypes[count] = temp
            
#             # self.update(prototypes, mask_unique_value, k)

#             # indexs = [x.item()-1 for x in mask_unique_value]

#             l = 0.0
#             prototypes = prototypes.unsqueeze(dim=0)
#             # c_prototypes = self.protos[k][indexs].unsqueeze(dim=0)
#             distances = torch.cdist(prototypes, prototypes, p=2.0)


#             # prototypes = prototypes.squeeze(dim=0)
#             # c_prototypes = self.protos[k][indexs].squeeze(dim=0)

#             # Z = prototypes
#             # B = c_prototypes.T
#             # Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True).to('cuda')  # Size (n, 1).
#             # B_norm = torch.linalg.norm(B, dim=0, keepdim=True).to('cuda')  # Size (1, b).

#             # Distance matrix of size (b, n).
#             # cosine_similarity = ((Z @ B) / (Z_norm @ B_norm))
#             # c_distances = 1 - torch.diag(cosine_similarity)**2 

#             # c_distances = torch.cdist(prototypes, c_prototypes, p=2.0)
#             # c_distances = torch.diag(c_distances)
#             # similarities = 1.0 / distances
#             # l = (1.0 / torch.mean(distances)) + torch.mean(c_distances) + l

#             l = (-1 * torch.mean(distances)) + l


#             # for i in range(unique_num):
#             #     for j in range(unique_num):
#             #         if i<j:
#             #             similarities[i,j] = self.cosine(prototypes[i], prototypes[j])**2

#             # Z = prototypes
#             # B = prototypes.T
#             # Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True).to('cuda')  # Size (n, 1).
#             # B_norm = torch.linalg.norm(B, dim=0, keepdim=True).to('cuda')  # Size (1, b).

#             # Distance matrix of size (b, n).
#             # cosine_similarity = ((Z @ B) / (Z_norm @ B_norm))
#             # cosine_similarity = cosine_similarity**2 - torch.eye(cosine_similarity.shape[0]).to('cuda')
#             # l = torch.sum(cosine_similarity) * 0.5      

#             # l = torch.sum(similarities)

#             loss = loss + l

#         return loss

#     @torch.no_grad()
#     def update(self, prototypes, mask_unique_value, k):
#         for count, p in enumerate(mask_unique_value):
#             p = p.long().item()
#             self.momentum = self.iteration / self.max_iteration
#             self.iteration = self.iteration + 1
#             self.protos[k][p-1] = self.protos[k][p-1] * self.momentum + prototypes[count] * (1 - self.momentum)


###################################################################################
###################################################################################
# Original
###################################################################################
###################################################################################
# freeze
class discriminate(nn.Module):
    def __init__(self):
        super(discriminate, self).__init__()
        self.epsilon = 1e-6

    def forward(self, masks, outputs):
        loss = 0.0
        B,C,H,W = outputs.shape

        for b in range(B):       
            output = outputs[b]
            mask = masks[b]
            mask_unique_value = torch.unique(mask)
            unique_num = len(mask_unique_value)
            if 1 < unique_num:
                prototypes = torch.zeros(size=(unique_num, C))
                for count,p in enumerate(mask_unique_value):
                    p = p.long()
                    bin_mask = torch.tensor(mask==p,dtype=torch.int8)
                    s = torch.sum(bin_mask)
                    bin_mask = bin_mask.unsqueeze(dim=0).expand_as(output)
                    v = torch.sum(bin_mask*output,dim=[1,2])/s
                    prototypes[count] = v
                    
                distances = torch.cdist(prototypes, prototypes, p=2.0)
                loss = loss +  1.0 / torch.mean(distances)
        
        return loss


class proto(nn.Module):
    def __init__(self):
        super(proto, self).__init__()
        self.epsilon = 1e-6

        # Synapse
        num_class = 9

        self.num_class = num_class

        self.protos = torch.zeros(num_class, 64)
        self.accumlator = torch.zeros(num_class, 64)
        self.iter_num = 0
        self.max_iterations = 5 * 368

    def forward(self, masks, outputs):

        targets = masks.long()
        predictions = torch.argmax(input=outputs,dim=1).long()

        loss = 0.0
        B,C,H,W = outputs.shape
        
        mask_unique_value = torch.unique(masks)
        unique_num = len(mask_unique_value)
        prototypes = torch.zeros(size=(unique_num,C))

        for count,p in enumerate(mask_unique_value):
            p = p.long()
            bin_mask = torch.tensor(masks==p,dtype=torch.int8)
            bin_m = bin_mask
            bin_mask = bin_mask.unsqueeze(dim=1).expand_as(outputs)

            bin_mask_t = torch.tensor(predictions==p,dtype=torch.int8)
            bin_mt = bin_mask_t
            bin_mask_t = bin_mask_t.unsqueeze(dim=1).expand_as(outputs)

            temp = 0.0
            for t in range(B):
                if torch.sum(bin_mask[t])!=0:
                    v = torch.sum(bin_mask[t]*outputs[t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
                    # w = torch.sum(bin_mask_t[t],dim=[1,2]) / torch.sum(bin_mask[t],dim=[1,2])
                    # w = self.dice(score=bin_mt, target=bin_m)
                    w = (1.0 - self.iter_num / self.max_iterations) ** 0.9
                    temp = temp + (w * v)
            prototypes[count] = temp 
        self.update(prototypes, mask_unique_value)
        self.iter_num = self.iter_num + 1
    @torch.no_grad()
    def update(self, prototypes, mask_unique_value):
        for count, p in enumerate(mask_unique_value):
            p = p.long().item()
            self.accumlator[p] = self.accumlator[p] + prototypes[count] 
            self.protos[p] = nn.functional.normalize(self.accumlator[p], p=2.0, dim=0, eps=1e-12, out=None)

    def psudo(self, outputs):
        B, C, H, W = outputs.shape
        temp = torch.zeros(B, self.num_class, H, W).cuda()
        for i in range(self.num_class):
            v = self.protos[i].unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(outputs).cuda()
            temp[:,i,:,:] = torch.norm(outputs-v, dim=1)
        label = torch.argmin(temp, dim=1)
        return label

    def dice(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice


class prototype_loss(nn.Module):
    def __init__(self):
        super(prototype_loss, self).__init__()
        self.epsilon = 1e-6
        self.down_scales = [1.0,1.0,0.5,0.25,0.125]

        # ENet
        # self.down_scales = [0.5,0.25,0.125,0.125]

        # ESPNet
        # self.down_scales = [1.0,0.5,0.5,0.25,0.125]

        # Mobile_NetV2
        # self.down_scales = [1.0,0.125,0.125,0.25,0.25]

        # SUNet
        # self.down_scales = [1.0,1.0,0.5,0.25,0.125]

        # ResNet_18
        # self.down_scales = [1.0, 0.25, 0.125, 0.0625, 0.03125]

        # ACDC
        # num_class = 3

        # CT-1K
        # num_class = 12

        # Synapse
        num_class = 9 

        self.num_class = num_class
        
        # Attention UNet
        # self.proto_1 = torch.zeros(num_class, 64 )
        # self.proto_2 = torch.zeros(num_class, 128)
        # self.proto_3 = torch.zeros(num_class, 256)
        # self.proto_4 = torch.zeros(num_class, 512)

        # ENet
        # self.proto_0 = torch.zeros(num_class, 13 )
        # self.proto_1 = torch.zeros(num_class, 16 )
        # self.proto_2 = torch.zeros(num_class, 64 )
        # self.proto_3 = torch.zeros(num_class, 128)
        # self.proto_4 = torch.zeros(num_class, 128)

        # ESPNet
        # self.proto_0 = torch.zeros(num_class, 9  )
        # self.proto_1 = torch.zeros(num_class, 16 )
        # self.proto_2 = torch.zeros(num_class, 9  )
        # self.proto_3 = torch.zeros(num_class, 64 )
        # self.proto_4 = torch.zeros(num_class, 128)

        # SUNet
        self.proto_0 = torch.zeros(num_class, self.num_class +1)
        self.proto_1 = torch.zeros(num_class, 8 )
        self.proto_2 = torch.zeros(num_class, 16)
        self.proto_3 = torch.zeros(num_class, 32)
        self.proto_4 = torch.zeros(num_class, 64)


        # Mobile_NetV2
        # self.proto_0 = torch.zeros(num_class, 9  )
        # self.proto_1 = torch.zeros(num_class, 9  )
        # self.proto_2 = torch.zeros(num_class, 320)
        # self.proto_3 = torch.zeros(num_class, 96 )
        # self.proto_4 = torch.zeros(num_class, 64 )

        # ResNet_18
        # self.proto_0 = torch.zeros(num_class, num_class+1)
        # self.proto_1 = torch.zeros(num_class, 64 )
        # self.proto_2 = torch.zeros(num_class, 128)
        # self.proto_3 = torch.zeros(num_class, 256)
        # self.proto_4 = torch.zeros(num_class, 512)


        # DABNet
        # self.proto_1 = torch.zeros(num_class, 9  )
        # self.proto_2 = torch.zeros(num_class, 64 )
        # self.proto_3 = torch.zeros(num_class, 128)
        # self.proto_4 = torch.zeros(num_class, 9  )

        # self.proto_0 = torch.zeros(num_class, num_class+1)
        # self.proto_1 = torch.zeros(num_class, 32 )
        # self.proto_2 = torch.zeros(num_class, 32 )
        # self.proto_3 = torch.zeros(num_class, 64 )
        # self.proto_4 = torch.zeros(num_class, 128)

        # self.protos = torch.load('/content/UNet_V2/protos_file.pth')
        # self.protos = [self.proto_1, self.proto_2, self.proto_3, self.proto_4]

        self.protos = [self.proto_0,self.proto_1, self.proto_2, self.proto_3, self.proto_4]
        self.momentum = torch.tensor(0.0)
        self.iteration = 0
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
        self.momentum_schedule = cosine_scheduler(0.85, 1.0, 60.0, 368)

        # self.momentum_schedule = cosine_scheduler(0.85, 1.0, 60.0, 213)
        # self.momentum_schedule = cosine_scheduler(0.85, 1.0, 60.0, 198)
        self.pixel_wise = CriterionPixelWise()


    def forward(self, masks, t_masks, up4, up3, up2, up1, outputs):
        loss = 0.0
        # loss = loss + self.pixel_wise(preds_S=outputs, masks=masks)
        up = [outputs, up1, up2, up3, up4]

        # print(outputs.shape)
        # print(up1.shape)
        # print(up2.shape)
        # print(up3.shape)
        # print(up4.shape)

        for k in range(5):
            indexs = []
            WP = []
            B,C,H,W = up[k].shape
            
            temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
            temp_masks = temp_masks.squeeze(dim=1)

            temp_t_masks = nn.functional.interpolate(t_masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
            temp_t_masks = temp_t_masks.squeeze(dim=1)

            # t_mask_unique_value = torch.unique(temp_t_masks)
            # t_mask_unique_value = t_mask_unique_value[1:]
            # unique_num_t = len(t_mask_unique_value)

            mask_unique_value = torch.unique(temp_masks)
            mask_unique_value = mask_unique_value[1:]
            unique_num = len(mask_unique_value)
            
            if unique_num<2:
                return 0

            prototypes = torch.zeros(size=(unique_num,C))

            for count,p in enumerate(mask_unique_value):
                p = p.long()
                bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
                bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])

                bin_mask_t = torch.tensor(temp_t_masks==p,dtype=torch.int8)
                bin_mask_t = bin_mask_t.unsqueeze(dim=1).expand_as(up[k])

                temp = 0.0
                batch_counter = 0
                for t in range(B):
                    if torch.sum(bin_mask[t])!=0:
                        v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
                        temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
                        batch_counter = batch_counter + 1
                temp = temp / batch_counter
                prototypes[count] = temp
                WP.append(torch.sum(bin_mask_t)/torch.sum(bin_mask))

            WP = torch.tensor(WP)
            WP = torch.diag(WP)
            WP = WP.detach()

            indexs = [x.item()-1 for x in mask_unique_value]
            indexs.sort()

            l = 0.0
            proto = self.protos[k][indexs].unsqueeze(dim=0)
            prototypes = prototypes.unsqueeze(dim=0)
            distances_c = torch.cdist(proto.clone().detach(), prototypes, p=2.0)
            proto = self.protos[k][indexs].squeeze(dim=0)
            prototypes = prototypes.squeeze(dim=0)
            x = (torch.eye(distances_c[0].shape[0],distances_c[0].shape[1]))
            diagonal = distances_c[0] * x

            weights = 1.0 / distances_c.clamp(min=self.epsilon)
            weights = weights / weights.max()
            weights = weights.detach()


            proto = prototypes.unsqueeze(dim=0)
            distances = torch.cdist(proto.clone().detach(), proto, p=2.0)
            l = l + (1.0 / torch.mean(weights * distances))

            l = l + (1.0 / torch.mean(weights * (distances_c[0]-diagonal)))
            l = l + (1.0 * torch.mean(WP * diagonal))

            # l = l + 0.5 * cosine_loss


            loss = loss + l

            self.update(prototypes, mask_unique_value, k)
        self.iteration = self.iteration + 1

        return loss


    @torch.no_grad()
    def update(self, prototypes, mask_unique_value, k):
        for count, p in enumerate(mask_unique_value):
            p = p.long().item()
            self.momentum = self.momentum_schedule[self.iteration] 
            self.protos[k][p-1] = self.protos[k][p-1] * self.momentum + prototypes[count] * (1 - self.momentum)
        
        # if self.iteration % 689==0:
        #     self.iteration = 0
        #     self.momentum_schedule = cosine_scheduler(0.85, 1.0, 5.0, 138)
        #     self.proto_0 = torch.zeros(self.num_class, 9  )
        #     self.proto_1 = torch.zeros(self.num_class, 16 )
        #     self.proto_2 = torch.zeros(self.num_class, 64 )
        #     self.proto_3 = torch.zeros(self.num_class, 128)
        #     self.proto_4 = torch.zeros(self.num_class, 128)
        #     self.protos = [self.proto_0,self.proto_1, self.proto_2, self.proto_3, self.proto_4]


class CriterionPixelWise(nn.Module):
    def __init__(self):
        super(CriterionPixelWise, self).__init__()
        num_class = 8
        self.num_class = num_class
        self.proto = torch.zeros(num_class+1, num_class+1)
        self.momentum = torch.tensor(0.0)
        self.iteration = 0
        self.momentum_schedule = cosine_scheduler(0.85, 1.0, 60.0, 368)

    def forward(self, preds_S, masks):
        loss = 0.0
        B,C,H,W = preds_S.shape
        temp_masks = masks
        mask_unique_value = torch.unique(temp_masks)
        unique_num = len(mask_unique_value) 
        prototypes = torch.zeros(size=(unique_num,C))

        for count,p in enumerate(mask_unique_value):
            p = p.long()
            bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
            bin_mask = bin_mask.unsqueeze(dim=1).expand_as(preds_S)
            temp = 0.0
            batch_counter = 0
            for t in range(B):
                if torch.sum(bin_mask[t])!=0:
                    v = torch.sum(bin_mask[t]*preds_S[t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
                    batch_counter = batch_counter + 1
            temp = temp / batch_counter
            prototypes[count] = temp

        preds_T = torch.zeros(preds_S.shape).to('cuda')
        mask_unique_value = torch.unique(masks)
        for i in mask_unique_value:
            i = i.long().item()
            expand = self.proto[i].unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(preds_S).to('cuda')
            temp_masks = (masks==i).unsqueeze(dim=1).expand_as(preds_S)
            preds_T = preds_T + (temp_masks * expand)
        preds_T.detach()
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss =  loss + (0.1 * (torch.sum(-softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H)
        self.update(prototypes, mask_unique_value)
        self.iteration = self.iteration + 1
        return loss

    @torch.no_grad()
    def update(self, prototypes, mask_unique_value):
        for count, p in enumerate(mask_unique_value):
            p = p.long().item()
            self.momentum = self.momentum_schedule[self.iteration] 
            self.proto[p] = self.proto[p] * self.momentum + prototypes[count] * (1 - self.momentum)

def ind(x, top):
    x=x+1
    if x==top:
        return 0
    else:
        return x

def one_hot_loss(unique_num, unique_num_t):
    xp = []
    yp = []
    for i in range(9):
        if i in unique_num:
            yp.append(1.0)
        else:
            yp.append(0.0)
        if i in unique_num_t:
            xp.append(1.0)
        else:
            xp.append(0.0)
    xp = torch.tensor(xp)
    yp = torch.tensor(yp)
    loss = torch.nn.functional.binary_cross_entropy(input=xp, target=yp)
    return loss

# class prototype_loss(nn.Module):
#     def __init__(self):
#         super(prototype_loss, self).__init__()
#         self.down_scales = [1.0,0.5,0.25,0.125]

#         # # ENet
#         # self.down_scales = [0.5,0.25,0.125,0.125]

#         num_class = 8
#         memory_size = 16
#         self.memory_size = memory_size
        
#         self.proto_1 = torch.zeros(memory_size, num_class, 64)
#         self.proto_1_index = torch.zeros(num_class, dtype=torch.int8)

#         self.proto_2 = torch.zeros(memory_size, num_class, 128)
#         self.proto_2_index = torch.zeros(num_class, dtype=torch.int8)

#         self.proto_3 = torch.zeros(memory_size, num_class, 256)
#         self.proto_3_index = torch.zeros(num_class, dtype=torch.int8)

#         self.proto_4 = torch.zeros(memory_size, num_class, 512)
#         self.proto_4_index = torch.zeros(num_class, dtype=torch.int8)

#         # # ENet
#         # self.proto_1 = torch.zeros(num_class, 16)
#         # self.proto_2 = torch.zeros(num_class, 64)
#         # self.proto_3 = torch.zeros(num_class, 128)
#         # self.proto_4 = torch.zeros(num_class, 128)

#         # self.proto_1 = torch.zeros(num_class, 64)
#         # self.proto_2 = torch.zeros(num_class, 64)
#         # self.proto_3 = torch.zeros(num_class, 128)
#         # self.proto_4 = torch.zeros(num_class, 256)

#         self.protos = [{'proto':self.proto_1,'index':self.proto_1_index}, 
#                        {'proto':self.proto_2,'index':self.proto_2_index},
#                        {'proto':self.proto_3,'index':self.proto_3_index},
#                        {'proto':self.proto_4,'index':self.proto_4_index}]

#         self.momentum = torch.tensor(0.0)
#         self.iteration = -1

#         self.momentum_schedule = cosine_scheduler(0.0, 1.0, 30, 368)

#     def forward(self, masks, t_masks, up4, up3, up2, up1):
#         loss = 0.0
#         up = [up1, up2, up3, up4]

#         for k in range(4):
#             indexs = []
#             weights = []
#             B,C,H,W = up[k].shape
            
#             temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks = temp_masks.squeeze(dim=1)

#             temp_t_masks = nn.functional.interpolate(t_masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_t_masks = temp_t_masks.squeeze(dim=1)

#             t_mask_unique_value = torch.unique(temp_t_masks)
#             t_mask_unique_value = t_mask_unique_value[1:]
#             unique_num_t = len(t_mask_unique_value)

#             mask_unique_value = torch.unique(temp_masks)
#             mask_unique_value = mask_unique_value[1:]
#             # mask_unique_value = [x for x in mask_unique_value if x in t_mask_unique_value]
#             # mask_unique_value.sort()
#             unique_num = len(mask_unique_value)
            
#             # if unique_num<2:
#             #     return 0

#             if k==1:
#                 loss = loss + one_hot_loss(unique_num=torch.unique(temp_masks), unique_num_t=torch.unique(temp_t_masks))
                

#             prototypes = torch.zeros(size=(unique_num,C))
#             prototypes_t = torch.zeros(size=(unique_num_t,C))
#             # prototypes_t = torch.zeros(size=(unique_num,C))

#             for count,p in enumerate(mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         batch_counter = batch_counter + 1
#                 temp = temp / batch_counter
#                 prototypes[count] = temp
                
#                 if p in t_mask_unique_value:
#                     indexs.append(count)

#             indexs.sort()

#             for count,p in enumerate(t_mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_t_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         batch_counter = batch_counter + 1
#                 temp = temp / batch_counter
#                 prototypes_t[count] = temp
                
#                 # if p in t_mask_unique_value:
#                 #     num = torch.tensor(temp_t_masks==p,dtype=torch.int8).sum()
#                 #     den = torch.tensor(temp_masks==p,dtype=torch.int8).sum()
#                 #     if 0.9 <= num/den:
#                 #         self.update(prototypes_t[count], k=k, p=p)

#             # indexs = [x.item()-1 for x in mask_unique_value]
#             # indexs.sort()
#             # indexs_all = [x for x in range(8)]
#             # temp_indexs = [x for x in indexs_all if x not in indexs]
#             # indexs = [*indexs,*temp_indexs]
#             # indexs = [float(x) for x in indexs]

#             #####################################################
#             #####################################################

#             # temp_t_masks = nn.functional.interpolate(t_masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             # temp_t_masks = temp_t_masks.squeeze(dim=1)

#             # t_mask_unique_value = torch.unique(temp_t_masks)
#             # t_mask_unique_value = t_mask_unique_value[1:]
#             # t_unique_num = len(t_mask_unique_value)
            
#             # prototypes_t = torch.zeros(size=(t_unique_num,C))

#             # for count,p in enumerate(t_mask_unique_value):
#             #     p = p.long()
#             #     bin_mask = torch.tensor(temp_t_masks==p,dtype=torch.int8)
#             #     bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
#             #     temp = 0.0
#             #     batch_counter = 0
#             #     for t in range(B):
#             #         if torch.sum(bin_mask[t])!=0:
#             #             v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#             #             temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#             #             batch_counter = batch_counter + 1
#             #     temp = temp / batch_counter
#             #     prototypes_t[count] = temp

#             #####################################################
#             #####################################################

#             l = 0.0
#             # proto = self.protos[k][indexs].unsqueeze(dim=0)
#             # prototypes = prototypes.unsqueeze(dim=0)
#             # distances_c = torch.cdist(proto.clone().detach(), prototypes, p=2.0)
#             # proto = self.protos[k][indexs].squeeze(dim=0)
#             # prototypes = prototypes.squeeze(dim=0)
#             # diagonal = distances_c[0] * (torch.eye(distances_c[0].shape[0],distances_c[0].shape[1]))

#             if 1<unique_num:
#                 proto = prototypes.unsqueeze(dim=0)
#                 distances = torch.cdist(proto.clone().detach(), proto, p=2.0)
#                 l = l + (1.0 / torch.mean(distances))

#             if 0<len(indexs):
#                 proto = prototypes[indexs].unsqueeze(dim=0)
#                 proto_t = prototypes_t.unsqueeze(dim=0)      
#                 distances_t = torch.cdist(proto_t.clone().detach(), proto, p=2.0)
#                 diagonal = distances_t[0] * (torch.eye(distances_t[0].shape[0],distances_t[0].shape[1]))
#                 l = l + torch.mean(diagonal)
                
#             # l = l + (1.0 / torch.mean(distances_c[0]-diagonal))
#             # l = l + (0.1 * (torch.mean(diagonal)))
#             loss = loss + l
#             # self.update(prototypes_t, t_mask_unique_value, k)

#         return loss

#     @torch.no_grad()
#     def update(self, prototypes, k, p):
#         index = self.protos[k]['index'][p-1]
#         self.protos[k]['proto'][index][p-1] = prototypes
#         self.protos[k]['index'][p-1] = ind(index, top=self.memory_size)

    # @torch.no_grad()
    # def update(self, prototypes, mask_unique_value, k):
    #     for count, p in enumerate(mask_unique_value):
    #         p = p.long().item()
    #         self.momentum = self.momentum_schedule[self.iteration] 
    #         self.protos[k][p-1] = self.protos[k][p-1] * self.momentum + prototypes[count] * (1 - self.momentum)

# class IM_loss(nn.Module):
#     def __init__(self):
#         super(IM_loss, self).__init__()

#     def forward(self, up4, up3, up2, up1, e4, e3, e2, e1):
#         loss = 0.0
#         loss = loss + self.IMD(t=up1, s=e1)
#         loss = loss + self.IMD(t=up2, s=e2)
#         loss = loss + self.IMD(t=up3, s=e3)
#         # loss = loss + self.IMD(t=up4, s=e4)
#         return loss

#     def at(self, x, exp):
#         """
#         attention value of a feature map
#         :param x: feature
#         :return: attention value
#         """
#         attention = x.pow(exp).mean(1) 
#         return F.normalize(attention.view(attention.size(0), -1))

#     def IMD(self, s, t, exp=2):
#         """
#         importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
#         Improving the Performance of Convolutional Neural Networks via Attention Transfer"
#         https://arxiv.org/abs/1612.03928
#         :param exp: exponent
#         :param s: student feature maps
#         :param t: teacher feature maps
#         :return: imd loss value
#         """
#         return torch.sum((self.at(s, exp) - self.at(t, exp)).pow(2), dim=1).mean()
        

# class IM_loss(nn.Module):
#     def __init__(self):
#         super(IM_loss, self).__init__()

#     def forward(self, masks, up3, up2, up1):
#         masks[masks!=0] = 1
#         loss = 0.0
#         loss = loss + self.IMD(t=up1, s=up2, masks=masks, scale_factor=0.5)
#         loss = loss + self.IMD(t=up2, s=up3, masks=masks, scale_factor=0.25)
#         return loss

#     def at(self, x, exp):
#         """
#         attention value of a feature map
#         :param x: feature
#         :return: attention value
#         """
#         attention = x.pow(exp).mean(1)
#         return F.normalize(attention.view(attention.size(0), -1))

#     def IMD(self, s, t, masks, scale_factor, exp=2):
#         """
#         importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
#         Improving the Performance of Convolutional Neural Networks via Attention Transfer"
#         https://arxiv.org/abs/1612.03928
#         :param exp: exponent
#         :param s: student feature maps
#         :param t: teacher feature maps
#         :return: imd loss value
#         """
#         masks = F.interpolate(masks.unsqueeze(dim=1), scale_factor=scale_factor, mode='nearest')
#         masks = masks.squeeze(dim=1)
#         if s.shape[2] != t.shape[2]:
#             t = F.interpolate(t, s.size()[-2:], mode='bilinear')
#         return torch.sum((self.at(s, exp, masks) - self.at(t, exp, masks)).pow(2), dim=1).mean()
        

class IM_loss(nn.Module):
    def __init__(self):
        super(IM_loss, self).__init__()

    def forward(self, masks, x4, x3, x2, x1):

        masks[masks!=0] = 1.0

        loss = 0.0
        # loss = loss + self.IMD(s=x4, masks=masks, scale_factor=0.125)
        # loss = loss + self.IMD(s=x3, masks=masks, scale_factor=0.125)
        loss = loss + self.IMD(s=x2, masks=masks, scale_factor=0.250)
        loss = loss + self.IMD(s=x1, masks=masks, scale_factor=0.500)

        return loss

    def at(self, x, exp, masks):
        """
        attention value of a feature map
        :param x: feature
        :return: attention value
        """
        attention = x.pow(exp).mean(1) * masks 
        return F.normalize(attention.view(attention.size(0), -1))


    def IMD(self, s, masks, scale_factor, exp=2):
        """
        importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
        Improving the Performance of Convolutional Neural Networks via Attention Transfer"
        https://arxiv.org/abs/1612.03928
        :param exp: exponent
        :param s: student feature maps
        :param t: teacher feature maps
        :return: imd loss value
        """
        masks = F.interpolate(masks.unsqueeze(dim=1), scale_factor=scale_factor, mode='nearest')
        masks = masks.squeeze(dim=1)

        masks_attention = F.normalize(masks.view(masks.size(0), -1))
        masks_attention = masks_attention.squeeze(dim=1)        

        return torch.sum((self.at(s, exp, masks) - masks_attention).pow(2), dim=1).mean()

    # def IMD(self, s, t, masks, scale_factor, exp=2):
    #     """
    #     importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    #     Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    #     https://arxiv.org/abs/1612.03928
    #     :param exp: exponent
    #     :param s: student feature maps
    #     :param t: teacher feature maps
    #     :return: imd loss value
    #     """
    #     masks = F.interpolate(masks.unsqueeze(dim=1), scale_factor=scale_factor, mode='nearest')
    #     masks = masks.squeeze(dim=1)
    #     if t.shape[2] != s.shape[2]:
    #         t = F.interpolate(t, s.size()[-2:], mode='bilinear')
    #     return torch.sum((self.at(s, exp, masks) - self.at(t, exp, masks)).pow(2), dim=1).mean()


# class M_loss(nn.Module):
#     def __init__(self):
#         super(M_loss, self).__init__()
#         self.softmax = Softmax(dim=2)
#     def forward(self, e5):
#         loss = 0
#         e5 = e5.to('cpu')
#         B, C, H, W = e5.shape # (B, C, H, W) ---> H=W=16, C=1024
#         C_step = C//4
#         for i in range(3):
#             base = i * C_step
#             end = (i+1) * C_step
#             x = e5[:,base:end,:,:].flatten(2) # (B, C, n_patches) ---> n_patches=256
#             Q = x
#             K = x.transpose(2, 1)  # (B, n_patches, C)
#             attention_scores = torch.matmul(Q, K) # (B, C, C)
#             attention_scores = attention_scores / math.sqrt(C) # (B, C, C)
#             attention_probs = self.softmax(attention_scores) # (B, C, C)
#             probs = attention_probs # (B, C, C)
#             probs = probs.sum(dim=0) # (C, C)
#             diag = probs * (torch.eye(probs.shape[0],probs.shape[1])) # (C, C)
#             probs = probs - diag # (C, C)
#             l = torch.norm(probs)
#             loss = loss + l
#         return loss

class M_loss(nn.Module):
    def __init__(self):
        super(M_loss, self).__init__()
        self.softmax = Softmax(dim=2)
    def forward(self, e5):
        loss = 0
        E = e5
        B, C, H, W = E.shape # (B, C, H, W) ---> H=W=16, C=512
        e = E
        x = e.flatten(2) # (B, C, n_patches) ---> n_patches=256
        Q = x
        K = x.transpose(2, 1)  # (B, n_patches, C)
        attention_scores = torch.matmul(Q, K) # (B, C, C)
        attention_scores = attention_scores / math.sqrt(C) # (B, C, C)
        attention_probs = self.softmax(attention_scores) # (B, C, C)
        probs = attention_probs # (B, C, C)
        probs = probs.sum(dim=0) # (C, C)
        diag = probs * (torch.eye(probs.shape[0],probs.shape[1]).to('cuda')) # (C, C)
        probs = probs - diag
        l = torch.norm(probs) 
        loss = loss + l
        return loss


# class M_loss(nn.Module):
#     def __init__(self):
#         super(M_loss, self).__init__()
#         self.softmax = Softmax(dim=2)
#         self.scales = [0.5, 0.25, 0.125, 0.0625]
#     def forward(self, up4, up3, up2, up1):
#         up = [up4, up3, up2, up1]
#         loss_net = 0.0
#         for i in range(3):
#             E = up[i].to('cpu')
#             E = nn.functional.interpolate(E, scale_factor=self.scales[i], mode='nearest')
#             B, C, H, W = E.shape # (B, C, H, W) ---> H=W=16, C=1024
#             x = E.flatten(2) # (B, C, n_patches) ---> n_patches=256
#             Q = x
#             K = x.transpose(2, 1)  # (B, n_patches, C)
#             attention_scores = torch.matmul(Q, K) # (B, C, C)
#             attention_scores = attention_scores / math.sqrt(C) # (B, C, C)
#             attention_probs = self.softmax(attention_scores) # (B, C, C)
#             probs = attention_probs # (B, C, C)
#             probs = probs.sum(dim=0) # (C, C)
#             diag = probs * (torch.eye(probs.shape[0],probs.shape[1])) # (C, C)
#             probs = probs - diag
#             loss = torch.norm(probs)
#             loss_net = loss_net + loss
#         return loss_net

# class M_loss(nn.Module):
#     def __init__(self):
#         super(M_loss, self).__init__()
#         self.Ci_num = [64, 128, 256, 512]
#     def forward(self, probs1, probs2, probs3, probs4):
#         loss = 0.0
#         probs_total = [probs1, probs2, probs3, probs4]
#         for i , prob in enumerate(probs_total):
#             prob = prob.to('cpu')
#             B, H, Ci, Csigma = prob.shape
#             if i==0:
#                 prob = prob[:,:,:,0:self.Ci_num[0]]
#             else:
#                 prob = prob[:,:,:,self.Ci_num[i-1]:(self.Ci_num[i-1]+self.Ci_num[i])]
#             prob = prob.sum(dim=0).sum(dim=1)
#             diag = prob * (torch.eye(prob.shape[0],prob.shape[1]))
#             prob = prob - diag
#             loss = loss + torch.norm(prob)
#         return loss


# class prototype_loss(nn.Module):
#     def __init__(self):
#         super(prototype_loss, self).__init__()

#         # DABNet
#         self.down_scales = [0.5,0.25,0.125]

#         num_class = 9

#         # self.proto_1 = torch.zeros(num_class, 64 )
#         # self.proto_2 = torch.zeros(num_class, 128)
#         # self.proto_3 = torch.zeros(num_class, 256)
#         # self.proto_4 = torch.zeros(num_class, 512)

#         # DABNet
#         self.proto_1 = torch.zeros(num_class, 32)
#         self.proto_2 = torch.zeros(num_class, 64)
#         self.proto_3 = torch.zeros(num_class, 128)


#         self.protos = [self.proto_1, self.proto_2, self.proto_3]
#         self.momentum = torch.tensor(0.9)
#         self.iteration = 0
#         self.max_iteration = 368 * 30.0

#         self.momentum_schedule = cosine_scheduler(0.8, 1.0, 45, 477)

#     def forward(self, masks, up3, up2, up1):
#         loss = 0.0
#         up = [up1, up2, up3]

#         for k in range(3):
#             B,C,H,W = up[k].shape
            
#             temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks = temp_masks.squeeze(dim=1)

#             mask_unique_value = torch.unique(temp_masks)
#             mask_unique_value = mask_unique_value[1:]
#             unique_num = len(mask_unique_value)
            
#             if unique_num<2:
#                 return 0

#             prototypes = torch.zeros(size=(unique_num,C))

#             for count,p in enumerate(mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         # temp = temp + self.softmax(nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None) / self.temp)
#                         batch_counter = batch_counter + 1
#                 temp = temp / batch_counter
#                 prototypes[count] = temp

#             indexs = [x.item()-1 for x in mask_unique_value]

#             l = 0.0

#             proto = self.protos[k][indexs].unsqueeze(dim=0)
#             # proto = self.protos[k].unsqueeze(dim=0)
#             prototypes = prototypes.unsqueeze(dim=0)
#             distances_c = torch.cdist(proto.clone().detach(), prototypes, p=2.0)
#             proto = self.protos[k][indexs].squeeze(dim=0)
#             # proto = self.protos[k].squeeze(dim=0)
#             prototypes = prototypes.squeeze(dim=0)

#             diagonal = distances_c[0] * (torch.eye(distances_c[0].shape[0],distances_c[0].shape[1]))

#             # prototypes = prototypes.unsqueeze(dim=0)
#             # distances = torch.cdist(prototypes.clone().detach(), prototypes, p=2.0)
#             # prototypes = prototypes.squeeze(dim=0)

#             # l = l + (1.0 / torch.mean(distances))
#             l = l + (1.0 / torch.mean(distances_c[0]-diagonal))
#             l = l + (torch.mean(diagonal))

#             loss = loss + l
#             self.update(prototypes, mask_unique_value, k)
#         self.iteration = self.iteration + 1
#         return loss

#     @torch.no_grad()
#     def update(self, prototypes, mask_unique_value, k):
#         for count, p in enumerate(mask_unique_value):
#             p = p.long().item()
#             # self.momentum = self.momentum_schedule[self.iteration] 
#             # self.momentum =  0.9 - (0.9 * ((1.0 - self.iteration / self.max_iteration) ** 0.9))
#             self.protos[k][p-1] = self.protos[k][p-1] * self.momentum + prototypes[count] * (1 - self.momentum)




# class prototype_loss(nn.Module):
#     def __init__(self):
#         super(prototype_loss, self).__init__()
#         self.down_scales = [1.0,0.5,0.25,0.125]
#         num_class = 9
#         self.temp = 0.1
#         self.softmax = torch.nn.Softmax(dim=0)

#         self.proto_1 = torch.zeros(num_class, 64 )
#         self.proto_2 = torch.zeros(num_class, 128)
#         self.proto_3 = torch.zeros(num_class, 256)
#         self.proto_4 = torch.zeros(num_class, 512)

#         # self.proto_1 = torch.zeros(num_class, 64)
#         # self.proto_2 = torch.zeros(num_class, 64)
#         # self.proto_3 = torch.zeros(num_class, 128)
#         # self.proto_4 = torch.zeros(num_class, 256)

#         self.protos = [self.proto_1, self.proto_2, self.proto_3, self.proto_4]
#         self.momentum = torch.tensor(0.9)
#         self.iteration = 0.0
#         self.max_iteration = 368 * 30.0
#     def forward(self, masks, up4, up3, up2, up1):
#         self.iteration = self.iteration + 1.0
#         loss = 0.0
#         up = [up1, up2, up3, up4]

#         for k in range(4):
#             B,C,H,W = up[k].shape
            
#             temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=self.down_scales[k], mode='nearest')
#             temp_masks = temp_masks.squeeze(dim=1)

#             mask_unique_value = torch.unique(temp_masks)
#             unique_num = len(mask_unique_value)
            
#             if unique_num<2:
#                 return 0

#             prototypes = torch.zeros(size=(unique_num,C))

#             for count,p in enumerate(mask_unique_value):
#                 p = p.long()
#                 bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
#                 bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
#                 temp = 0.0
#                 batch_counter = 0
#                 for t in range(B):
#                     if torch.sum(bin_mask[t])!=0:
#                         v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
#                         temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
#                         batch_counter = batch_counter + 1
#                 temp = temp / batch_counter
#                 prototypes[count] = temp

#             indexs = [x.item() for x in mask_unique_value]

#             l = 0.0

#             proto = self.protos[k][indexs].unsqueeze(dim=0)
#             # proto = self.protos[k].unsqueeze(dim=0)
#             prototypes = prototypes.unsqueeze(dim=0)
#             distances_c = torch.cdist(proto.clone().detach(), prototypes, p=2.0)
#             proto = self.protos[k][indexs].squeeze(dim=0)
#             # proto = self.protos[k].squeeze(dim=0)
#             prototypes = prototypes.squeeze(dim=0)

#             diagonal = distances_c[0] * (torch.eye(distances_c[0].shape[0],distances_c[0].shape[1]))

#             # prototypes = prototypes.unsqueeze(dim=0)
#             # distances = torch.cdist(prototypes.clone().detach(), prototypes, p=2.0)
#             # prototypes = prototypes.squeeze(dim=0)

#             # l = l + (1.0 / torch.mean(distances))
#             l = l + (1.0 / torch.mean(distances_c[0]-diagonal))
#             l = l + (torch.mean(diagonal))

#             loss = loss + l
#             self.update(prototypes, mask_unique_value, k)

#         return loss

#     @torch.no_grad()
#     def update(self, prototypes, mask_unique_value, k):
#         for count, p in enumerate(mask_unique_value):
#             p = p.long().item()
#             if p!=0:
#                 self.protos[k][p] = self.protos[k][p] * self.momentum + prototypes[count] * (1 - self.momentum)




class prototype_loss_kd(nn.Module):
    def __init__(self):
        super(prototype_loss_kd, self).__init__()
        self.num_class = 9
        self.temp = 1.0
        self.softmax = torch.nn.Softmax(dim=0)

        self.proto = torch.zeros(self.num_class-1, self.num_class)
        self.momentum = torch.tensor(0.9)

    def forward(self, masks, logits):
        B,C,H,W = logits.shape
        mask_unique_value = torch.unique(masks)
        mask_unique_value = mask_unique_value[1:]
        unique_num = len(mask_unique_value)

        if unique_num==0:
            return 0            
    
        prototypes = torch.zeros(size=(unique_num,self.num_class))

        for count,p in enumerate(mask_unique_value):
            p = p.long()
            bin_mask = torch.tensor(masks==p,dtype=torch.int8)
            bin_mask = bin_mask.unsqueeze(dim=1).expand_as(logits)
            temp = 0.0
            batch_counter = 0
            for t in range(B):
                if torch.sum(bin_mask[t])!=0:
                    v = torch.sum(bin_mask[t]*logits[t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
                    temp = temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
                    batch_counter = batch_counter + 1
            temp = temp / batch_counter
            prototypes[count] = temp

        indexs = [x.item()-1 for x in mask_unique_value]
        l = 0.0
        proto = self.proto[indexs].unsqueeze(dim=0)
        prototypes = prototypes.unsqueeze(dim=0)
        distances_c = torch.cdist(proto.clone().detach(), prototypes, p=2.0)
        proto = self.proto[indexs].squeeze(dim=0)
        prototypes = prototypes.squeeze(dim=0)

        diagonal = distances_c[0] * (torch.eye(distances_c[0].shape[0],distances_c[0].shape[1]))

        # prototypes = prototypes.unsqueeze(dim=0)
        # distances = torch.cdist(prototypes.clone().detach(), prototypes, p=2.0)
        # prototypes = prototypes.squeeze(dim=0)

        # l = l + (1.0 / torch.mean(distances))
        l = l + (1.0 / torch.mean(distances_c[0]-diagonal))
        l = l + (torch.mean(diagonal))

        self.update(prototypes, mask_unique_value)
        loss = l
        return loss

    @torch.no_grad()
    def update(self, prototypes, mask_unique_value):
        for count, p in enumerate(mask_unique_value):
            p = p.long().item()
            self.proto[p-1] = self.proto[p-1] * self.momentum + prototypes[count] * (1 - self.momentum)

    def mask_kd(self, logits, masks, mask_unique_value):
        teacher_scores = torch.zeros_like(logits).cuda()
        B,C,H,W = teacher_scores.shape
        for count, p in enumerate(mask_unique_value):
            p = p.long().item()
            bin_masks = masks
            bin_masks[bin_masks!=p] = 0.0 
            bin_masks[bin_masks==p] = 1.0 
            bin_masks = bin_masks.unsqueeze(dim=1)
            temp = self.proto[p-1].unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(teacher_scores).to('cuda') * bin_masks.to('cuda')
            teacher_scores = teacher_scores + temp
        return teacher_scores

    def prediction_map_distillation(self, logits, teacher_scores, masks, T=1.0) :
        """
        basic KD loss function based on "Distilling the Knowledge in a Neural Network"
        https://arxiv.org/abs/1503.02531
        :param y: student score map
        :param teacher_scores: teacher score map
        :param T:  for softmax
        :return: loss value
        """
        logits = logits.cuda()
        masks = masks.long()
        masks = masks.cuda()

        bin_masks = masks
        bin_masks[bin_masks!=0] = 1.0 

        logits_prime = logits * bin_masks.unsqueeze(dim=1).expand_as(logits)
        teacher_scores_prime = teacher_scores * bin_masks.unsqueeze(dim=1).expand_as(teacher_scores)

        p = F.log_softmax(logits_prime / T , dim=1)
        q = F.softmax(teacher_scores_prime / T, dim=1)

        p = p.view(-1, 2)
        q = q.view(-1, 2)

        l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
        return l_kl

class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=1, device='cpu'):
        super(SpatialSoftmax, self).__init__()

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature).to(device)
        else:
            self.temperature = 1.0

    def forward(self, feature):
        feature = feature.view(feature.shape[0], -1, feature.shape[1] * feature.shape[2])
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)

        return softmax_attention

class atten_loss(nn.Module):

    def __init__(self):
        super(atten_loss, self).__init__()
        self.at_gen_upsample = nn.Upsample(scale_factor=2)
        self.at_gen_l2_loss = nn.MSELoss(reduction='mean')

    def at_gen(self, x1, x2, mask=False):
        """
        x1 - previous encoder step feature map
        x2 - current encoder step feature map
        """

        # G^2_sum
        sps = SpatialSoftmax(device = x1.device)

        if mask:
            x1 = sps(x1)
            x2 = x2.pow(2).sum(dim=1, keepdim=True)
            x2 = torch.squeeze(self.at_gen_upsample(x2), dim=1)
            x2 = sps(x2)
            loss = self.at_gen_l2_loss(x1, x2)
            return loss
        else:
            if x1.size() != x2.size():
                x1 = x1.pow(2).sum(dim=1)
                x1 = sps(x1)
                x2 = x2.pow(2).sum(dim=1, keepdim=True)
                x2 = torch.squeeze(self.at_gen_upsample(x2), dim=1)
                x2 = sps(x2)
            else:
                x1 = x1.pow(2).sum(dim=1)
                x1 = sps(x1)
                x2 = x2.pow(2).sum(dim=1)
                x2 = sps(x2)

            loss = self.at_gen_l2_loss(x1, x2)
            return loss

    def forward(self, masks, x1, x2, x3, x4):
        loss = 0.0
        masks[masks!=0] = 1
        loss = loss + self.at_gen(masks, x2, mask=True)
        loss = loss + self.at_gen(x1, x2)
        loss = loss + self.at_gen(x2, x3)
        loss = loss + self.at_gen(x3, x4)
        return loss


def masking(image,label,palate):
    assert image.shape==label.shape,f'Dimesion Mismatch: label Dim={label.shape}, img Dim={image.shape}'
    row,col = image.shape
    image_expand = np.expand_dims(image,axis=2)
    temp = np.concatenate((image_expand,image_expand,image_expand),axis=2)
    label = label.astype(dtype=np.uint8)
    for r in range(row):
        for c in range(col):
            if label[r,c]!=0:
                temp[r,c] = palate[label[r,c]]
    return temp

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    output = params/1000000
    return output


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)
    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
        This function can be called in an interleaved way.
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
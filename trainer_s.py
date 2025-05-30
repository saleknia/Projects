import utils
from utils import cosine_scheduler
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from utils import print_progress
import torch.nn.functional as F
import warnings
from utils import focal_loss, Dilation2d, Erosion2d
from torch.autograd import Variable
from torch.nn.functional import mse_loss as MSE
from utils import importance_maps_distillation as imd
from valid_s import valid_s
from sklearn.metrics import confusion_matrix
from SCL import SemanticConnectivityLoss
warnings.filterwarnings("ignore")

ALPHA = 0.8
GAMMA = 2

class M_loss(nn.Module):
    def __init__(self):
        super(M_loss, self).__init__()
        self.Ci_num = [96, 96, 96]
    def forward(self, probs1, probs2, probs3):
        loss = 0.0
        probs_total = [probs1, probs2, probs3]
        for i , prob in enumerate(probs_total):
            prob = prob.to('cpu')
            B, H, Ci, Csigma = prob.shape
            if i==0:
                prob = prob[:,:,:,0:self.Ci_num[0]]
            else:
                prob = prob[:,:,:,self.Ci_num[i-1]:(self.Ci_num[i-1]+self.Ci_num[i])]
            prob = prob.sum(dim=0).sum(dim=1)
            diag = prob * (torch.eye(prob.shape[0],prob.shape[1]))
            prob = prob - diag
            loss = loss + torch.norm(prob)
        return loss * 0.01

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

erosion = Erosion2d(1, 1, 9, soft_max=False)
dilate  = Dilation2d(1, 1, 9, soft_max=False)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))

def region_contrast(x, gt):
    """
    calculate region contrast value
    :param x: feature
    :param gt: mask
    :return: value
    """
    smooth = 1.0
    mask0 = gt[:, 0].unsqueeze(1)
    mask1 = gt[:, 1].unsqueeze(1)

    region0 = torch.sum(x * mask0, dim=(2, 3)) / torch.sum(mask0, dim=(2, 3))
    region1 = torch.sum(x * mask1, dim=(2, 3)) / (torch.sum(mask1, dim=(2, 3)) + smooth)
    return F.cosine_similarity(region0, region1, dim=1)


def region_affinity_distillation(s, t, gt):
    """
    region affinity distillation KD loss
    :param s: student feature
    :param t: teacher feature
    :return: loss value
    """
    gt = F.interpolate(gt, s.size()[2:])
    return (region_contrast(s, gt) - region_contrast(t, gt)).pow(2).mean()

def importance_maps_distillation(s, t, exp=4):
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    # if s.shape[2] != t.shape[2]:
    #     s = F.interpolate(s, t.size()[-2:], mode='bilinear')
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()

def attention_loss(e1, e2, e3, e1_t, e2_t, e3_t):

    return importance_maps_distillation(e1, e1_t) + importance_maps_distillation(e2, e2_t) + importance_maps_distillation(e3, e3_t)  

class CriterionPixelWise(nn.Module):
    def __init__(self):
        super(CriterionPixelWise, self).__init__()

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = preds_T.permute(0,2,3,1).contiguous().view(-1,C)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss

from torchmetrics.classification import BinaryConfusionMatrix

class Evaluator(object):
    ''' For using this evaluator target and prediction
        dims should be [B,H,W] '''
    def __init__(self):
        self.reset()
        self.metric = BinaryConfusionMatrix().to('cuda')
    def Pixel_Accuracy(self):
        Acc = torch.tensor(self.acc).mean()
        return Acc

    def Mean_Intersection_over_Union(self,per_class=False,show=False):
        IoU = torch.tensor(self.iou).mean()
        return IoU

    def Dice(self,per_class=False,show=False):
        Dice = torch.tensor(self.dice).mean()
        return Dice

    def add_batch(self, gt_image, pre_image):
        gt_image = gt_image.int()
        pre_image = pre_image.int()
        
        for i in range(gt_image.shape[0]):
            tn, fp, fn, tp = self.metric(pre_image[i].reshape(-1), gt_image[i].reshape(-1)).ravel()
            Acc = (tp + tn) / (tp + tn + fp + fn)
            IoU = (tp) / (tp + fp + fn)
            Dice =  (2 * tp) / ((2 * tp) + fp + fn)
            self.acc.append(Acc)
            self.iou.append(IoU)
            self.dice.append(Dice)

    def reset(self):
        self.acc = []
        self.iou = []
        self.dice = []

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

# class Evaluator(object):
#     ''' For using this evaluator target and prediction
#         dims should be [B,H,W] '''
#     def __init__(self):
#         self.reset()
        
#     def Pixel_Accuracy(self):
#         Acc = torch.tensor(np.mean(self.acc))
#         return Acc

#     def Mean_Intersection_over_Union(self,per_class=False,show=False):
#         IoU = torch.tensor(np.mean(self.iou))
#         return IoU

#     def Dice(self,per_class=False,show=False):
#         Dice = torch.tensor(np.mean(self.dice))
#         return Dice

#     def add_batch(self, gt_image, pre_image):
#         gt_image=gt_image.int().detach().cpu().numpy()
#         pre_image=pre_image.int().detach().cpu().numpy()
#         for i in range(gt_image.shape[0]):
#             tn, fp, fn, tp = confusion_matrix(gt_image[i].reshape(-1), pre_image[i].reshape(-1), labels=[0, 1]).ravel()
#             Acc = (tp + tn) / (tp + tn + fp + fn)
#             IoU = (tp) / (tp + fp + fn)
#             Dice =  (2 * tp) / ((2 * tp) + fp + fn)
#             self.acc.append(Acc)
#             self.iou.append(IoU)
#             self.dice.append(Dice)

#     def reset(self):
#         self.acc = []
#         self.iou = []
#         self.dice = []

# class Evaluator(object):
#     ''' For using this evaluator target and prediction
#         dims should be [B,H,W] '''
#     def __init__(self):
#         self.reset()
        
#     def Pixel_Accuracy(self):
#         Acc = torch.tensor(np.mean(self.acc))
#         return Acc

#     def Mean_Intersection_over_Union(self,per_class=False,show=False):
#         IoU = torch.tensor(np.mean(self.iou))
#         return IoU

#     def Dice(self,per_class=False,show=False):
#         Dice = torch.tensor(np.mean(self.dice))
#         return Dice

#     def add_batch(self, gt_image, pre_image):
#         gt_image=gt_image.int().detach().cpu().numpy()
#         pre_image=pre_image.int().detach().cpu().numpy()
#         for i in range(gt_image.shape[0]):
#             tn, fp, fn, tp = confusion_matrix(gt_image[i].reshape(-1), pre_image[i].reshape(-1)).ravel()

#             Acc_F = (tp + tn) / (tp + tn + fp + fn)
#             IoU_F = (tp) / (tp + fp + fn)
#             Dice_F =  (2 * tp) / ((2 * tp) + fp + fn)

#             tn, fp, fn, tp = confusion_matrix(np.invert(gt_image[i]).reshape(-1), np.invert(pre_image[i]).reshape(-1)).ravel()

#             Acc_B = (tp + tn) / (tp + tn + fp + fn)
#             IoU_B = (tp) / (tp + fp + fn)
#             Dice_B =  (2 * tp) / ((2 * tp) + fp + fn)

#             Acc = 0.5 * (Acc_F+Acc_B)
#             IoU = 0.5 * (IoU_B+IoU_F)
#             Dice = 0.5 * (Dice_B+Dice_F)

#             self.acc.append(Acc)
#             self.iou.append(IoU)
#             self.dice.append(Dice)

#     def reset(self):
#         self.acc = []
#         self.iou = []
#         self.dice = []

def trainer_s(end_epoch,epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,writer,logger,loss_function):
    torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model=model.to('cuda')
    model.train()

    loss_total = utils.AverageMeter()
    loss_ce_total = utils.AverageMeter()
    loss_dice_total = utils.AverageMeter()
    loss_att_total = utils.AverageMeter()

    Eval = Evaluator()

    mIOU = 0.0
    Dice = 0.0

    total_batchs = len(dataloader['train'])
    loader = dataloader['train'] 
    # pos_weight = dataloader['pos_weight']
    dice_loss = DiceLoss()
    ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=None)
    # att_loss = M_loss()


    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    # if epoch_num % 10==0:   
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr'] * 0.1   

    # scaler = torch.cuda.amp.GradScaler()
    # for batch_idx, (inputs, targets) in enumerate(loader):

    #     inputs, targets = inputs.to(device), targets.to(device)
    #     targets = targets.float()

    #     inputs = inputs.float()
    #     alpha = 0.6
    #     with torch.autocast(device_type=device, dtype=torch.float16):
    #         outputs = model(inputs)
    #         if type(outputs)==tuple:
    #             loss_ce = ce_loss(outputs[0], targets.unsqueeze(dim=1)) + ce_loss(outputs[1], targets.unsqueeze(dim=1)) + ce_loss(outputs[2], targets.unsqueeze(dim=1)) + ce_loss(outputs[3], targets.unsqueeze(dim=1)) 
    #             loss_dice = dice_loss(inputs=outputs[0], targets=targets) + dice_loss(inputs=outputs[1], targets=targets) + dice_loss(inputs=outputs[2], targets=targets) + dice_loss(inputs=outputs[3], targets=targets) 
    #             loss_att = 0.0
    #             loss = loss_ce + loss_dice 
    #         else:
    #             loss_ce = ce_loss(outputs, targets.unsqueeze(dim=1))
    #             loss_dice = dice_loss(inputs=outputs, targets=targets)
    #             loss_att = 0.0
    #             loss = loss_ce + loss_dice + loss_att

    #     scaler.scale(loss).backward()
    #     scaler.step(optimizer)
    #     scaler.update()
    #     optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()
        inputs = inputs.float()

        outputs = model(inputs)
        # outputs, outputs_t, e1, e2, e3, e1_t, e2_t, e3_t = model(inputs)

        if type(outputs)==tuple:
            loss_ce   = ce_loss(outputs[0], targets.unsqueeze(dim=1)) + ce_loss(outputs[1], targets.unsqueeze(dim=1)) + ce_loss(outputs[2], targets.unsqueeze(dim=1)) 
            loss_dice = dice_loss(inputs=outputs[0], targets=targets) + dice_loss(inputs=outputs[1], targets=targets) + dice_loss(inputs=outputs[2], targets=targets) 
            loss_att  = 0.0
            loss = loss_ce + loss_dice + loss_att
            # loss = structure_loss(outputs[0], targets.unsqueeze(dim=1)) + structure_loss(outputs[1], targets.unsqueeze(dim=1)) 
        else:
            # loss_ce   = ce_loss(outputs, targets.unsqueeze(dim=1)) 
            # loss_dice = dice_loss(inputs=outputs, targets=targets)

            loss_ce   = ce_loss(outputs, targets.unsqueeze(dim=1))
            loss_dice = dice_loss(inputs=outputs, targets=targets)
            loss_att  = 0.0
            loss = loss_ce + loss_dice + loss_att

        # lr_ = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9     
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_
        # iter_num = iter_num + 1   

        # lr_ = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_

        # iter_num = iter_num + 1   

        # iter_num = iter_num + 1 
        # if iter_num % (total_batchs*3)==0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * 0.5   

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()   

        loss_total.update(loss)
        loss_ce_total.update(loss_ce)
        loss_dice_total.update(loss_dice)
        loss_att_total.update(loss_att)

        targets = targets.long()

        if type(outputs)==tuple:
            predictions = torch.round(0.5 * (torch.sigmoid(torch.squeeze(outputs[0], dim=1)) + torch.sigmoid(torch.squeeze(outputs[1], dim=1))))

        else:
            predictions = torch.round(torch.sigmoid(torch.squeeze(outputs, dim=1)))

        # predictions = torch.round(torch.squeeze(outputs, dim=1))
        # predictions = torch.round(torch.sigmoid(torch.squeeze(outputs, dim=1)))
        Eval.add_batch(gt_image=targets,pre_image=predictions)
        # accuracy.update(Eval.Pixel_Accuracy())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            # suffix=f'loss = {loss_total.avg:.4f} , loss_ce = {loss_ce_total.avg:.4f} , loss_dice = {loss_dice_total.avg:.4f} , Dice = {Eval.Dice()*100.0:.2f} , IoU = {Eval.Mean_Intersection_over_Union()*100.0:.2f} , Pixel Accuracy = {Eval.Pixel_Accuracy()*100.0:.2f}',          
            suffix=f'loss = {loss_total.avg:.4f} , loss_att = {loss_att_total.avg:.4f} , Dice = {Eval.Dice()*100.0:.2f} , IoU = {Eval.Mean_Intersection_over_Union()*100.0:.2f} , Pixel Accuracy = {Eval.Pixel_Accuracy()*100.0:.2f}',          
            bar_length=45
        )  
  
    acc =  Eval.Pixel_Accuracy() * 100.0
    mIOU = Eval.Mean_Intersection_over_Union() * 100.0

    Dice = Eval.Dice() * 100.0
    Dice_per_class = Dice * 100.0

    # if lr_scheduler is not None:
    #     lr_scheduler.step()        
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss = {loss_total.avg:.4f} , Dice = {Dice:.2f} , IoU = {mIOU:.2f} , Pixel Accuracy = {acc:.2f} , lr = {optimizer.param_groups[0]["lr"]}')
    valid_s(end_epoch,epoch_num,model,dataloader,device,ckpt,num_class,writer,logger,optimizer)

import numpy as np
import torch
import torch.nn.functional as F

class Evaluator_New(object):
    ''' For using this evaluator target and prediction
        dims should be [B,H,W] '''
    def __init__(self):
        self.reset()

    def Pixel_Accuracy(self):
        Acc = torch.tensor(self.acc).mean()
        return Acc

    def Mean_Intersection_over_Union(self,per_class=False,show=False):
        IoU = torch.tensor(self.iou).mean()
        return IoU

    def Dice(self,per_class=False,show=False):
        Dice = torch.tensor(self.dice).mean()
        return Dice

    def get_accuracy(self,SR,GT,threshold=0.5):
        SR = SR > threshold
        GT = GT == torch.max(GT)
        corr = torch.sum(SR==GT)
        tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
        acc = float(corr)/float(tensor_size)
        return acc

    def get_sensitivity(self,SR,GT,threshold=0.5):
        # Sensitivity == Recall
        SE = 0
        SR = SR > threshold
        GT = GT == torch.max(GT)
            # TP : True Positive
            # FN : False Negative
        TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
        FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
        SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
        return SE

    def get_specificity(self,SR,GT,threshold=0.5):
        SP = 0
        SR = SR > threshold
        GT = GT == torch.max(GT)
            # TN : True Negative
            # FP : False Positive
        TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
        FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
        SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
        return SP

    def get_precision(self,SR,GT,threshold=0.5):
        PC = 0
        SR = SR > threshold
        GT = GT== torch.max(GT)
            # TP : True Positive
            # FP : False Positive
        TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
        FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
        PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
        return PC

    def iou_score(self,output, target):
        smooth = 1e-5

        if torch.is_tensor(output):
            output = torch.sigmoid(output).data.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5
        
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        dice = (2* iou) / (iou+1)
        
        output_ = torch.tensor(output_)
        target_ = torch.tensor(target_)

        SE  = self.get_sensitivity(output_,target_,threshold=0.5)
        PC  = self.get_precision(output_,target_,threshold=0.5)
        SP  = self.get_specificity(output_,target_,threshold=0.5)
        ACC = self.get_accuracy(output_,target_,threshold=0.5)

        F1  = 2*SE*PC/(SE+PC + 1e-6)

        return iou, dice, SE, PC, F1, SP, ACC

    def add_batch(self, gt_image, pre_image):
        gt_image = gt_image.int()
        pre_image = pre_image.int()
        
        for i in range(gt_image.shape[0]):
            iou, dice, SE, PC, F1, SP, ACC = self.iou_score(pre_image, gt_image)
            self.acc.append(ACC)
            self.iou.append(iou)
            self.dice.append(dice)

    def reset(self):
        self.acc  = []
        self.iou  = []
        self.dice = []
        
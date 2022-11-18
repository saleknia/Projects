import utils
from utils import cosine_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from utils import print_progress
import torch.nn.functional as F
import warnings
from utils import focal_loss
from torch.autograd import Variable
from torch.nn.functional import mse_loss as MSE
from utils import importance_maps_distillation as imd
from valid_s import valid_s
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")

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

def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))


def importance_maps_distillation(s, t, exp=2):
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    if s.shape[2] != t.shape[2]:
        t = F.interpolate(t, s.size()[-2:], mode='bilinear')
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()

def attention_loss(up4, up3, up2, up1):
    loss = 0.0
    loss = loss + importance_maps_distillation(s=up3, t=up4.detach().clone())
    loss = loss + importance_maps_distillation(s=up2, t=up3.detach().clone())
    loss = loss + importance_maps_distillation(s=up1, t=up2.detach().clone())
    return loss

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

class Evaluator(object):
    ''' For using this evaluator target and prediction
        dims should be [B,H,W] '''
    def __init__(self):
        self.reset()
        
    def Pixel_Accuracy(self):
        Acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        Acc = torch.tensor(Acc)
        return Acc

    def Mean_Intersection_over_Union(self,per_class=False,show=False):
        IoU = (self.tp) / (self.tp + self.fp + self.fn)
        IoU = torch.tensor(IoU)
        return IoU

    def Dice(self,per_class=False,show=False):
        Dice =  (2 * self.tp) / ((2 * self.tp) + self.fp + self.fn)
        Dice = torch.tensor(Dice)
        return Dice

    def add_batch(self, gt_image, pre_image):
        gt_image=gt_image.int().detach().cpu().numpy()
        pre_image=pre_image.int().detach().cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(gt_image.reshape(-1), pre_image.reshape(-1)).ravel()
        self.tn = self.tn + tn
        self.fp = self.fp + fp
        self.fn = self.fn + fn
        self.tp = self.tp + tp  

    def reset(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

def trainer_s(end_epoch,epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,writer,logger,loss_function):
    torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model=model.to('cuda')
    model.train()

    loss_total = utils.AverageMeter()
    loss_ce_total = utils.AverageMeter()
    loss_dice_total = utils.AverageMeter()
    loss_kd_total = utils.AverageMeter()
    loss_att_total = utils.AverageMeter()

    Eval = Evaluator()

    mIOU = 0.0
    Dice = 0.0

    # accuracy = utils.AverageMeter()
    # accuracy_eval = utils.AverageMeter()

    dice_loss = DiceLoss()
    # ce_loss = CrossEntropyLoss()
    ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4], device=device))
    # ce_loss = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    # kd_loss = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')

    total_batchs = len(dataloader['train'])
    loader = dataloader['train'] 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()

        # targets = targets + 1.0
        # targets[targets==2.0] = 0.0
        
        inputs = inputs.float()
        outputs = model(inputs)
        # outputs, up4 = model(inputs)
        # outputs, up4, up3, up2, up1, e5, e4, e3, e2, e1 = model(inputs)


        # soft_label = 0.5 * (outputs.detach().clone() + targets.long().unsqueeze(dim=1))
        loss_kd = 0.0
        loss_att = 0.0

        loss_ce = ce_loss(outputs, targets.unsqueeze(dim=1)) 
        # loss_ce = ce_loss(outputs, soft_label) 
        loss_dice = dice_loss(inputs=outputs, targets=targets)
        loss = 0.6 * loss_ce + 0.4 * loss_dice
        # loss = loss_ce

        # iter_num = iter_num + 1 
        # if iter_num % (total_batchs*10)==0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * 0.5
 
        # lr_ = 0.001 * (1.0 - iter_num / max_iterations) ** 0.9

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_

        # iter_num = iter_num + 1        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total.update(loss)
        loss_ce_total.update(loss_ce)
        loss_dice_total.update(loss_dice)
        loss_kd_total.update(loss_kd)
        loss_att_total.update(loss_att)

        targets = targets.long()

        # predictions = torch.round(torch.squeeze(outputs, dim=1))
        predictions = torch.round(torch.sigmoid(torch.squeeze(outputs, dim=1)))
        Eval.add_batch(gt_image=targets,pre_image=predictions)
        # accuracy.update(Eval.Pixel_Accuracy())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'Dice_loss = {0.4*loss_dice_total.avg:.4f} , CE_loss = {0.6*loss_ce_total.avg:.4f} , Dice = {Eval.Dice()*100.0:.2f} , IoU = {Eval.Mean_Intersection_over_Union()*100.0:.2f} , Pixel Accuracy = {Eval.Pixel_Accuracy()*100.0:.2f}',          
            bar_length=45
        )  
  
    # acc = 100*accuracy.avg
    acc =  Eval.Pixel_Accuracy() * 100.0
    mIOU = Eval.Mean_Intersection_over_Union() * 100.0

    Dice = Eval.Dice() * 100.0
    Dice_per_class = Dice * 100.0
    # Dice,Dice_per_class = Eval.Dice(per_class=True)
    # Dice,Dice_per_class = 100*Dice,100*Dice_per_class

    if lr_scheduler is not None:
        lr_scheduler.step()        
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss = {loss_total.avg:.4f} , Dice = {Dice:.2f} , IoU = {mIOU:.2f} , Pixel Accuracy = {acc:.2f} , lr = {optimizer.param_groups[0]["lr"]}')
    valid_s(end_epoch,epoch_num,model,dataloader['valid'],device,ckpt,num_class,writer,logger,optimizer)




import utils
from utils import cosine_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from utils import atten_loss,prototype_loss,IM_loss,M_loss, disparity, disparity_loss, discriminate
from tqdm import tqdm
from utils import print_progress
import torch.nn.functional as F
import warnings
from utils import focal_loss
from torch.autograd import Variable
from torch.nn.functional import mse_loss as MSE
from utils import importance_maps_distillation as imd
from valid_s import valid_s
warnings.filterwarnings("ignore")

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
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
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = preds_T.permute(0,2,3,1).contiguous().view(-1,C)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss

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

    Eval = utils.Evaluator(num_class=2)

    mIOU = 0.0
    Dice = 0.0

    accuracy = utils.AverageMeter()
    accuracy_eval = utils.AverageMeter()

    dice_loss = DiceLoss()
    # ce_loss = CrossEntropyLoss()
    ce_loss = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    kd_loss = CriterionPixelWise()
    att_loss = discriminate()

    total_batchs = len(dataloader['train'])
    loader = dataloader['train'] 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()
        inputs = inputs.float()
        outputs = model(inputs)
        # outputs, up4 = model(inputs)
        # outputs, up4, up3, up2, up1, e5, e4, e3, e2, e1 = model(inputs)


        # soft_label = 0.5 * (torch.nn.functional.softmax(outputs) + torch.nn.functional.one_hot(targets.long(), num_classes=2).permute(0,3,1,2))
        # loss_kd = kd_loss(preds_S=outputs, preds_T=soft_label)
        # loss_att = att_loss(masks=targets, outputs=up4)
        loss_kd = 0.0
        loss_att = 0.0


        loss_ce = ce_loss(outputs, targets.unsqueeze(dim=1)) 
        loss_dice = dice_loss(inputs=outputs, targets=targets)
        loss = 0.6 * loss_ce + 0.4 * loss_dice

 
        lr_ = 0.001 * (1.0 - iter_num / max_iterations) ** 0.9

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total.update(loss)
        loss_ce_total.update(loss_ce)
        loss_dice_total.update(loss_dice)
        loss_kd_total.update(loss_kd)
        loss_att_total.update(loss_att)

        targets = targets.long()

        predictions = torch.round(torch.squeeze(outputs, dim=1))
        # predictions = torch.argmax(input=outputs, dim=1).long()
        Eval.add_batch(gt_image=targets,pre_image=predictions)
        accuracy.update(Eval.Pixel_Accuracy())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'Dice_loss = {0.5*loss_dice_total.avg:.4f} , CE_loss = {0.5*loss_ce_total.avg:.4f}, kd_loss = {loss_kd_total.avg:.4f}, att_loss = {loss_att_total.avg:.4f} , Dice = {Eval.Dice()*100:.2f} , Pixel Accuracy: {accuracy.avg*100:.2f}',          
            bar_length=45
        )  
  
    acc = 100*accuracy.avg
    mIOU = 100*Eval.Mean_Intersection_over_Union()

    Dice = Eval.Dice() * 100.0
    Dice_per_class = Dice * 100.0
    # Dice,Dice_per_class = Eval.Dice(per_class=True)
    # Dice,Dice_per_class = 100*Dice,100*Dice_per_class

    if lr_scheduler is not None:
        lr_scheduler.step()        
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss: {loss_total.avg:.4f} , Dice: {Dice:.2f} , mIoU: {mIOU:.2f} , Pixel Accuracy: {acc:.2f} , Pixel Accuracy Eval: {accuracy_eval.avg*100:.2f} , lr: {optimizer.param_groups[0]["lr"]}')
    valid_s(end_epoch,epoch_num,model,dataloader['valid'],device,ckpt,num_class,writer,logger,optimizer)




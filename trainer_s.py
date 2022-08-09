import utils
from utils import cosine_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss,atten_loss,prototype_loss,IM_loss, disparity
from tqdm import tqdm
from utils import print_progress
import torch.nn.functional as F
import warnings
from valid_s import valid_s
import numpy as np
warnings.filterwarnings("ignore")

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

    def MIOU_out(self,per_class=False,show=False):
        numerator = np.diag(self.confusion_matrix) 
        denominator = (np.sum(self.confusion_matrix,axis=1) + np.sum(self.confusion_matrix, axis=0)-np.diag(self.confusion_matrix))
        if show:
            # print('Intersection Pixels: ',numerator)
            # print('Union Pixels: ',denominator)
            print('MIoU Per Class: ',numerator/denominator)
        class_MIoU = numerator/denominator
        class_MIoU = class_MIoU[0:self.num_class-1]
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
        class_Dice = class_Dice[0:self.num_class]
        Dice = np.nanmean(class_Dice)
        Dice = torch.tensor(Dice)
        class_Dice = torch.tensor(class_Dice)
        if per_class:
            return Dice,class_Dice
        else:
            return Dice


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

def trainer_s(end_epoch,epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,writer,logger,loss_function,weight):
    torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model=model.to('cuda')
    model.train()

    loss_total = utils.AverageMeter()
    loss_ce_total = utils.AverageMeter()
    loss_disparity_total = utils.AverageMeter()

    Eval = Evaluator(num_class=num_class+1)

    mIOU = 0.0

    accuracy = utils.AverageMeter()

    ce_loss = CrossEntropyLoss(weight=weight, ignore_index=11)
    disparity_loss = disparity()

    total_batchs = len(dataloader['train'])
    loader = dataloader['train'] 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()
        outputs, up4, up3, up2, up1 = model(inputs)

        targets = targets.long()
        predictions = torch.argmax(input=outputs,dim=1).long()
        overlap = (predictions==targets).float()
        t_masks = targets * overlap
        targets = targets.float()

        loss_ce = ce_loss(outputs, targets[:].long())
        loss_disparity = disparity_loss(masks=targets, t_masks=t_masks, up4=up4, up3=up3, up2=up2, up1=up1) * 0.1
        loss = loss_ce + loss_disparity
 
        lr_ = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total.update(loss)
        loss_ce_total.update(loss_ce)
        loss_disparity_total.update(loss_disparity)

        targets = targets.long()
        predictions = torch.argmax(input=outputs,dim=1).long()
        Eval.add_batch(gt_image=targets,pre_image=predictions)

        accuracy.update(Eval.Pixel_Accuracy())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'loss = {loss_total.avg:.4f} , loss_ce = {loss_ce_total.avg:.4f} , loss_dis = {loss_disparity_total.avg:.4f} , MIOU = {Eval.MIOU_out()*100:.2f}',         
            bar_length=45
        )  
  
    acc = 100*accuracy.avg
    mIOU,mIOU_per_class = Eval.MIOU_out(per_class=True)
    mIOU,mIOU_per_class = 100*mIOU,100*mIOU_per_class

    if lr_scheduler is not None:
        lr_scheduler.step()        
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss: {loss_total.avg:.4f} , mIoU: {mIOU:.2f} , Pixel Accuracy: {acc:.2f}, lr: {optimizer.param_groups[0]["lr"]}')
    valid_s(end_epoch,epoch_num,model,dataloader['valid'],device,ckpt,num_class,writer,logger,optimizer,weight)




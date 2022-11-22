import utils
import torch
import numpy as np
from utils import print_progress
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from utils import hd95
import warnings
from medpy import metric
import medpy
import numpy as np
import torch.nn.functional as F
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
        # IoU = (self.tp) / (self.tp + self.fp + self.fn)
        IoU = self.TP / (self.FPN - self.TP)
        IoU = torch.tensor(IoU)
        return IoU

    def Dice(self,per_class=False,show=False):
        # Dice =  (2 * self.tp) / ((2 * self.tp) + self.fp + self.fn)
        Dice = (2 * self.TP) / self.FPN
        Dice = torch.tensor(Dice)
        return Dice

    def add_batch(self, gt_image, pre_image):
        gt_image=gt_image.int().detach().cpu().numpy()
        pre_image=pre_image.int().detach().cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(gt_image.reshape(-1), pre_image.reshape(-1)).ravel()
        TP = np.sum(pre_image*gt_image)
        FPN = np.sum(pre_image) + np.sum(gt_image)
        self.tn = self.tn + tn
        self.fp = self.fp + fp
        self.fn = self.fn + fn
        self.tp = self.tp + tp  
        self.TP = self.TP + TP
        self.FPN = self.FPN + FPN

    def reset(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.TP = 0
        self.FPN = 0
        
# class Evaluator(object):
#     ''' For using this evaluator target and prediction
#         dims should be [B,H,W] '''
#     def __init__(self):
#         self.reset()
        
#     def Pixel_Accuracy(self):
#         Acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
#         Acc = torch.tensor(Acc)
#         return Acc

#     def Mean_Intersection_over_Union(self,per_class=False,show=False):
#         IoU = (self.tp) / (self.tp + self.fp + self.fn)
#         IoU = torch.tensor(IoU)
#         return IoU

#     def Dice(self,per_class=False,show=False):
#         Dice =  (2 * self.tp) / ((2 * self.tp) + self.fp + self.fn)
#         Dice = torch.tensor(Dice)
#         return Dice

#     def add_batch(self, gt_image, pre_image):
#         gt_image=gt_image.int().detach().cpu().numpy()
#         pre_image=pre_image.int().detach().cpu().numpy()
#         tn, fp, fn, tp = confusion_matrix(gt_image.reshape(-1), pre_image.reshape(-1)).ravel()
#         self.tn = self.tn + tn
#         self.fp = self.fp + fp
#         self.fn = self.fn + fn
#         self.tp = self.tp + tp  

#     def reset(self):
#         self.tn = 0
#         self.fp = 0
#         self.fn = 0
#         self.tp = 0

def valid_s(end_epoch,epoch_num,model,dataloader,device,ckpt,num_class,writer,logger,optimizer):
    model=model.to(device)
    model.eval()
    loss_total = utils.AverageMeter()
    Eval = Evaluator()
    mIOU = 0.0

    total_batchs = len(dataloader['valid'])
    loader = dataloader['valid']
    pos_weight = dataloader['pos_weight']

    dice_loss = DiceLoss()
    ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            targets = targets.float()

            # targets = targets + 1.0
            # targets[targets==2.0] = 0.0

            outputs = model(inputs)

            loss_ce = ce_loss(outputs, targets.unsqueeze(dim=1))
            loss_dice = dice_loss(inputs=outputs, targets=targets)
            loss = loss_ce + loss_dice           

            # loss_ce = ce_loss(outputs, targets[:].long())
            # loss_dice = dice_loss(inputs=outputs, target=targets, softmax=True)
            # loss = 0.5 * loss_ce + 0.5 * loss_dice

            loss_total.update(loss)

            targets = targets.long()

            # predictions = torch.round(torch.squeeze(outputs, dim=1))
            predictions = torch.round(torch.sigmoid(torch.squeeze(outputs, dim=1)))
            Eval.add_batch(gt_image=targets,pre_image=predictions)

            # accuracy.update(Eval.Pixel_Accuracy())

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Valid {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss= {loss_total.avg:.4f} , Dice = {Eval.Dice()*100.0:.2f} , IoU = {Eval.Mean_Intersection_over_Union()*100.0:.2f} , Pixel Accuracy = {Eval.Pixel_Accuracy()*100.0:.2f}',
                bar_length=45
            )  

    # acc = 100*accuracy.avg
    acc =  Eval.Pixel_Accuracy() * 100.0
    mIOU = 100*Eval.Mean_Intersection_over_Union()

    Dice = Eval.Dice() * 100.0
    Dice_per_class = Dice * 100.0
    # Dice,Dice_per_class = Eval.Dice(per_class=True)
    # Dice,Dice_per_class = 100*Dice,100*Dice_per_class
   
    logger.info(f'Epoch: {epoch_num} ---> Valid , Loss = {loss_total.avg:.4f} , Dice = {Dice:.2f}  , IoU = {mIOU:.2f} , Pixel Accuracy = {acc:.2f} , lr = {optimizer.param_groups[0]["lr"]}')

    # # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=None)



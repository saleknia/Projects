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
import ttach as tta

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
        Acc = torch.tensor(np.mean(self.acc))
        return Acc

    def Mean_Intersection_over_Union(self,per_class=False,show=False):
        IoU = torch.tensor(np.mean(self.iou))
        return IoU

    def Dice(self,per_class=False,show=False):
        Dice = torch.tensor(np.mean(self.dice))
        return Dice

    def add_batch(self, gt_image, pre_image):
        gt_image=gt_image.int().detach().cpu().numpy()
        pre_image=pre_image.int().detach().cpu().numpy()
        for i in range(gt_image.shape[0]):
            tn, fp, fn, tp = confusion_matrix(gt_image[i].reshape(-1), pre_image[i].reshape(-1)).ravel()
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
    # model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    loss_total = utils.AverageMeter()
    Eval = Evaluator()
    mIOU = 0.0

    total_batchs = len(dataloader['valid'])
    loader = dataloader['valid']
    pos_weight = dataloader['pos_weight']

    dice_loss = DiceLoss()
    ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=None)
    # ce_loss = torch.nn.BCELoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            targets = targets.float()

            # targets = targets + 1.0
            # targets[targets==2.0] = 0.0

            outputs = model(inputs)

            if type(outputs)==tuple:
                loss_ce = ce_loss(outputs[0], targets.unsqueeze(dim=1))
                loss_dice = dice_loss(inputs=outputs[0], targets=targets)
                loss = loss_ce + loss_dice     
            else:
                loss_ce = ce_loss(outputs, targets.unsqueeze(dim=1))
                loss_dice = dice_loss(inputs=outputs, targets=targets)
                loss = loss_ce + loss_dice     

            # loss_ce = ce_loss(outputs, targets[:].long())
            # loss_dice = dice_loss(inputs=outputs, target=targets, softmax=True)
            # loss = 0.5 * loss_ce + 0.5 * loss_dice

            loss_total.update(loss)

            targets = targets.long()

            if type(outputs)==tuple:
                predictions = torch.round((torch.sigmoid(torch.squeeze(outputs[0], dim=1)) + torch.sigmoid(torch.squeeze(outputs[1], dim=1))) / 2.0)  

            else:
                predictions = torch.round(torch.sigmoid(torch.squeeze(outputs, dim=1)))


            # predictions = torch.round(torch.squeeze(outputs, dim=1))
            # predictions = torch.round(torch.sigmoid(torch.squeeze(outputs, dim=1)))
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
    Dice_per_class = Eval.Dice() * 100.0
    # Dice,Dice_per_class = Eval.Dice(per_class=True)
    # Dice,Dice_per_class = 100*Dice,100*Dice_per_class
   
    logger.info(f'Epoch: {epoch_num} ---> Valid , Loss = {loss_total.avg:.4f} , Dice = {Dice:.2f}  , IoU = {mIOU:.2f} , Pixel Accuracy = {acc:.2f} , lr = {optimizer.param_groups[0]["lr"]}')

    # # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=None)



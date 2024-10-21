import utils
import torch
import numpy as np
import torch.nn as nn
from utils import print_progress
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


from torchmetrics.classification import BinaryConfusionMatrix

# class Evaluator(object):
#     ''' For using this evaluator target and prediction
#         dims should be [B,H,W] '''
#     def __init__(self):
#         self.reset()
#         self.metric = BinaryConfusionMatrix().to('cuda')
#     def Pixel_Accuracy(self):
#         Acc = torch.tensor(self.acc).mean()
#         return Acc

#     def Mean_Intersection_over_Union(self,per_class=False,show=False):
#         IoU = torch.tensor(self.iou).mean()
#         return IoU

#     def Dice(self,per_class=False,show=False):
#         Dice = torch.tensor(self.dice).mean()
#         return Dice

#     def add_batch(self, gt_image, pre_image):
#         gt_image = gt_image.int()
#         pre_image = pre_image.int()
        
#         for i in range(gt_image.shape[0]):
#             tn, fp, fn, tp = self.metric(pre_image[i].reshape(-1), gt_image[i].reshape(-1)).ravel()
#             Acc_F  = (tp + tn) / (tp + tn + fp + fn)
#             IoU_F  = (tp) / (tp + fp + fn)
#             Dice_F =  (2 * tp) / ((2 * tp) + fp + fn)


#             pre_image[i] = (torch.logical_not(pre_image[i].bool())).int()
#             gt_image[i]  = (torch.logical_not(gt_image[i].bool())).int()

#             tn, fp, fn, tp = self.metric(pre_image[i].reshape(-1), gt_image[i].reshape(-1)).ravel()
#             Acc_B  = (tp + tn) / (tp + tn + fp + fn)
#             IoU_B  = (tp) / (tp + fp + fn)
#             Dice_B =  (2 * tp) / ((2 * tp) + fp + fn)

#             Acc  = 0.5 * (Acc_F  + Acc_B)
#             IoU  = 0.5 * (IoU_B  + IoU_F)
#             Dice = 0.5 * (Dice_B + Dice_F)

#             self.acc.append(Acc)
#             self.iou.append(IoU)
#             self.dice.append(Dice)

#     def reset(self):
#         self.acc = []
#         self.iou = []
#         self.dice = []

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

    def F1(self,per_class=False,show=False):
        f1 = torch.tensor(self.f1).mean()
        return f1

    def add_batch(self, gt_image, pre_image):
        gt_image = gt_image.int()
        pre_image = pre_image.int()
        
        for i in range(gt_image.shape[0]):
            tn, fp, fn, tp = self.metric(pre_image[i].reshape(-1), gt_image[i].reshape(-1)).ravel()
            Acc  = (tp + tn) / (tp + tn + fp + fn)
            IoU  = (tp) / ((tp + fp + fn) + 1e-5)
            Dice =  (2 * tp) / ((2 * tp) + fp + fn + 1e-6)
            f1   = (tp) / (tp + (0.5 * (fp + fn)) + 1e-6)

            self.acc.append(Acc)
            self.iou.append(IoU)
            self.dice.append(Dice)
            self.f1.append(f1)

    def reset(self):
        self.acc = []
        self.iou = []
        self.dice = []
        self.f1 = []


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

def iou_score(output, target):
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
    return iou, dice

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

def tester_s(end_epoch,epoch_num,model,dataloader,device,ckpt,num_class,writer,logger,optimizer,lr_scheduler,early_stopping):
    model=model.to(device)
    model.eval()

    # model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    loss_total = utils.AverageMeter()
    Eval = Evaluator()
    mIOU = 0.0
    Dice = 0.0

    DICE = AverageMeter()
    MIOU = AverageMeter()

    total_batchs = len(dataloader['test'])
    loader = dataloader['test']
    # pos_weight = dataloader['pos_weight']

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
                predictions = torch.round(torch.sigmoid(torch.squeeze(outputs[0], dim=1)) + torch.sigmoid(torch.squeeze(outputs[1], dim=1)))
                # predictions = torch.round(0.5 * (torch.sigmoid(torch.squeeze(outputs[0], dim=1)) + torch.sigmoid(torch.squeeze(outputs[1], dim=1))))
                # predictions = torch.round((torch.sigmoid(torch.squeeze(outputs[1], dim=1)) + torch.sigmoid(torch.squeeze(outputs[2], dim=1))) / 2.0)  

            else:
                predictions = torch.round(torch.sigmoid(torch.squeeze(outputs, dim=1)))

            # predictions = torch.round(torch.squeeze(outputs, dim=1))    
            # predictions = torch.round(torch.sigmoid(torch.squeeze(outputs, dim=1)))
            Eval.add_batch(gt_image=targets,pre_image=predictions)
            
            iou,dice = iou_score(torch.squeeze(outputs, dim=1), targets)
            MIOU.update(iou, inputs.size(0))
            DICE.update(dice, inputs.size(0))
            
            # accuracy.update(Eval.Pixel_Accuracy())

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Test Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss= {loss_total.avg:.4f} , Dice = {Eval.Dice()*100.0:.2f}  , IoU = {Eval.Mean_Intersection_over_Union()*100.0:.2f} , Pixel Accuracy = {Eval.Pixel_Accuracy()*100.0:.2f}',
                bar_length=45
            )  

        # acc = 100*accuracy.avg
        acc =  Eval.Pixel_Accuracy() * 100.0
        mIOU = 100*Eval.Mean_Intersection_over_Union()

        Dice = Eval.Dice() * 100.0
        Dice_per_class = Eval.Dice() * 100.0
        # Dice,Dice_per_class = Eval.Dice(per_class=True)
        # Dice,Dice_per_class = 100*Dice,100*Dice_per_class


        logger.info(f'Epoch: {epoch_num} ---> Test , Loss = {loss_total.avg:.4f} , Dice = {Dice:.2f} , IoU = {mIOU:.2f} , Pixel Accuracy = {acc:.2f}') 

        print('F1: %.4f' % (Eval.F1() * 100.0))


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

        # print(output_.shape)
        # print(target_.shape)
        
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
            iou, dice, SE, PC, F1, SP, ACC = self.iou_score(pre_image[i], gt_image[i])
            self.acc.append(ACC)
            self.iou.append(iou)
            self.dice.append(dice)

    def reset(self):
        self.acc  = []
        self.iou  = []
        self.dice = []

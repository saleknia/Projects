import utils
import torch
from utils import print_progress
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss, hd95
import warnings
from medpy import metric
import medpy
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

def tester_s(end_epoch,epoch_num,model,dataloader,device,ckpt,num_class,writer,logger,optimizer,lr_scheduler,early_stopping,weight):
    model=model.to(device)
    model.eval()
    loss_total = utils.AverageMeter()
    Eval = Evaluator(num_class=num_class+1)
    mIOU = 0.0
    accuracy = utils.AverageMeter()

    ce_loss = CrossEntropyLoss(weight=weight, ignore_index=11)

    total_batchs = len(dataloader)
    loader = dataloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)

            targets = targets.float()
            outputs = model(inputs)

            loss = ce_loss(outputs, targets[:].long())

            loss_total.update(loss)

            targets = targets.long()
            predictions = torch.argmax(input=outputs,dim=1).long()
            Eval.add_batch(gt_image=targets,pre_image=predictions)

            accuracy.update(Eval.Pixel_Accuracy())

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Valid {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss= {loss_total.avg:.4f} , Accuracy= {accuracy.avg*100:.2f} , mIoU= {Eval.MIOU_out()*100:.2f}',
                bar_length=45
            )  

        acc = 100*accuracy.avg
        mIOU,mIOU_per_class = Eval.MIOU_out(per_class=True)
        mIOU,mIOU_per_class = 100*mIOU,100*mIOU_per_class
        mIOU_per_class = [x.item() for x in mIOU_per_class]


        logger.info(f'Epoch: {epoch_num} ---> Valid , Loss: {loss_total.avg:.4f} , mIoU: {mIOU:.2f} , Pixel Accuracy: {acc:.2f}') 
        logger.info(f'MIOU Per Class: {mIOU_per_class}') 


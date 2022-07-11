# Instaling Libraries
import os
import copy
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
import pickle
import argparse
from torch.backends import cudnn
# from albumentations.pytorch.transforms import ToTensorV2
# import albumentations as A
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import random_split
from tqdm.notebook import tqdm
import torch.optim as optim
from models.UNet import UNet
from models.UNet_loss import UNet_loss
from models.UNet_plus import NestedUNet
from models.UNet_plus_loss import NestedUNet_loss
from models.att_unet import AttentionUNet
from models.att_unet_loss import AttentionUNet_loss
from models.multi_res_unet import MultiResUnet
from models.U import U
from models.U_loss import U_loss
from models.ERFNet import ERFNet
from models.ERFNet_loss import ERFNet_loss
from models.multi_res_unet_loss import MultiResUnet_loss
from models.UCTransNet import UCTransNet
from models.GT_UNet import GT_U_Net
from models.ENet import ENet
from models.Mobile_netV2 import Mobile_netV2
from models.Mobile_netV2_loss import Mobile_netV2_loss
from models.Fast_SCNN import Fast_SCNN
from models.Fast_SCNN_loss import Fast_SCNN_loss
from models.ESPNet import ESPNet
from models.ESPNet_loss import ESPNet_loss
from models.DABNet import DABNet
from models.DABNet_loss import DABNet_loss
from models.ENet_loss import ENet_loss
from models.UCTransNet_GT import UCTransNet_GT
from models.GT_CTrans import GT_CTrans
from utils import print_progress
import utils
from utils import color
from utils import Save_Checkpoint
from trainer import trainer
from tester import tester
from dataset import COVID_19,Synapse_dataset,RandomGenerator,ValGenerator,ACDC,CT_1K,SSL
from utils import DiceLoss,atten_loss,prototype_loss,prototype_loss_kd
from config import *
from tabulate import tabulate
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss

# from testing import inference
# from testingV2 import inferenceV2
import warnings
warnings.filterwarnings('ignore')

lr = 0.01

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
            print('MIoU Per Class: ',numerator/denominator)
        class_MIoU = numerator/denominator
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
            print('Dice Per Class: ',numerator/denominator)
        class_Dice = numerator/denominator
        Dice = np.nanmean(class_Dice)
        Dice = torch.tensor(Dice)
        class_Dice = torch.tensor(class_Dice)
        if per_class:
            return Dice,class_Dice
        else:
            return Dice

def main(args):

    train_tf = transforms.Compose([RandomGenerator(output_size=[IMAGE_HEIGHT, IMAGE_WIDTH])])
    val_tf = ValGenerator(output_size=[IMAGE_HEIGHT, IMAGE_WIDTH])



# LOAD_MODEL

    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    num_parameters = utils.count_parameters(model)

    model_table = tabulate(
        tabular_data=[['UNet_Base', f'{num_parameters:.2f} M', DEVICE]],
        headers=['Builded Model', '#Parameters', 'Device'],
        tablefmt="fancy_grid"
        )
    logger.info(model_table)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)


    train_dataset = SSL(split='train', joint_transform=train_tf)

    train_loader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            worker_init_fn=worker_init,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY,
                            drop_last=True,
                            )
    data_loader={'train':train_loader}


    if args.train=='True':
        logger.info(50*'*')
        logger.info('Training Phase')
        logger.info(50*'*')

        for epoch in range(1,NUM_EPOCHS+1):
            print(f'Epoch: {epoch} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')

            model=model.to(DEVICE)
            model.train()

            loss_total = utils.AverageMeter()
            loss_dice_total = utils.AverageMeter()
            loss_ce_total = utils.AverageMeter()
            Eval = Evaluator(num_class=1)
            accuracy = utils.AverageMeter()

            mIOU = 0.0
            Dice = 0.0


            ce_loss = CrossEntropyLoss()
            dice_loss = DiceLoss(1)

            total_batchs = len(data_loader['train'])
            loader = data_loader['train'] 

            base_iter = (epoch-1) * total_batchs
            iter_num = base_iter
            max_iterations = NUM_EPOCHS * total_batchs

            for batch_idx, (inputs, targets) in enumerate(loader):

                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                targets = targets.float()
                outputs = model(inputs)
                loss_ce = ce_loss(outputs, targets[:].long())
                loss_dice = dice_loss(outputs, targets, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice 

                lr_ = lr * (1.0 - iter_num / max_iterations) ** 0.9

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1        
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                loss_total.update(loss)
                loss_dice_total.update(loss_dice)
                loss_ce_total.update(loss_ce)
                
                targets = targets.long()

                predictions = torch.argmax(input=outputs,dim=1).long()
                Eval.add_batch(gt_image=targets,pre_image=predictions)

                accuracy.update(Eval.Pixel_Accuracy())

                print_progress(
                    iteration=batch_idx+1,
                    total=total_batchs,
                    prefix=f'Train {epoch} Batch {batch_idx+1}/{total_batchs} ',
                    suffix=f'Dice_loss = {0.5*loss_dice_total.avg:.4f} , CE_loss = {0.5*loss_ce_total.avg:.4f} , Dice = {Eval.Dice()*100:.2f}',          
                    bar_length=45
                )  
  
            acc = 100*accuracy.avg
            mIOU = 100*Eval.Mean_Intersection_over_Union()
            Dice,Dice_per_class = Eval.Dice(per_class=True)
            Dice,Dice_per_class = 100*Dice,100*Dice_per_class
     
        
            logger.info(f'Epoch: {epoch} ---> Train , Loss: {loss_total.avg:.4f} , mIoU: {mIOU:.2f} , Dice: {Dice:.2f} , Pixel Accuracy: {acc:.2f}, lr: {optimizer.param_groups[0]["lr"]}')


parser = argparse.ArgumentParser()
parser.add_argument('--inference', type=str,default='False')
parser.add_argument('--train', type=str,default='True')
args = parser.parse_args()

def worker_init(worker_id):
    random.seed(SEED + worker_id)

if __name__ == "__main__":
    
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(SEED)    
    np.random.seed(SEED)  
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) 


    main(args)
    
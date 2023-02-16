import utils
import torch
from utils import print_progress
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss, hd95
import warnings
from medpy import metric
import medpy
import numpy as np
import pickle
from utils import proto
from config import class_index
warnings.filterwarnings("ignore")
import ttach as tta


def tester(end_epoch,epoch_num,model,dataloader,device,ckpt,num_class,writer,logger,optimizer,lr_scheduler,early_stopping):
    model.eval()

    loss_total = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    ce_loss = CrossEntropyLoss()
    total_batchs = len(dataloader)
    loader = dataloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)

            targets = targets.float()

            # targets[targets!=class_index] = 10.0
            # targets[targets==class_index] = 1.00
            # targets[targets==10.0]        = 0.00

            outputs = model(inputs)

            loss_ce = ce_loss(outputs, targets[:].long())
            loss = loss_ce
            loss_total.update(loss)

            targets = targets.long()

            predictions = torch.argmax(input=outputs,dim=1).long()
            accuracy.update(torch.sum(targets==predictions)/torch.sum(targets==targets))

            # if 0.0 < torch.sum(targets==0.0):          
            #     accuracy.update(torch.sum((targets+predictions)==0.0)/torch.sum(targets==0.0))

            # if 0.0 < torch.sum(targets):
            #     accuracy.update(torch.sum((targets+predictions)==2.0)/torch.sum(targets))

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Test {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss= {loss_total.avg:.4f} , Accuracy= {accuracy.avg*100:.2f} ',
                bar_length=45
            )  

        acc = 100*accuracy.avg

        logger.info(f'Epoch: {epoch_num} ---> Test , Loss: {loss_total.avg:.4f} , Accuracy: {acc:.2f}') 




import utils
from utils import cosine_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing.pool import Pool
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss,atten_loss,prototype_loss,IM_loss,M_loss
from tqdm import tqdm
from utils import print_progress
import torch.nn.functional as F
import warnings
from utils import focal_loss
from torch.autograd import Variable
from torch.nn.functional import mse_loss as MSE
from utils import importance_maps_distillation as imd
import os
warnings.filterwarnings("ignore")

def disparity(labels, outputs):
    B, N = outputs.shape
    unique = torch.unique(labels)
    prototypes = torch.zeros(len(unique), N, device='cuda')
    for i,p in enumerate(unique):
        prototypes[i] = torch.mean(outputs[labels==p],dim=0)
    prototypes = torch.unsqueeze(prototypes, dim=0)
    distances = torch.cdist(prototypes, prototypes, p=2.0)
    loss = 1.0 / torch.mean(distances)
    return loss

def trainer(end_epoch,epoch_num,model,teacher_model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,writer,logger,loss_function):
    torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')

    if teacher_model is not None:
        teacher_model=teacher_model.to(device)
        teacher_model.eval()

    model=model.to(device)
    model.train()

    loss_total = utils.AverageMeter()
    loss_ce_total = utils.AverageMeter()
    loss_disparity_total = utils.AverageMeter()

    accuracy = utils.AverageMeter()

    ce_loss = CrossEntropyLoss()
    ##################################################################

    total_batchs = len(dataloader)
    loader = dataloader 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        targets = targets.float()

        # outputs = model(inputs)
        outputs, features = model(inputs)

        predictions = torch.argmax(input=outputs,dim=1).long()
        accuracy.update(torch.sum(targets==predictions)/torch.sum(targets==targets))

        if teacher_model is not None:
            with torch.no_grad():
                outputs_t, up1_t, up2_t, up3_t, up4_t, x1_t, x2_t, x3_t, x4_t, x5_t = teacher_model(inputs,multiple=True)

        loss_ce = ce_loss(outputs, targets.long())
        loss_disparity = 20.0 * disparity(labels=targets, outputs=features)
        # loss_disparity = 0.0
        ###############################################
        # loss = loss_ce
        loss = loss_ce + loss_disparity
        ###############################################

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
        ###############################################
        targets = targets.long()

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'CE_loss = {loss_ce_total.avg:.4f} , disparity_loss = {loss_disparity_total.avg:.4f} , Accuracy = {100 * accuracy.avg:.4f}',          
            bar_length=45
        )  
  
    acc = 100*accuracy.avg

    if lr_scheduler is not None:
        lr_scheduler.step()        
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss_CE : {loss_ce_total.avg:.4f} , Loss_disparity : {loss_disparity_total.avg:.4f} , Accuracy : {acc:.2f} , lr: {optimizer.param_groups[0]["lr"]}')

    # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=acc, acc_per_class=acc, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
    if ckpt is not None:
        ckpt.save_last(acc=acc, acc_per_class=acc, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)






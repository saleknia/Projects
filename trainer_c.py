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

def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.05
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss

def loss_kd_regularization(outputs, labels):
    """
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    alpha = params.reg_alpha
    T = params.reg_temperature
    correct_prob = 0.99    # the probability for correct class in u(k)
    loss_CE = F.cross_entropy(outputs, labels)
    K = outputs.size(1)

    teacher_soft = torch.ones_like(outputs).cuda()
    teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
    for i in range(outputs.shape[0]):
        teacher_soft[i ,labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1))*params.multiplier

    KD_loss = (1. - alpha)*loss_CE + alpha*loss_soft_regu

    return KD_loss

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
    if teacher_model is not None:
        ce_loss = CrossEntropyLoss(reduce=False)
    else:
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

        outputs = model(inputs)

        predictions = torch.argmax(input=outputs,dim=1).long()
        accuracy.update(torch.sum(targets==predictions)/torch.sum(targets==targets))

        if teacher_model is not None:
            with torch.no_grad():
                outputs_t = teacher_model(inputs)
                weights = F.cross_entropy(outputs_t, targets.long(), reduce=False)
                weights = 1.0 + (weights / weights.max())
                weights = weights.detach()

        if teacher_model is not None:
            loss_ce = ce_loss(outputs, targets.long()) * weights
            loss_ce = torch.mean(loss_ce)
        else:
            loss_ce = ce_loss(outputs, targets.long())

        loss_disparity = 0.0
        ###############################################
        loss = loss_ce
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






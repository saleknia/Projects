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
import numpy as np
from config import class_index
warnings.filterwarnings("ignore")

general_labels = np.load('/content/UNet_V2/labels.npy')

def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.zeros(N, C).to('cuda')
    g_labels = torch.tensor(general_labels).to('cuda')

    for i in range(len(labels)):
        smoothed_labels[i] = g_labels[labels[i]]

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss


# def loss_label_smoothing(outputs, labels):
#     """
#     loss function for label smoothing regularization
#     """
#     alpha = 0.4
#     N = outputs.size(0)  # batch_size
#     C = outputs.size(1)  # number of classes
#     smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1)).cuda()
#     smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

#     log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
#     loss = -torch.sum(log_prob * smoothed_labels) / N

#     return loss


def loss_kd_regularization(outputs, labels):
    """
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    alpha = 0.4
    T = 20
    correct_prob = 0.99    # the probability for correct class in u(k)
    loss_CE = F.cross_entropy(outputs, labels)
    K = outputs.size(1)

    teacher_soft = torch.ones_like(outputs).cuda()
    teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
    for i in range(outputs.shape[0]):
        teacher_soft[i ,labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1))

    KD_loss = (1. - alpha)*loss_CE + alpha*loss_soft_regu

    return KD_loss


class FSP(nn.Module):
	'''
	A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
	http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
	'''
	def __init__(self):
		super(FSP, self).__init__()

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.fsp_matrix(fm_s,fm_s), self.fsp_matrix(fm_t,fm_t))

		return loss

	def fsp_matrix(self, fm1, fm2):
		if fm1.size(2) > fm2.size(2):
			fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

		fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
		fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

		fsp = torch.bmm(fm1, fm2) / fm1.size(2)

		return fsp

def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))

def intra_fd(f_s):
    sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
    f_s = torch.index_select(f_s, 1, indices_s)
    intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
    return intra_fd_loss

def importance_maps_distillation(s, t, exp=4):
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
        s = F.interpolate(s, t.size()[-2:], mode='bilinear')
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()

# def distillation(outputs, labels):
#     unique = torch.unique(labels)
#     temp   = torch.zeros((len(unique),1280), device='cuda')
#     for i, v in enumerate(unique):
#         temp[i] = torch.mean(outputs[labels==v], dim=0)
#     distances = torch.cdist(temp, temp, p=2.0)
#     loss = (distances-(torch.sum(distances)/(distances.shape[0]**2-distances.shape[0])))**2
#     loss = torch.mean(loss)
#     return loss * 0.01

def distillation(outputs, labels):
    unique = torch.unique(labels)
    temp   = torch.zeros((len(unique),40), device='cuda')
    for i, v in enumerate(unique):
        temp[i] = torch.mean(outputs[labels==v], dim=0)
    distances = torch.cdist(temp, temp, p=2.0)
    # loss = (distances-(torch.sum(distances)/(distances.shape[0]**2-distances.shape[0])))**2
    # loss = torch.mean(loss)
    loss = 1.0 / (torch.mean(distances))
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
        ce_loss = CrossEntropyLoss(reduce=False, label_smoothing=0.0)
    else:
        ce_loss = CrossEntropyLoss(label_smoothing=0.1)
    # disparity_loss = loss_function
    ##################################################################

    total_batchs = len(dataloader)
    loader = dataloader 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        targets = targets.float()
        
        # targets[targets!=class_index] = 10.0
        # targets[targets==class_index] = 1.00
        # targets[targets==10.0]        = 0.00

        outputs = model(inputs)
        # loss_function(outputs=outputs, labels=targets.long(), epoch=epoch_num)

        predictions = torch.argmax(input=outputs,dim=1).long()
        accuracy.update(torch.sum(targets==predictions)/torch.sum(targets==targets))

        if teacher_model is not None:
            with torch.no_grad():
                outputs_t = teacher_model(inputs)
                weights = F.cross_entropy(outputs_t, targets.long(), reduce=False, label_smoothing=0.0)
                weights = torch.nn.functional.softmax(weights)
                weights = weights.detach()

        if teacher_model is not None:
            loss_ce = ce_loss(outputs, targets.long()) * weights
            loss_ce = torch.mean(loss_ce)
        else:
            loss_ce = ce_loss(outputs, targets.long())
            # loss_ce = loss_kd_regularization(outputs=outputs, labels=targets.long())
            # loss_ce = loss_label_smoothing(outputs=outputs, labels=targets.long())

        # loss_ce = ce_loss(outputs, targets.long())

        # loss_disparity   = distillation(outputs, targets.long())
        loss_disparity = 0
        # loss_disparity = disparity_loss(labels=targets, outputs=outputs)
        # loss_disparity = importance_maps_distillation(s=layer3, t=layer4) + importance_maps_distillation(s=layer2, t=layer3) + importance_maps_distillation(s=layer2, t=layer1)
        # loss_disparity = 5.0 * disparity_loss(fm_s=features_b, fm_t=features_a)
        ###############################################
        loss = loss_ce + loss_disparity
        ###############################################

        lr_ = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9     
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        iter_num = iter_num + 1   

        # iter_num = iter_num + 1 
        # if iter_num % (total_batchs*10)==0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * 0.5


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






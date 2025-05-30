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
from torchnet.meter import mAPMeter
from torcheval.metrics import MulticlassAccuracy
convert = [2, 4, 3, 0, 3, 1, 1, 0, 3, 3, 3, 1, 2, 4, 2, 1, 0, 4, 3, 1, 0, 4, 1, 2, 3, 0, 3, 1, 4, 0, 3, 3, 4, 2, 2, 0, 4, 1, 4, 0, 2, 1, 1, 2, 0, 4, 3, 2, 1, 4, 4, 1, 2, 2, 3, 4, 0, 1, 4, 2, 0, 2, 4, 0, 2, 4, 1]
warnings.filterwarnings("ignore")

general_labels = np.load('/content/UNet_V2/labels.npy')

# def loss_label_smoothing(outputs, labels):
#     """
#     loss function for label smoothing regularization
#     """
#     N = outputs.size(0)  # batch_size
#     C = outputs.size(1)  # number of classes
#     smoothed_labels = torch.zeros(N, C).to('cuda')
#     g_labels = torch.tensor(general_labels).to('cuda')

#     for i in range(len(labels)):
#         smoothed_labels[i] = g_labels[labels[i]]

#     log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
#     loss = -torch.sum(log_prob * smoothed_labels) / N

#     return loss


def loss_label_smoothing(outputs, labels, alpha):
    """
    loss function for label smoothing regularization
    """
    alpha = alpha
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss

def label_smoothing(labels, outputs_t):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.0
    N, C = outputs_t.shape
    # N = 40  # batch_size
    # C = 40  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    # smoothed_labels = (smoothed_labels + outputs_t) / 2.0
    smoothed_labels = outputs_t

    return smoothed_labels

def loss_kd_regularization(outputs, labels):
    """
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    alpha = 0.9
    T = 4
    correct_prob = 0.9    # the probability for correct class in u(k)
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

    # accuracy = utils.AverageMeter()
    metric = MulticlassAccuracy(average="macro", num_classes=num_class).to('cuda')
    # accuracy = mAPMeter()

    if teacher_model is not None:
        ce_loss = CrossEntropyLoss(reduce=False, label_smoothing=0.0)
    else:
        ce_loss = CrossEntropyLoss(label_smoothing=0.0)

    # rkd = RKD()
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
        
        # with torch.autocast(device_type=device, dtype=torch.float16):

        outputs = model(inputs)
        KD = (type(outputs)==tuple)
        if KD:
            outputs, outputs_t = outputs

        ################################################################
        ################################################################
        # k = int(outputs.shape[0]*0.4)
        # indexes = F.cross_entropy(outputs.clone(), targets.long(), reduce=False, label_smoothing=0.0).topk(k).indices
        # weights = torch.zeros(outputs.shape[0]).cuda()
        # weights[indexes] = 1.0
        # loss_ce = F.cross_entropy(outputs, targets.long(), reduce=False, label_smoothing=0.0) * weights
        # loss_ce = torch.mean(loss_ce)
        ################################################################
        ################################################################

        # loss_disparity = 1.0 * importance_maps_distillation(s=x3, t=x3_t) 

        # loss_ce = ce_loss(outputs, label_smoothing(targets.long(), outputs_t))
        
        ####################################################################################################
        ####################################################################################################
        if KD:
            oh_targets = torch.nn.functional.one_hot(targets.long(), num_classes=num_class)
            loss_ce    = ce_loss(outputs, (outputs_t + oh_targets) / 2.0) 
        else:
            loss_ce = ce_loss(outputs, targets.long()) 
        ####################################################################################################
        ####################################################################################################

        # loss_ce = torch.nn.functional.mse_loss(outputs, outputs_t, size_average=None, reduce=None, reduction='mean')
        
        # loss_ce = torch.nn.functional.cross_entropy(outputs, targets.long(), weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)
        # loss_disparity = 1.0 * (importance_maps_distillation(s=x2, t=x2_t) + importance_maps_distillation(s=x3, t=x3_t)) 

        predictions = torch.argmax(input=torch.softmax(outputs, dim=1),dim=1).long()

        metric.update(predictions, targets.long())

        # accuracy.add(torch.softmax(outputs.clone().detach(), dim=1), torch.nn.functional.one_hot(targets.long(), num_classes=num_class))

        # accuracy.update(torch.sum(targets==predictions)/torch.sum(targets==targets))


        # if 0.0 < torch.sum(targets):
        #     accuracy.update(torch.sum((targets+predictions)==2.0)/torch.sum(targets))

        # loss_ce = ce_loss(outputs, outputs_t)

        # loss_ce = ce_loss(outputs, label_smoothing(targets.long(), outputs_t))

        # loss_ce = ce_loss(outputs, targets.long()) + 1.0 * torch.nn.functional.mse_loss(outputs, outputs_t)
        
        ####################################################################################################
        ####################################################################################################

        # T = 3.0
        # loss_ce = (ce_loss(outputs, targets.long())) + (F.kl_div(F.log_softmax(outputs/T, dim=1),F.softmax(outputs_t/T, dim=1),reduction='batchmean') * T * T)

        ####################################################################################################
        ####################################################################################################

        # loss_disparity = distillation(outputs, targets.long())

        loss_disparity = 0.0

        # temp = 4.0
        # alpha = 0.1
        # loss_disparity = (F.kl_div(F.log_softmax(outputs/temp, dim=1),F.softmax(outputs_t/temp, dim=1),reduction='batchmean') * temp * temp)

        # loss_disparity = rkd(x_s, x_t)
        # loss_disparity = 1.0 * importance_maps_distillation(s=x_s, t=x_t) 
        # loss_disparity = 1.0 * (importance_maps_distillation(s=features_s[0], t=features_t[0]) + importance_maps_distillation(s=features_s[1], t=features_t[1])) 
        # loss_disparity = 5.0 * disparity_loss(fm_s=features_b, fm_t=features_a)
        ###############################################
        # loss = (alpha * loss_ce) + ((1.0 - alpha) * loss_disparity)
        loss = loss_ce + loss_disparity
        ###############################################

        lr_ = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9     
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        iter_num = iter_num + 1   

        # iter_num = iter_num + 1 
        # if iter_num <= total_batchs*10:
        #     param_group['lr'] = 0.01 + ((iter_num/(total_batchs*10))*0.099)
        # else:
        #     if iter_num % (total_batchs*10)==0:
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = param_group['lr'] * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        loss_total.update(loss)
        loss_ce_total.update(loss_ce)
        loss_disparity_total.update(loss_disparity)

        ###############################################

        targets = targets.long()

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'CE_loss = {loss_ce_total.avg:.4f} , disparity_loss = {loss_disparity_total.avg:.4f} , Accuracy = {100 * metric.compute():.4f}',   
            # suffix=f'CE_loss = {loss_ce_total.avg:.4f} , disparity_loss = {loss_disparity_total.avg:.4f} , Accuracy = {100 * accuracy.value().item():.4f}',                 
            # suffix=f'CE_loss = {loss_ce_total.avg:.4f} , disparity_loss = {loss_disparity_total.avg:.4f} , Accuracy = {100 * accuracy.avg:.4f}',   
            bar_length=45
        )  

    acc = 100 * metric.compute()

    # acc = 100*accuracy.value().item()

    # acc = 100*accuracy.avg
    

    if lr_scheduler is not None:
        lr_scheduler.step()        
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss_CE : {loss_ce_total.avg:.4f} , Loss_disparity : {loss_disparity_total.avg:.4f} , Accuracy : {acc:.2f} , lr: {optimizer.param_groups[0]["lr"]}')

    # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=acc, acc_per_class=acc, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
    if ckpt is not None:
        ckpt.save_last(acc=acc, acc_per_class=acc, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)

class RKD(nn.Module):

	def __init__(self, w_dist=25, w_angle=50):
		super(RKD, self).__init__()

		self.w_dist  = w_dist
		self.w_angle = w_angle

	def forward(self, feat_s, feat_t):
		loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + \
			   self.w_angle * self.rkd_angle(feat_s, feat_t)

		return loss

	def rkd_dist(self, feat_s, feat_t):
		feat_t_dist = self.pdist(feat_t, squared=False)
		mean_feat_t_dist = feat_t_dist[feat_t_dist>0].mean()
		feat_t_dist = feat_t_dist / mean_feat_t_dist

		feat_s_dist = self.pdist(feat_s, squared=False)
		mean_feat_s_dist = feat_s_dist[feat_s_dist>0].mean()
		feat_s_dist = feat_s_dist / mean_feat_s_dist

		loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

		return loss

	def rkd_angle(self, feat_s, feat_t):
		# N x C --> N x N x C
		feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
		norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
		feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

		feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
		norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
		feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

		loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

		return loss

	def pdist(self, feat, squared=False, eps=1e-12):
		feat_square = feat.pow(2).sum(dim=1)
		feat_prod   = torch.mm(feat, feat.t())
		feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

		if not squared:
			feat_dist = feat_dist.sqrt()

		feat_dist = feat_dist.clone()
		feat_dist[range(len(feat)), range(len(feat))] = 0

		return feat_dist




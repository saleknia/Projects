import utils
from utils import cosine_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing.pool import Pool
from torch.nn.modules.loss import CrossEntropyLoss
from utils import atten_loss,prototype_loss,IM_loss,M_loss
from tqdm import tqdm
from utils import print_progress
import torch.nn.functional as F
import warnings
from utils import focal_loss
from torch.autograd import Variable
from torch.nn.functional import mse_loss as MSE
from utils import importance_maps_distillation as imd
warnings.filterwarnings("ignore")

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class disparity(nn.Module):
    def __init__(self):
        super(disparity, self).__init__()


    def forward(self, outputs, masks):
        loss = 0.0
        B,C,H,W = outputs.shape
        
        temp_masks = masks

        mask_unique_value = torch.unique(temp_masks)
        mask_unique_value = mask_unique_value[1:]
        unique_num = len(mask_unique_value)

        if unique_num<2:
            return 0

        prototypes = torch.zeros(size=(unique_num,C),device='cuda')

        for count,p in enumerate(mask_unique_value):
            p = p.long()
            bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
            bin_mask = bin_mask.unsqueeze(dim=1).expand_as(outputs)

            temp = 0.0
            batch_counter = 0
            for t in range(B):
                if torch.sum(bin_mask[t])!=0:
                    v = torch.sum(bin_mask[t]*outputs[t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
                    temp = temp + v
                    batch_counter = batch_counter + 1
            temp = temp / batch_counter
            prototypes[count] = temp

        distances = torch.cdist(prototypes, prototypes, p=2.0)
        loss = 1.0 / (torch.mean(distances)) 

        return loss


def im_distill(student, teacher):
    exp=4
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    s = student
    t = teacher
    if s.shape[2] != t.shape[2]:
        s = F.interpolate(s, t.size()[-2:], mode='bilinear')
    loss = torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()
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
    loss_dice_total = utils.AverageMeter()
    loss_ce_total = utils.AverageMeter()
    loss_disparity_total = utils.AverageMeter()

    Eval = utils.Evaluator(num_class=num_class)

    mIOU = 0.0
    Dice = 0.0

    accuracy = utils.AverageMeter()

    dice_loss = DiceLoss(num_class)
    ce_loss = CrossEntropyLoss()    
    disparity_loss = disparity()

    ##################################################################

    total_batchs = len(dataloader)
    loader = dataloader 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        targets = targets.float()

        # targets = targets[:, 0, :, :]

        # print(torch.unique(targets))

        # with torch.autocast(device_type=device, dtype=torch.float16):
        outputs = model(inputs)

        if teacher_model is not None:
            with torch.no_grad():
                outputs_t, up1_t, up2_t, up3_t, up4_t, x1_t, x2_t, x3_t, x4_t, x5_t = teacher_model(inputs,multiple=True)

        if type(outputs)==tuple:
            loss_ce   = ce_loss(outputs[0], targets[:].long()) + ce_loss(outputs[1], targets[:].long()) + ce_loss(outputs[2], targets[:].long()) 
            loss_dice = dice_loss(inputs=outputs[0], target=targets, softmax=True) + dice_loss(inputs=outputs[1], target=targets, softmax=True) + dice_loss(inputs=outputs[2], target=targets, softmax=True)
            loss_att  = 0.0
            loss = loss_ce + loss_dice + loss_att
        else:
            loss_ce   = ce_loss(outputs, targets[:].long())
            loss_dice = dice_loss(inputs=outputs, target=targets, softmax=True)
            loss_att  = 0.0
            loss = loss_ce + loss_dice + loss_att

        # loss_ce = ce_loss(outputs, targets[:].long())
        # loss_dice = dice_loss(inputs=outputs, target=targets, softmax=True)

        # loss_disparity = disparity_loss(outputs, targets)
        loss_disparity = 0.0

        ###############################################
        alpha = 1.0
        beta  = 1.0
        theta = 0.0

        loss = alpha * loss_dice + beta * loss_ce + theta * loss_disparity
        ###############################################

        lr_ = 0.001 * (1.0 - iter_num / max_iterations) ** 0.9

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1       

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total.update(loss)
        loss_dice_total.update(loss_dice)
        loss_ce_total.update(loss_ce)
        loss_disparity_total.update(loss_disparity)
        ###############################################
        targets = targets.long()

        if type(outputs)==tuple:
            predictions = torch.argmax(input=outputs[0],dim=1).long()
        else:
            predictions = torch.argmax(input=outputs,dim=1).long()

        Eval.add_batch(gt_image=targets,pre_image=predictions)

        accuracy.update(Eval.Pixel_Accuracy())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'Dice_loss = {alpha*loss_dice_total.avg:.4f} , CE_loss = {beta*loss_ce_total.avg:.4f} , kd_loss = {theta * loss_disparity_total.avg:.4f} , Dice = {Eval.Dice()*100:.2f}',          
            # suffix=f'Dice_loss = {alpha*loss_dice_total.avg:.4f}, CE_loss = {beta*loss_ce_total.avg:.4f}, kd_loss = {loss_kd_total.avg:.4f}, att_loss = {loss_att_total.avg:.4f}, proto_loss = {loss_proto_total.avg:.4f}, Dice = {Eval.Dice()*100:.2f}',          
            bar_length=45
        )  
  
    acc = 100*accuracy.avg
    mIOU = 100*Eval.Mean_Intersection_over_Union()
    Dice,Dice_per_class = Eval.Dice(per_class=True)
    Dice,Dice_per_class = 100*Dice,100*Dice_per_class

    if writer is not None:
        writer.add_scalar('Loss/train', loss_total.avg.item(), epoch_num)
        writer.add_scalar('Acc/train', acc.item(), epoch_num)
        writer.add_scalar('Dice/train', Dice.item(), epoch_num)
        writer.add_scalar('MIoU/train', mIOU.item(), epoch_num)

    if lr_scheduler is not None:
        lr_scheduler.step()        
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss: {loss_total.avg:.4f} , mIoU: {mIOU:.2f} , Dice: {Dice:.2f} , Pixel Accuracy: {acc:.2f}, lr: {optimizer.param_groups[0]["lr"]}')

    # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
    if ckpt is not None:
        ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)






# import utils
# from utils import cosine_scheduler
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.modules.loss import CrossEntropyLoss
# from utils import DiceLoss,atten_loss,prototype_loss,IM_loss,M_loss,CriterionPixelWise
# from tqdm import tqdm
# from utils import print_progress
# import torch.nn.functional as F
# import warnings
# from utils import focal_loss
# from torch.autograd import Variable
# warnings.filterwarnings("ignore")

# def one_hot_loss(exist_pred, targets):
#     targets=targets.cpu()
#     exist_pred=exist_pred.cpu()
#     yp = torch.zeros(6,9)
#     for i in range(6):
#         unique_num = torch.unique(targets[i])
#         for j in range(9):
#             if j in unique_num:
#                 yp[i,j] = 1.0
#             else:
#                 yp[i,j] = 0.0
#     loss = torch.nn.functional.binary_cross_entropy(input=exist_pred, target=yp)
#     return loss

# def loss_kd_regularization(outputs, masks):
#     """
#     loss function for mannually-designed regularization: Tf-KD_{reg}
#     """
#     correct_prob = 0.95    # the probability for correct class in u(k)
#     K = outputs.size(1)

#     teacher_scores = torch.ones_like(outputs).cuda()
#     teacher_scores = teacher_scores*(1-correct_prob)/(K-1)  # p^d(k)

#     teacher_scores[masks] = correct_prob

#     return teacher_scores

# def prediction_map_distillation(y, masks) :
#     """
#     basic KD loss function based on "Distilling the Knowledge in a Neural Network"
#     https://arxiv.org/abs/1503.02531
#     :param y: student score map
#     :param teacher_scores: teacher score map
#     :param T:  for softmax
#     :return: loss value
#     """
#     y = y.cuda()
#     masks = masks.long()
#     masks = masks.cuda()

#     masks_temp = F.one_hot(masks, num_classes=9)
#     masks_temp = torch.permute(masks_temp, (0, 3, 1, 2))
#     masks_temp = masks_temp.bool()

#     teacher_scores = loss_kd_regularization(outputs=y, masks=masks_temp)

#     return teacher_scores

# def flatten(tensor):
#     """Flattens a given tensor such that the channel axis is first.
#     The shapes are transformed as follows:
#        (N, C, D, H, W) -> (C, N * D * H * W)
#     """
#     # number of channels
#     C = tensor.size(1)
#     # new axis order
#     axis_order = (1, 0) + tuple(range(2, tensor.dim()))
#     # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
#     transposed = tensor.permute(axis_order)
#     # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
#     return transposed.contiguous().view(C, -1)

# class _AbstractDiceLoss(nn.Module):
#     """
#     Base class for different implementations of Dice loss.
#     """

#     def __init__(self, weight=None, normalization='sigmoid'):
#         super(_AbstractDiceLoss, self).__init__()
#         self.register_buffer('weight', weight)
#         # The output from the network during training is assumed to be un-normalized probabilities and we would
#         # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
#         # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
#         # However if one would like to apply Softmax in order to get the proper probability distribution from the
#         # output, just specify `normalization=Softmax`
#         assert normalization in ['sigmoid', 'softmax', 'none']
#         if normalization == 'sigmoid':
#             self.normalization = nn.Sigmoid()
#         elif normalization == 'softmax':
#             self.normalization = nn.Softmax(dim=1)
#         else:
#             self.normalization = lambda x: x

#     def dice(self, input, target, weight):
#         # actual Dice score computation; to be implemented by the subclass
#         raise NotImplementedError

#     def forward(self, input, target):
#         # get probabilities from logits
#         input = self.normalization(input)

#         # compute per channel Dice coefficient
#         per_channel_dice = self.dice(input, target, weight=self.weight)

#         # average Dice score across all channels/classes
#         return 1. - torch.mean(per_channel_dice)

# class GeneralizedDiceLoss(_AbstractDiceLoss):
#     """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
#     """

#     def __init__(self, num_classes, normalization='softmax', epsilon=1e-6):
#         super().__init__(weight=None, normalization=normalization)
#         self.epsilon = epsilon
#         self.num_classes = num_classes
#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.num_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def dice(self, input, target, weight):
#         target = self._one_hot_encoder(target)
#         assert input.size() == target.size(), "'input' and 'target' must have the same shape"

#         input = flatten(input)
#         target = flatten(target)
#         target = target.float()

#         if input.size(0) == 1:
#             # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
#             # put foreground and background voxels in separate channels
#             input = torch.cat((input, 1 - input), dim=0)
#             target = torch.cat((target, 1 - target), dim=0)

#         # GDL weighting: the contribution of each label is corrected by the inverse of its volume
#         w_l = target.sum(-1)
#         w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
#         w_l.requires_grad = False

#         intersect = (input * target).sum(-1)
#         intersect = intersect * w_l

#         denominator = (input + target).sum(-1)
#         denominator = (denominator * w_l).clamp(min=self.epsilon)

#         return 2 * (intersect.sum() / denominator.sum())

# class WeightedCrossEntropyLoss(nn.Module):
#     """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
#     """

#     def __init__(self, ignore_index=-1):
#         super(WeightedCrossEntropyLoss, self).__init__()
#         self.ignore_index = ignore_index

#     def forward(self, input, target):
#         weight = self._class_weights(input)
#         return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

#     @staticmethod
#     def _class_weights(input):
#         # normalize the input first
#         input = F.softmax(input, dim=1)
#         flattened = flatten(input)
#         nominator = (1. - flattened).sum(-1)
#         denominator = flattened.sum(-1)
#         class_weights = Variable(nominator / denominator, requires_grad=False)
#         return class_weights

# # def prediction_map_distillation(y, masks, T=4.0) :
# #     """
# #     basic KD loss function based on "Distilling the Knowledge in a Neural Network"
# #     https://arxiv.org/abs/1503.02531
# #     :param y: student score map
# #     :param teacher_scores: teacher score map
# #     :param T:  for softmax
# #     :return: loss value
# #     """
# #     y = y.cuda()
# #     masks = masks.long()
# #     masks = masks.cuda()

# #     bin_masks = masks
# #     bin_masks[bin_masks!=0] = 1.0 

# #     masks_temp = F.one_hot(masks, num_classes=9)
# #     masks_temp = torch.permute(masks_temp, (0, 3, 1, 2))
# #     masks_temp = masks_temp.bool()

# #     teacher_scores = loss_kd_regularization(outputs=y, masks=masks_temp)

# #     # y_prime = y * bin_masks.unsqueeze(dim=1).expand_as(y)
# #     # teacher_scores_prime = teacher_scores * bin_masks.unsqueeze(dim=1).expand_as(teacher_scores)

# #     y_prime = y
# #     teacher_scores_prime = teacher_scores 

# #     p = F.log_softmax(y_prime / T , dim=1)
# #     q = F.softmax(teacher_scores_prime / T, dim=1)

# #     p = p.view(-1, 2)
# #     q = q.view(-1, 2)

# #     l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
# #     return l_kl


# def trainer(end_epoch,epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,writer,logger,loss_function):
#     torch.autograd.set_detect_anomaly(True)
#     print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')

#     model=model.to(device)
#     model.train()

#     loss_total = utils.AverageMeter()
#     loss_dice_total = utils.AverageMeter()
#     loss_ce_total = utils.AverageMeter()
#     loss_proto_total = utils.AverageMeter()
#     loss_kd_total = utils.AverageMeter()

#     Eval = utils.Evaluator(num_class=num_class)

#     mIOU = 0.0
#     Dice = 0.0

#     accuracy = utils.AverageMeter()

#     # ce_loss = WeightedCrossEntropyLoss()
#     dice_loss = GeneralizedDiceLoss(num_classes=num_class)
#     ce_loss = CrossEntropyLoss()
#     # dice_loss = DiceLoss(num_class)
#     ##################################################################
#     kd_loss = M_loss()    
#     proto_loss = loss_function
#     ##################################################################
#     total_batchs = len(dataloader)
#     loader = dataloader 

#     base_iter = (epoch_num-1) * total_batchs
#     iter_num = base_iter
#     max_iterations = end_epoch * total_batchs
#     # max_iterations = 50 * total_batchs

#     for batch_idx, (inputs, targets) in enumerate(loader):

#         inputs, targets = inputs.to(device), targets.to(device)

#         targets = targets.float()
        
#         outputs, up4, up3, up2, up1 = model(inputs)
     
#         targets = targets.long()
#         predictions = torch.argmax(input=outputs,dim=1).long()
#         overlap = (predictions==targets).float()
#         t_masks = targets * overlap
#         targets = targets.float()

#         # loss_ce = ce_loss(input=outputs, target=targets.long())
#         loss_dice = dice_loss(input=outputs, target=targets)

#         loss_ce = ce_loss(outputs, targets[:].long())
#         # loss_dice = dice_loss(inputs=outputs, target=targets, softmax=True)

#         # loss_proto = proto_loss(masks=targets.clone(), t_masks=t_masks, up4=up4, up3=up3, up2=up2, up1=up1, outputs=outputs)
#         # loss_kd = kd_loss(e5=e5, e4=e4, e3=e3, e2=e2)

#         loss_proto = 0.0
#         loss_kd = 0.0
#         # loss_ce = 0
#         # loss_dice = 0

#         # loss_kd_out = prediction_map_distillation(y=outputs, masks=targets)
#         # loss_kd_out = kd_out_loss(masks=targets.clone(), x4=up4, x3=up3, x2=up2, x1=up1)
#         # loss_kd = kd_loss(e5)
#         ###############################################
#         alpha = 1.0
#         beta = 1.0
#         gamma = 0.01
#         loss = alpha * loss_dice + beta * loss_ce 
#         # loss = alpha * loss_dice + beta * loss_ce + gamma * loss_proto         
#         # loss = 0.5 * loss_ce + 0.5 * loss_dice + beta * loss_kd
#         # loss = loss_kd 
#         ###############################################

#         # lr_ = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9

#         # for param_group in optimizer.param_groups:
#         #     param_group['lr'] = lr_

#         # iter_num = iter_num + 1        
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # lr_scheduler.step()


#         loss_total.update(loss)
#         loss_dice_total.update(loss_dice)
#         loss_ce_total.update(loss_ce)
#         loss_proto_total.update(loss_proto)
#         loss_kd_total.update(loss_kd)
#         ###############################################
#         targets = targets.long()

#         predictions = torch.argmax(input=outputs,dim=1).long()
#         Eval.add_batch(gt_image=targets,pre_image=predictions)

#         accuracy.update(Eval.Pixel_Accuracy())

#         print_progress(
#             iteration=batch_idx+1,
#             total=total_batchs,
#             prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
#             # suffix=f'Dice_loss = {loss_dice_total.avg:.4f} , CE_loss={loss_ce_total.avg:.4f} , Att_loss = {loss_att_total.avg:.6f} , mIoU = {Eval.Mean_Intersection_over_Union()*100:.2f} , Dice = {Eval.Dice()*100:.2f}',
#             # suffix=f'Dice_loss = {loss_dice_total.avg:.4f} , CE_loss={loss_ce_total.avg:.4f} , mIoU = {Eval.Mean_Intersection_over_Union()*100:.2f} , Dice = {Eval.Dice()*100:.2f}',          
#             # suffix=f'Dice_loss = {0.5*loss_dice_total.avg:.4f} , CE_loss = {0.5*loss_ce_total.avg:.4f} , proto_loss = {alpha*loss_proto_total.avg:.8f} , Dice = {Eval.Dice()*100:.2f}',         
#             suffix=f'Dice_loss = {alpha*loss_dice_total.avg:.4f} , CE_loss = {beta*loss_ce_total.avg:.4f} , proto_loss = {gamma*loss_proto_total.avg:.8f} , kd_loss = {beta*loss_kd_total.avg:.4f}, Dice = {Eval.Dice()*100:.2f}',          
#             # suffix=f'Dice_loss = {0.5*loss_dice_total.avg:.4f} , CE_loss = {0.5*loss_ce_total.avg:.4f} , loss_kd = {beta*loss_kd_total.avg:.8f} , Dice = {Eval.Dice()*100:.2f}',          
#             bar_length=45
#         )  
  
#     # acc = 100*accuracy.avg
#     # mIOU = 100*Eval.Mean_Intersection_over_Union()
#     # Dice = 100*Eval.Dice()
#     acc = 100*accuracy.avg
#     mIOU = 100*Eval.Mean_Intersection_over_Union()
#     Dice,Dice_per_class = Eval.Dice(per_class=True)
#     Dice,Dice_per_class = 100*Dice,100*Dice_per_class

#     if writer is not None:
#         writer.add_scalar('Loss/train', loss_total.avg.item(), epoch_num)
#         writer.add_scalar('Acc/train', acc.item(), epoch_num)
#         writer.add_scalar('Dice/train', Dice.item(), epoch_num)
#         writer.add_scalar('MIoU/train', mIOU.item(), epoch_num)

#     if lr_scheduler is not None:
#         lr_scheduler.step()        
        
#     logger.info(f'Epoch: {epoch_num} ---> Train , Loss: {loss_total.avg:.4f} , mIoU: {mIOU:.2f} , Dice: {Dice:.2f} , Pixel Accuracy: {acc:.2f}, lr: {optimizer.param_groups[0]["lr"]}')

#     # Save checkpoint
#     if ckpt is not None:
#         ckpt.save_best(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
#     if ckpt is not None:
#         ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
#     # if ckpt is not None and (epoch_num==end_epoch):
#     #     ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
#     # if ckpt is not None and (early_stopping < ckpt.early_stopping(epoch_num)):
#     #     ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)  
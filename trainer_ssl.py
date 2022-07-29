import utils
from utils import cosine_scheduler
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss,atten_loss,prototype_loss,IM_loss,M_loss,CriterionPixelWise
from tqdm import tqdm
from utils import print_progress
import torch.nn.functional as F
import warnings
from utils import focal_loss
from torch.autograd import Variable
warnings.filterwarnings("ignore")

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    v = torch.nn.functional.softmax(v, dim=1)
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, num_classes, normalization='softmax', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon
        self.num_classes = num_classes
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def dice(self, input, target, weight):
        target = self._one_hot_encoder(target)
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())

class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


def trainer(end_epoch,epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,writer,logger,loss_function):
    torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')

    proto = loss_function

    model=model.to(device)
    model.train()

    loss_total_1 = utils.AverageMeter()
    loss_total_2 = utils.AverageMeter()
    loss_total_3 = utils.AverageMeter()


    Eval_1 = utils.Evaluator(num_class=2)
    Eval_2 = utils.Evaluator(num_class=9)


    mIOU_1 = 0.0
    Dice_1 = 0.0

    mIOU_2 = 0.0
    Dice_2 = 0.0

    accuracy_1 = utils.AverageMeter()
    accuracy_2 = utils.AverageMeter()

    ce_loss_1 = CrossEntropyLoss()
    dice_loss_1 = DiceLoss(2)

    ce_loss_2 = CrossEntropyLoss()
    dice_loss_2 = DiceLoss(9)

    total_batchs = len(dataloader)
    loader = dataloader 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    for batch_idx, ((inputs_1, targets_1),(inputs_2, targets_2)) in enumerate(loader):

        inputs_1, targets_1 = inputs_1.to(device), targets_1.to(device)
        targets_1[targets_1!=4.0] = 0.0
        targets_1[targets_1==4.0] = 1.0

        inputs_2, targets_2 = inputs_2.to(device), targets_2.to(device)

        targets_1 = targets_1.float()
        targets_2 = targets_2.float()
 
        outputs_1_1 , outputs_1_2 = model(inputs_1, num_head=2.0) # input_1 ---> Single
        outputs_2_1 , outputs_2_2 = model(inputs_2, num_head=2.0) # input_2 ---> Multi

        proto(masks=targets_2, outputs=outputs_2_2)

        loss_dice_1 = dice_loss_1(inputs=outputs_1_1, target=targets_1, softmax=True)
        loss_ce_1 = ce_loss_1(outputs_1_1, targets_1[:].long())

        loss_dice_2 = dice_loss_2(inputs=outputs_2_2, target=targets_2, softmax=True)
        loss_ce_2 = ce_loss_2(outputs_2_2, targets_2[:].long())

        loss_3 = proto.align(outputs=outputs_1_2) * 0.5

        alpha_1 = 1.0
        beta_1 = 1.0

        alpha_2 = 1.0
        beta_2 = 1.0

        loss_1 = alpha_1 * loss_dice_1 + beta_1 * loss_ce_1
        loss_2 = alpha_2 * loss_dice_2 + beta_2 * loss_ce_2
        loss = loss_1 + loss_2 + loss_3
 
        lr_ = 0.05 * (1.0 - iter_num / max_iterations) ** 0.9

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        loss_total_1.update(loss_1)
        loss_total_2.update(loss_2)
        loss_total_3.update(loss_3)

        targets_1 = targets_1.long()
        targets_2 = targets_2.long()



        predictions_1 = torch.argmax(input=outputs_1_1,dim=1).long()
        Eval_1.add_batch(gt_image=targets_1,pre_image=predictions_1)

        predictions_2 = torch.argmax(input=outputs_2_2,dim=1).long()
        Eval_2.add_batch(gt_image=targets_2,pre_image=predictions_2)

        accuracy_1.update(Eval_1.Pixel_Accuracy())
        accuracy_2.update(Eval_2.Pixel_Accuracy())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'loss_1 = {loss_total_1.avg:.4f} , loss_2 = {loss_total_2.avg:.4f} , loss_3 = {loss_total_3.avg:.4f} , Dice_1 = {Eval_1.Dice()*100:.2f} , Dice_2 = {Eval_2.Dice()*100:.2f}',          
            bar_length=45
        )  
  
    acc = 100 * accuracy_1.avg
    mIOU = 100 * Eval_1.Mean_Intersection_over_Union()
    Dice,Dice_per_class = Eval_1.Dice(per_class=True)
    Dice,Dice_per_class = 100*Dice,100*Dice_per_class

    if lr_scheduler is not None:
        lr_scheduler.step()        
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss: {loss_total_1.avg:.4f} , mIoU: {mIOU:.2f} , Dice: {Dice:.2f} , Pixel Accuracy: {acc:.2f}, lr: {optimizer.param_groups[0]["lr"]}')

    # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
    if ckpt is not None:
        ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)




import utils
from utils import cosine_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss,atten_loss,prototype_loss,IM_loss
from tqdm import tqdm
from utils import print_progress
import torch.nn.functional as F
import warnings
from valid_s import valid_s
warnings.filterwarnings("ignore")

def loss_kd_regularization(outputs, masks):
    """
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    correct_prob = 0.9    # the probability for correct class in u(k)
    K = outputs.size(1)

    teacher_scores = torch.ones_like(outputs).cuda()
    teacher_scores = teacher_scores*(1-correct_prob)/(K-1)  # p^d(k)

    teacher_scores[masks] = correct_prob

    return teacher_scores

def prediction_map_distillation(y, masks, T=2.0) :
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param y: student score map
    :param teacher_scores: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    y = y.cuda()
    masks = masks.long()
    masks = masks.cuda()

    bin_masks = masks
    bin_masks[bin_masks!=0] = 1.0 

    masks_temp = F.one_hot(masks, num_classes=9)
    masks_temp = torch.permute(masks_temp, (0, 3, 1, 2))
    masks_temp = masks_temp.bool()

    teacher_scores = loss_kd_regularization(outputs=y, masks=masks_temp)

    y_prime = y * bin_masks.unsqueeze(dim=1).expand_as(y)
    teacher_scores_prime = teacher_scores * bin_masks.unsqueeze(dim=1).expand_as(teacher_scores)

    p = F.log_softmax(y_prime / T , dim=1)
    q = F.softmax(teacher_scores_prime / T, dim=1)

    p = p.view(-1, 2)
    q = q.view(-1, 2)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return l_kl


def trainer_s(end_epoch,epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,writer,logger,loss_function):
    torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    model.train()

    loss_total = utils.AverageMeter()
    loss_dice_total = utils.AverageMeter()
    loss_ce_total = utils.AverageMeter()

    Eval = utils.Evaluator(num_class=num_class)

    mIOU = 0.0
    Dice = 0.0

    accuracy = utils.AverageMeter()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_class)
    total_batchs = len(dataloader['train'])
    loader = dataloader['train'] 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()
        outputs = model(inputs)

        loss_ce = ce_loss(outputs, targets[:].long())
        loss_dice = dice_loss(outputs, targets, softmax=True)

        alpha = 0.5
        beta = 0.5
        loss = 0.5 * loss_ce + 0.5 * loss_dice

        lr_ = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9

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
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'Dice_loss = {0.5*loss_dice_total.avg:.4f} , CE_loss = {0.5*loss_ce_total.avg:.4f} , Dice = {Eval.Dice()*100:.2f}',         
            bar_length=45
        )  
  
    # acc = 100*accuracy.avg
    # mIOU = 100*Eval.Mean_Intersection_over_Union()
    # Dice = 100*Eval.Dice()
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
    valid_s(end_epoch,epoch_num,model,dataloader['valid'],device,ckpt,num_class,writer,logger,optimizer)




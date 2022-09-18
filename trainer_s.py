import utils
from utils import cosine_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss,atten_loss,prototype_loss,IM_loss,M_loss, disparity, dice_iou
from tqdm import tqdm
from utils import print_progress
import torch.nn.functional as F
import warnings
from utils import focal_loss
from torch.autograd import Variable
from torch.nn.functional import mse_loss as MSE
from utils import importance_maps_distillation as imd
from valid_s import valid_s
warnings.filterwarnings("ignore")

class CriterionPixelWise(nn.Module):
    def __init__(self):
        super(CriterionPixelWise, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = preds_T.permute(0,2,3,1).contiguous().view(-1,C)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss

def trainer_s(end_epoch,epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,writer,logger,loss_function):
    torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model=model.to('cuda')
    model.train()

    loss_total = utils.AverageMeter()
    loss_ce_total = utils.AverageMeter()
    loss_dice_total = utils.AverageMeter()

    Eval = utils.Evaluator(num_class=num_class)

    mIOU = 0.0
    Dice = 0.0

    accuracy = utils.AverageMeter()
    accuracy_eval = utils.AverageMeter()

    dice_loss = DiceLoss(num_class)
    ce_loss = CrossEntropyLoss()
    kd_loss = CriterionPixelWise()

    total_batchs = len(dataloader['train'])
    loader = dataloader['train'] 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()
        inputs = inputs.float()
        outputs = model(inputs)

        # targets = targets.long()
        # predictions = torch.argmax(input=outputs,dim=1).long()
        # overlap = (predictions==targets).float()
        # t_masks = targets * overlap
        # targets = targets.float()

        # print(targets.shape)
        # print(outputs.shape)
        soft_label = 0.5 * (torch.nn.functional.softmax(outputs) + torch.nn.functional.one_hot(targets.long()).permute(0,3,1,2))
        loss_kd = kd_loss(preds_S=outputs, preds_T=soft_label)
        loss_ce = ce_loss(outputs, targets[:].long()) 
        loss_dice = dice_loss(inputs=outputs, target=targets, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice + loss_kd
 
        lr_ = 0.001 * (1.0 - iter_num / max_iterations) ** 0.9

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total.update(loss)
        loss_ce_total.update(loss_ce)
        loss_dice_total.update(loss_dice)

        targets = targets.long()

        predictions = torch.argmax(input=outputs,dim=1).long()
        Eval.add_batch(gt_image=targets,pre_image=predictions)
        accuracy.update(Eval.Pixel_Accuracy())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'Dice_loss = {0.5*loss_dice_total.avg:.4f} , CE_loss = {0.5*loss_ce_total.avg:.4f} , Dice = {Eval.Dice()*100:.2f} , Pixel Accuracy: {accuracy.avg*100:.2f}',          
            bar_length=45
        )  
  
    acc = 100*accuracy.avg
    mIOU = 100*Eval.Mean_Intersection_over_Union()

    Dice = Eval.Dice()
    Dice_per_class = Dice
    # Dice,Dice_per_class = Eval.Dice(per_class=True)
    # Dice,Dice_per_class = 100*Dice,100*Dice_per_class

    if lr_scheduler is not None:
        lr_scheduler.step()        
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss: {loss_total.avg:.4f} , Dice: {Dice:.2f} , mIoU: {mIOU:.2f} , Pixel Accuracy: {acc:.2f} , Pixel Accuracy Eval: {accuracy_eval.avg*100:.2f} , lr: {optimizer.param_groups[0]["lr"]}')
    valid_s(end_epoch,epoch_num,model,dataloader['valid'],device,ckpt,num_class,writer,logger,optimizer)




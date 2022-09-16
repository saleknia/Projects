import utils
import torch
from utils import print_progress
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss, hd95
import warnings
from medpy import metric
import medpy
import numpy as np
warnings.filterwarnings("ignore")

def valid_s(end_epoch,epoch_num,model,dataloader,device,ckpt,num_class,writer,logger,optimizer):
    model=model.to(device)
    model.eval()
    loss_total = utils.AverageMeter()
    Eval = utils.Evaluator(num_class=num_class)
    mIOU = 0.0
    accuracy = utils.AverageMeter()

    dice_loss = DiceLoss(num_class)
    ce_loss = CrossEntropyLoss()

    total_batchs = len(dataloader)
    loader = dataloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            targets = targets.float()
            outputs = model(inputs)

            loss_ce = ce_loss(outputs, targets[:].long())
            loss_dice = dice_loss(inputs=outputs, target=targets, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            loss_total.update(loss)

            targets = targets.long()
            predictions = torch.argmax(input=outputs,dim=1).long()
            Eval.add_batch(gt_image=targets,pre_image=predictions)
            accuracy.update(Eval.Pixel_Accuracy())

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Valid {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss= {loss_total.avg:.4f} , Accuracy= {accuracy.avg*100:.2f} ,  Dice = {Eval.Dice()*100:.2f}',
                bar_length=45
            )  

    acc = 100*accuracy.avg
    mIOU = 100*Eval.Mean_Intersection_over_Union()
    Dice,Dice_per_class = Eval.Dice(per_class=True)
    Dice,Dice_per_class = 100*Dice,100*Dice_per_class
   
    logger.info(f'Epoch: {epoch_num} ---> Valid , Loss: {loss_total.avg:.4f} , mIoU: {mIOU:.2f} , Dice: {Dice:.2f} , Pixel Accuracy: {acc:.2f}, lr: {optimizer.param_groups[0]["lr"]}')

    # # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=None)



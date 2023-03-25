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
warnings.filterwarnings("ignore")

def tester(end_epoch,epoch_num,model,dataloader,device,ckpt,num_class,writer,logger,optimizer,lr_scheduler,early_stopping):
    model.eval()
    loss_total = utils.AverageMeter()
    hd95_total = utils.AverageMeter()
    Eval = utils.Evaluator(num_class=num_class)
    mIOU = 0.0
    Dice = 0.0
    accuracy = utils.AverageMeter()
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_class)

    total_batchs = len(dataloader)
    loader = dataloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)

            targets = targets.float()

            # targets[targets!=4.0] = 0.0
            # targets[targets==4.0] = 1.0

            outputs = model(inputs)

            loss_ce = ce_loss(outputs, targets[:].long())
            loss_dice = dice_loss(outputs, targets, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            # loss = loss_func(inputs=outputs, target=targets, softmax=True)
            loss_total.update(loss)

            targets = targets.long()
            targets = targets[:, 0, :, :]

            predictions = torch.argmax(input=outputs,dim=1).long()

            # predictions[predictions!=6.0] = 0.0
            # predictions[predictions==6.0] = 1.0

            Eval.add_batch(gt_image=targets,pre_image=predictions)
            hd95_acc = hd95(masks=targets,preds=predictions,num_class=num_class)

            # hd95_acc = 0.0
            if not np.isnan(hd95_acc):
                hd95_total.update(hd95_acc)
            accuracy.update(Eval.Pixel_Accuracy())

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Valid {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss= {loss_total.avg:.4f} , Accuracy= {accuracy.avg*100:.2f} , mIoU= {Eval.Mean_Intersection_over_Union()*100:.2f} , Dice= {Eval.Dice()*100:.2f} , hd95= {hd95_total.avg:.4f}',
                bar_length=45
            )  

        acc = 100*accuracy.avg
        mIOU = 100*Eval.Mean_Intersection_over_Union()
        Dice,Dice_per_class = Eval.Dice(per_class=True)
        Dice,Dice_per_class = 100*Dice,100*Dice_per_class
        Dice_per_class = [x.item() for x in Dice_per_class]
        if writer is not None:
            writer.add_scalar('Loss/valid', loss_total.avg.item(), epoch_num)
            writer.add_scalar('Acc/valid', acc.item(), epoch_num)
            writer.add_scalar('Dice/valid', Dice.item(), epoch_num)
            writer.add_scalar('MIoU/valid', mIOU.item(), epoch_num)

        logger.info(f'Epoch: {epoch_num} ---> Valid , Loss: {loss_total.avg:.4f} , mIoU: {mIOU:.2f} , Dice: {Dice:.2f} , hd95: {hd95_total.avg:.4f} , Pixel Accuracy: {acc:.2f}') 
        logger.info(f'Dice Per Class: {Dice_per_class}') 
    # # Save checkpoint
    # if ckpt is not None:
    #     ckpt.save_best(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
    # if ckpt is not None and (epoch_num==end_epoch):
    #     ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
    # if ckpt is not None and (early_stopping < ckpt.early_stopping(epoch_num)):
    #     ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)


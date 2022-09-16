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

def valid_s(end_epoch,epoch_num,model,dataloader,device,ckpt,num_class,writer,logger,optimizer,weight):
    model=model.to(device)
    model.eval()
    loss_total = utils.AverageMeter()
    Eval = Evaluator(num_class=num_class+1)
    mIOU = 0.0
    accuracy = utils.AverageMeter()

    ce_loss = CrossEntropyLoss(weight=weight, ignore_index=11)

    total_batchs = len(dataloader)
    loader = dataloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)

            targets = targets.float()
            outputs = model(inputs)

            loss = ce_loss(outputs, targets[:].long())

            loss_total.update(loss)

            targets = targets.long()
            predictions = torch.argmax(input=outputs,dim=1).long()
            Eval.add_batch(gt_image=targets,pre_image=predictions)
            accuracy.update(Eval.Pixel_Accuracy())

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Valid {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss= {loss_total.avg:.4f} , Accuracy= {accuracy.avg*100:.2f} , mIoU= {Eval.MIOU_out()*100:.2f}',
                bar_length=45
            )  

        acc = 100*accuracy.avg
        mIOU,mIOU_per_class = Eval.MIOU_out(per_class=True)
        mIOU,mIOU_per_class = 100*mIOU,100*mIOU_per_class
        mIOU_per_class = [x.item() for x in mIOU_per_class]


        logger.info(f'Epoch: {epoch_num} ---> Valid , Loss: {loss_total.avg:.4f} , mIoU: {mIOU:.2f} , Pixel Accuracy: {acc:.2f}') 

    # # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=mIOU, acc_per_class=mIOU_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=None)



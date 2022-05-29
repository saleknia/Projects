import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss, BCELoss
from utils import DiceLoss,atten_loss,prototype_loss,prototype_loss_mean
from tqdm import tqdm
from utils import print_progress
import warnings
warnings.filterwarnings("ignore")

class patch_loss(nn.Module):
    def __init__(self,p,class_num):
        super(patch_loss, self).__init__()
        self.p = p
        self.class_num=class_num
        self.pc_loss = BCELoss()
    def forward(self, pc, targets):
        loss_pc = self.pc_loss(pc, patch_mask(targets,self.p, class_num=self.class_num).unsqueeze(dim=1).float()) 
        return loss_pc

# class patch_loss(nn.Module):
#     def __init__(self,p=16):
#         super(patch_loss, self).__init__()
#         self.p = p
#         self.pc_loss = DiceLoss(1)
#     def forward(self, pc, targets):
#         loss_pc = self.pc_loss(pc, patch_mask(targets,self.p).float()) 
#         return loss_pc

def patch_mask(mask, p, class_num=2.0):
    B, H, W = mask.shape
    b, h, w = B, H//p, W//p
    mask_c = torch.zeros(b, h, w, dtype=torch.long)

    for k in range(b):
        for i in range(p):
            for j in range(p):
                # if class_num in mask[i*p:(i+1)*p,j*p:(j+1)*p]:
                if torch.sum(mask[i*p:(i+1)*p,j*p:(j+1)*p]==class_num)<p*p and (class_num in mask[i*p:(i+1)*p,j*p:(j+1)*p]):
                    mask_c[k, i, j] = 1.0

    mask_c = mask_c.to('cuda')
    return mask_c

def trainer(end_epoch,epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,writer,logger):
    torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')

    model=model.to(device)
    ##################################################################
    # activation = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output
    #     return hook
    # model.up4.register_forward_hook(get_activation('up4'))
    # model.up3.register_forward_hook(get_activation('up3'))
    # model.up2.register_forward_hook(get_activation('up2'))
    # model.up1.register_forward_hook(get_activation('up1'))
    ##################################################################
    model.train()

    loss_total = utils.AverageMeter()
    loss_dice_total = utils.AverageMeter()
    loss_ce_total = utils.AverageMeter()
    loss_pc_total = utils.AverageMeter()

    Eval = utils.Evaluator(num_class=num_class)

    mIOU = 0.0
    Dice = 0.0

    accuracy = utils.AverageMeter()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_class)
    ##################################################################
    proto_loss = prototype_loss_mean()
    ##################################################################
    total_batchs = len(dataloader)
    loader = dataloader 

    base_iter = (epoch_num-1) * total_batchs
    iter_num = base_iter
    max_iterations = end_epoch * total_batchs
    # max_iterations = 50 * total_batchs

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        targets = targets.float()

        ##################################################################
        # masks = nn.functional.interpolate(targets.clone().unsqueeze(dim=1), scale_factor=0.125, mode='nearest')
        # masks = masks.squeeze(dim=1)
        ##################################################################


        outputs, proto = model(inputs)

        # print(activation['up4'].shape)
        # print(activation['up3'].shape)
        # print(activation['up2'].shape)
        # print(activation['up1'].shape)

        loss_ce = ce_loss(outputs, targets[:].long())
        loss_dice = dice_loss(outputs, targets, softmax=True)
        # loss_proto = proto_loss(masks=targets, up4=up4, up3=up3, up2=up2, up1=up1)
        # loss_proto = proto_loss(masks=targets, up3=up3, up2=up2, up1=up1)
        loss_proto = proto_loss(masks=targets, outputs=proto)


        ###############################################
        # alpha = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9
        alpha = 0.01
        loss = 0.5 * loss_ce + 0.5 * loss_dice + alpha * loss_proto
        ###############################################

        lr_ = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # lr_ = 0.01 * (1.0 - iter_num / max_iterations) ** 0.9
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_

        # iter_num = iter_num + 1

        loss_total.update(loss)
        loss_dice_total.update(loss_dice)
        loss_ce_total.update(loss_ce)
        loss_pc_total.update(loss_pc)
        ###############################################
        targets = targets.long()
        predictions = torch.argmax(input=outputs,dim=1).long()
        Eval.add_batch(gt_image=targets,pre_image=predictions)

        accuracy.update(Eval.Pixel_Accuracy())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            # suffix=f'Dice_loss = {loss_dice_total.avg:.4f} , CE_loss={loss_ce_total.avg:.4f} , Att_loss = {loss_att_total.avg:.6f} , mIoU = {Eval.Mean_Intersection_over_Union()*100:.2f} , Dice = {Eval.Dice()*100:.2f}',
            # suffix=f'Dice_loss = {loss_dice_total.avg:.4f} , CE_loss={loss_ce_total.avg:.4f} , mIoU = {Eval.Mean_Intersection_over_Union()*100:.2f} , Dice = {Eval.Dice()*100:.2f}',          
            suffix=f'Dice_loss = {0.5*loss_dice_total.avg:.4f} , CE_loss = {0.5*loss_ce_total.avg:.4f} , Patch_loss = {alpha*loss_pc_total.avg:.4f} , Dice = {Eval.Dice()*100:.2f}',          
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

    # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
    if ckpt is not None:
        ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
    # if ckpt is not None and (epoch_num==end_epoch):
    #     ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)
    # if ckpt is not None and (early_stopping < ckpt.early_stopping(epoch_num)):
    #     ckpt.save_last(acc=Dice, acc_per_class=Dice_per_class, epoch=epoch_num, net=model, optimizer=optimizer,lr_scheduler=lr_scheduler)  



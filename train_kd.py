# Instaling Libraries
import os
import copy
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
import pickle
import argparse
from torch.backends import cudnn
# from albumentations.pytorch.transforms import ToTensorV2
# import albumentations as A
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import random_split
from tqdm.notebook import tqdm
import torch.optim as optim
from models.UNet import UNet
from models.UNet_loss import UNet_loss
from models.UNet_plus import NestedUNet
from models.UNet_plus_loss import NestedUNet_loss
from models.att_unet import AttentionUNet
from models.att_unet_loss import AttentionUNet_loss
from models.multi_res_unet import MultiResUnet
from models.U import U
from models.U_loss import U_loss
from models.ERFNet import ERFNet
from models.ERFNet_loss import ERFNet_loss
from models.multi_res_unet_loss import MultiResUnet_loss
from models.UCTransNet import UCTransNet
from models.GT_UNet import GT_U_Net
from models.ENet import ENet
from models.Mobile_netV2 import Mobile_netV2
from models.Mobile_netV2_loss import Mobile_netV2_loss
from models.Fast_SCNN import Fast_SCNN
from models.Fast_SCNN_loss import Fast_SCNN_loss
from models.ESPNet import ESPNet
from models.ESPNet_loss import ESPNet_loss
from models.DABNet import DABNet
from models.DABNet_loss import DABNet_loss
from models.ENet_loss import ENet_loss
from models.UCTransNet_GT import UCTransNet_GT
from models.GT_CTrans import GT_CTrans
# from models.original_UNet import original_UNet
import utils
from utils import color
from utils import Save_Checkpoint
from trainer_kd import trainer_kd
from tester import tester
from dataset import COVID_19,Synapse_dataset,RandomGenerator,ValGenerator,ACDC,CT_1K
from utils import DiceLoss,atten_loss,prototype_loss,prototype_loss_kd,proto
from tabulate import tabulate
from tensorboardX import SummaryWriter
# from testing import inference
# from testingV2 import inferenceV2
import warnings
warnings.filterwarnings('ignore')

NUM_WORKERS = 4
PIN_MEMORY = True

SEED = 666


class Checkpoint(object):
    def __init__(self,filename):
        self.best_acc = 0.0
        self.best_epoch = 1
        self.folder = 'checkpoint'
        self.filename=filename
        self.best_path = '/content/drive/MyDrive/checkpoint/' + self.filename + '_best.pth'
        self.last_path = '/content/drive/MyDrive/checkpoint/' + self.filename + '_last.pth'
        os.makedirs(self.folder, exist_ok=True)

    def save_best(self, acc, acc_per_class, epoch, net, optimizer, lr_scheduler):
        if self.best_acc < acc:
            print(color.BOLD+color.RED+'Saving best checkpoint...'+color.END)
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc': acc,
                'acc_per_class':acc_per_class,
                'best_epoch': epoch
            }
            self.best_epoch = epoch
            self.best_acc = acc
            torch.save(state, self.best_path)

    def save_last(self, acc, acc_per_class, epoch, net, optimizer, lr_scheduler):
        print(color.BOLD+color.RED+'Saving last checkpoint...'+color.END)
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': acc,
            'best_acc': self.best_acc,
            'acc_per_class': acc_per_class,
            'best_epoch': self.best_epoch,
            'num_epoch': epoch
        }
        
    def best_accuracy(self):
        return self.best_acc

def main(args, student_train=True, device='cuda'):
    
    if student_train:
        train_tf = transforms.Compose([RandomGenerator(output_size=[args.image_height, args.image_width])])
        val_tf = ValGenerator(output_size=[args.image_height, args.image_width])

        if args.student_name=='UNet':
            student_model = UNet(n_channels=1, n_classes=args.num_class).to(device)

        elif args.student_name=='UNet_loss':
            student_model = UNet_loss(n_channels=1, n_classes=args.num_class).to(device)

        elif args.student_name=='U':
            student_model = U(bilinear=False, n_classes=args.num_class).to(device)

        elif args.student_name=='U_loss':
            student_model = U_loss(bilinear=False).to(device)

        elif args.student_name == 'AttUNet':
            student_model = AttentionUNet(img_ch=1, output_ch=args.num_class).to(device)

        elif args.student_name == 'AttUNet_loss':
            student_model = AttentionUNet_loss(img_ch=1, output_ch=args.num_class).to(device) 

        elif args.student_name == 'ENet':
            student_model = ENet(nclass=args.num_class).to(device)

        elif args.student_name == 'ENet_loss':
            student_model = ENet_loss(nclass=args.num_class).to(device)

        elif args.student_name == 'ESPNet':
            student_model = ESPNet(num_classes=args.num_class).to(device)

        elif args.student_name == 'ESPNet_loss':
            student_model = ESPNet_loss(num_classes=args.num_class).to(device)

        else: 
            raise TypeError('Please enter a valid name for the model type')

        if args.teacher_name=='UNet':
            teacher_model = UNet(n_channels=1, n_classes=args.num_class).to(device)

        elif args.teacher_name=='UNet_loss':
            teacher_model = UNet_loss(n_channels=1, n_classes=args.num_class).to(device)

        elif args.teacher_name=='U':
            teacher_model = U(bilinear=False, n_classes=args.num_class).to(device)

        elif args.teacher_name=='U_loss':
            teacher_model = U_loss(bilinear=False).to(device)

        elif args.teacher_name == 'AttUNet':
            teacher_model = AttentionUNet(img_ch=1, output_ch=args.num_class).to(device)

        elif args.teacher_name == 'AttUNet_loss':
            teacher_model = AttentionUNet_loss(img_ch=1, output_ch=args.num_class).to(device) 

        elif args.teacher_name == 'ENet':
            teacher_model = ENet(nclass=args.num_class).to(device)

        elif args.teacher_name == 'ENet_loss':
            teacher_model = ENet_loss(nclass=args.num_class).to(device)

        elif args.teacher_name == 'ESPNet':
            teacher_model = ESPNet(num_classes=args.num_class).to(device)

        elif args.teacher_name == 'ESPNet_loss':
            teacher_model = ESPNet_loss(num_classes=args.num_class).to(device)

        else: 
            raise TypeError('Please enter a valid name for the model type')

        num_parameters = utils.count_parameters(student_model)

        model_table = tabulate(
            tabular_data=[[args.student_name, f'{num_parameters:.2f} M', device]],
            headers=['student Model', '#Parameters', 'Device'],
            tablefmt="fancy_grid"
            )
        logger.info(model_table)

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, student_model.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001) 


        checkpoint_path = '/content/drive/MyDrive/checkpoint/' + args.teacher_name + '_teacher.pth'
        logger.info('Loading Teacher Model...')
        if os.path.isfile(checkpoint_path):
            pretrained_model_path = checkpoint_path
            loaded_data = torch.load(pretrained_model_path, map_location='cuda')
            pretrained = loaded_data['net']
            model2_dict = teacher_model.state_dict()
            state_dict = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}
            model2_dict.update(state_dict)
            teacher_model.load_state_dict(model2_dict)

            initial_best_acc=loaded_data['best_acc']
            loaded_acc=loaded_data['acc']
            initial_best_epoch=loaded_data['best_epoch']
            last_num_epoch=loaded_data['num_epoch']

            optimizer.load_state_dict(loaded_data['optimizer'])

            table = tabulate(
                            tabular_data=[[loaded_acc, initial_best_acc, initial_best_epoch, last_num_epoch]],
                            headers=['Loaded Model Acc', 'Initial Best Acc', 'Best Epoch Number', 'Num Epochs'],
                            tablefmt="fancy_grid"
                            )
            logger.info(table)
        else:
            logger.info(f'No Such file : {checkpoint_path}')
        logger.info('\n')

        for param in teacher_model.parameters():
            param.requires_grad = False

        start_epoch = 1
        end_epoch =  args.num_epochs


        if args.task_name=='Synapse':
            train_dataset=Synapse_dataset(split='train', joint_transform=train_tf)
            valid_dataset=Synapse_dataset(split='valid', joint_transform=val_tf)

            train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    worker_init_fn=worker_init,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    drop_last=True,
                                    )
            valid_loader = DataLoader(valid_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    worker_init_fn=worker_init,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    drop_last=True,
                                    )

            data_loader={'train':train_loader,'valid':valid_loader}

        elif args.task_name=='ACDC':

            train_dataset=ACDC(split='train', joint_transform=train_tf)
            valid_dataset=ACDC(split='test', joint_transform=val_tf)
            train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    worker_init_fn=worker_init,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    drop_last=True,
                                    )
            valid_loader = DataLoader(valid_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    worker_init_fn=worker_init,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    drop_last=True,
                                    )

            data_loader={'train':train_loader,'valid':valid_loader}

        elif args.task_name=='CT-1K':

            train_dataset=CT_1K(split='train', joint_transform=train_tf)
            valid_dataset=CT_1K(split='test', joint_transform=val_tf)

            train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    worker_init_fn=worker_init,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    drop_last=True,
                                    )
            valid_loader = DataLoader(valid_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    worker_init_fn=worker_init,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    drop_last=True,
                                    )

            data_loader={'train':train_loader,'valid':valid_loader}


        if args.save_model:
            checkpoint = Checkpoint(args.ckpt_name)
        else:
            checkpoint = None

        if args.train=='True':
            logger.info(50*'*')
            logger.info('Student Training Phase')
            logger.info(50*'*')
            for epoch in range(start_epoch,end_epoch+1):
                trainer_kd(
                        end_epoch=end_epoch,
                        epoch_num=epoch,
                        teacher_model=teacher_model,
                        student_model=student_model,
                        dataloader=data_loader['train'],
                        optimizer=optimizer,
                        device='cuda',
                        ckpt=checkpoint,                
                        num_class=args.num_class,
                        logger=logger)
            
            if epoch==end_epoch:

                if args.save_model and 0 < checkpoint.best_accuracy():
                    pretrained_model_path = '/content/drive/MyDrive/checkpoint/' + args.student_name + '_best.pth'
                    loaded_data = torch.load(pretrained_model_path, map_location='cuda')
                    pretrained = loaded_data['net']
                    model2_dict = student_model.state_dict()
                    state_dict = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}
                    model2_dict.update(state_dict)
                    student_model.load_state_dict(model2_dict)

                    acc=loaded_data['acc']
                    acc_per_class=loaded_data['acc_per_class'].tolist()
                    acc_per_class=[round(x,2) for x in acc_per_class]
                    best_epoch=loaded_data['best_epoch']

                    logger.info(50*'*')
                    logger.info(f'Best Accuracy over training: {acc:.2f}')
                    logger.info(f'Best Accuracy Per Class over training: {acc_per_class}')
                    logger.info(f'Epoch Number: {best_epoch}')

                    if args.inference=='True':
                        logger.info(50*'*')
                        logger.info('Inference Phase')
                        tester(
                            end_epoch=1,
                            epoch_num=1,
                            model=copy.deepcopy(model),
                            dataloader=data_loader['valid'],
                            device=DEVICE,
                            ckpt=None,
                            num_class=NUM_CLASS,
                            writer=writer,
                            logger=logger,
                            optimizer=None,
                            lr_scheduler=None,
                            early_stopping=None)
            
                    logger.info(50*'*')
                    logger.info(50*'*')
                    logger.info('\n')

parser = argparse.ArgumentParser()
parser.add_argument('--inference', type=str,default='False')
parser.add_argument('--train', type=str,default='True')
args = parser.parse_args()

def worker_init(worker_id):
    random.seed(SEED + worker_id)

if __name__ == "__main__":
    
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(SEED)    
    np.random.seed(SEED)  
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) 


    main(args)
    
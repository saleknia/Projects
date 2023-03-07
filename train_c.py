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
from models.SEUNet import SEUNet
from models.UNet import UNet
from models.UNet_loss import UNet_loss
from models.UNet_plus import NestedUNet
from models.UNet_plus_loss import NestedUNet_loss
from models.att_unet import AttentionUNet
from models.att_unet_loss import AttentionUNet_loss
from models.multi_res_unet import MultiResUnet
from models.ERFNet import ERFNet
from models.ERFNet_loss import ERFNet_loss
from models.U import U
from models.U_loss import U_loss
from models.multi_res_unet_loss import MultiResUnet_loss
from models.UCTransNet import UCTransNet
from models.GT_UNet import GT_U_Net
from models.ENet import ENet
from models.DABNet import DABNet
from models.DABNet_loss import DABNet_loss
from models.Mobile_netV2 import Mobile_netV2
from models.Mobile_netV2_loss import Mobile_netV2_loss
from models.ESPNet import ESPNet
from models.ESPNet_loss import ESPNet_loss
from models.ENet_loss import ENet_loss
from models.UCTransNet_GT import UCTransNet_GT
from models.GT_CTrans import GT_CTrans
from models.Fast_SCNN import Fast_SCNN
from models.Fast_SCNN_loss import Fast_SCNN_loss
from models.TransFuse import TransFuse_S
from models.DATUNet import DATUNet
from models.Cross_unet import Cross_unet
from models.Cross import Cross
# from models.original_UNet import original_UNet
import utils
from utils import color
from utils import Save_Checkpoint
from trainer_s import trainer_s
from tester_s import tester_s
from trainer_c import trainer
from tester_c import tester
from dataset import COVID_19,Synapse_dataset,RandomGenerator,ValGenerator,ACDC,CT_1K,TCIA,ISIC2017,ISIC2016,ISIC2018
from utils import DiceLoss,atten_loss,prototype_loss,prototype_loss_kd
from config import *
from tabulate import tabulate
from tensorboardX import SummaryWriter
from dataset_builder import build_dataset_train, build_dataset_test
# from testing import inference
# from testingV2 import inferenceV2 TCIA tester
import warnings
warnings.filterwarnings('ignore')

class collect(nn.Module):
    def __init__(self, start_epoch):
        super(collect, self).__init__()

        self.outputs = None
        self.labels = None
        self.start_epoch = start_epoch

    def forward(self, outputs, labels, epoch):
        if self.start_epoch <= epoch:
            if self.outputs==None:
                self.outputs = outputs.detach().cpu()
                self.labels = labels.detach().cpu()
            else:
                outputs = outputs.detach().cpu()
                labels = labels.detach().cpu()
                self.outputs = torch.cat([self.outputs, outputs], dim=0)
                self.labels = torch.cat([self.labels, labels], dim=0)
    def save(self):
        self.outputs = self.outputs.numpy()
        self.labels = self.labels.numpy()
        np.savez('outputs.npz', outputs=self.outputs, labels=self.labels)

def main(args):

    if tensorboard:
        tensorboard_log = tensorboard_folder
        logger.info(f'Tensorboard Directory: {tensorboard_log}')
        if not os.path.isdir(tensorboard_log):
            os.makedirs(tensorboard_log)
        writer = SummaryWriter(tensorboard_log)
    else:
        writer = None

    train_tf = transforms.Compose([RandomGenerator(output_size=[IMAGE_HEIGHT, IMAGE_WIDTH])])
    val_tf = ValGenerator(output_size=[IMAGE_HEIGHT, IMAGE_WIDTH])

# LOAD_MODEL

    if MODEL_NAME=='UNet':
        model = UNet(n_channels=3, n_classes=NUM_CLASS).to(DEVICE)
        # model = original_UNet().to(DEVICE)

    elif MODEL_NAME=='UNet_loss':
        model = UNet_loss(n_channels=1, n_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME=='U':
        model = U(bilinear=False).to(DEVICE)

    elif MODEL_NAME=='U_loss':
        model = U_loss(bilinear=False).to(DEVICE)

    elif MODEL_NAME=='UCTransNet':
        config_vit = get_CTranS_config()
        model = UCTransNet(config_vit,n_channels=3,n_classes=n_labels,img_size=IMAGE_HEIGHT).to(DEVICE)

    elif MODEL_NAME=='UCTransNet_GT':
        config_vit = get_CTranS_config()
        model = UCTransNet_GT(config_vit,n_channels=n_channels,n_classes=n_labels,img_size=IMAGE_HEIGHT).to(DEVICE)

    elif MODEL_NAME=='GT_UNet':
        model = GT_U_Net(img_ch=1,output_ch=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME=='GT_CTrans':
        config_vit = get_CTranS_config()
        model = GT_CTrans(config_vit,img_ch=1,output_ch=NUM_CLASS,img_size=256).to(DEVICE)

    elif MODEL_NAME == 'AttUNet':
        model = AttentionUNet(img_ch=3, output_ch=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'AttUNet_loss':
        model = AttentionUNet_loss(img_ch=1, output_ch=NUM_CLASS).to(DEVICE) 

    elif MODEL_NAME == 'MultiResUnet':
        model = MultiResUnet().to(DEVICE)

    elif MODEL_NAME == 'MultiResUnet_loss':
        model = MultiResUnet_loss().to(DEVICE)

    elif MODEL_NAME == 'UNet++':
        model = NestedUNet().to(DEVICE)

    elif MODEL_NAME == 'UNet++_loss':
        model = NestedUNet_loss().to(DEVICE)

    elif MODEL_NAME == 'ENet':
        model = ENet(nclass=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'ENet_loss':
        model = ENet_loss(nclass=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'ERFNet':
        model = ERFNet(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'ERFNet_loss':
        model = ERFNet_loss(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'ESPNet':
        model = ESPNet(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'ESPNet_loss':
        model = ESPNet_loss(num_classes=NUM_CLASS).to(DEVICE)
        
    elif MODEL_NAME == 'DABNet':
        model = DABNet(classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'DABNet_loss':
        model = DABNet_loss(classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'Fast_SCNN':
        model = Fast_SCNN(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'Fast_SCNN_loss':
        model = Fast_SCNN_loss(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'Mobile_NetV2':
        model = Mobile_netV2(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'Mobile_NetV2_loss':
        model = Mobile_netV2_loss(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'TransFuse_S':
        model = TransFuse_S(pretrained=True).to(DEVICE)

    elif MODEL_NAME == 'DATUNet':
        model = DATUNet().to(DEVICE)

    elif MODEL_NAME=='SEUNet':
        model = SEUNet(n_channels=3, n_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME=='Cross_unet':
        model = Cross_unet().to(DEVICE)

    elif MODEL_NAME=='Cross':
        model = Cross().to(DEVICE)

    else: 
        raise TypeError('Please enter a valid name for the model type')


    num_parameters = utils.count_parameters(model)

    model_table = tabulate(
        tabular_data=[[MODEL_NAME, f'{num_parameters:.2f} M', DEVICE]],
        headers=['Builded Model', '#Parameters', 'Device'],
        tablefmt="fancy_grid"
        )
    logger.info(model_table)

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)

    # optimizer = None

    if COSINE_LR is True:
        lr_scheduler = utils.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler =  None     

    if TEACHER is True:
        logger.info('Loading Teacher Checkpoint...')
        teacher_model = Mobile_netV2_loss().to(DEVICE)
    else:
        teacher_model =  None     

    # teacher_model =  None     

    initial_best_acc = 0
    initial_best_epoch = 1
    current_num_epoch = 0
    last_num_epoch = 0
    if LOAD_MODEL:
        checkpoint_path = '/content/drive/MyDrive/checkpoint/'+CKPT_NAME+'_last.pth'
        logger.info('Loading Checkpoint...')
        if os.path.isfile(checkpoint_path):

            pretrained_model_path = checkpoint_path
            loaded_data = torch.load(pretrained_model_path, map_location='cuda')
            pretrained = loaded_data['net']
            model2_dict = model.state_dict()
            state_dict = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}
            # logger.info(state_dict.keys())
            model2_dict.update(state_dict)
            model.load_state_dict(model2_dict)

            initial_best_acc=loaded_data['best_acc']
            loaded_acc=loaded_data['acc']
            initial_best_epoch=loaded_data['best_epoch']
            last_num_epoch=loaded_data['num_epoch']
            current_num_epoch=last_num_epoch+current_num_epoch

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

    start_epoch = last_num_epoch + 1
    end_epoch =  NUM_EPOCHS

    if TASK_NAME=='COVID-19':

        train_dataset=COVID_19(split='train', joint_transform=train_tf)
        valid_dataset=COVID_19(split='test', joint_transform=val_tf)

        train_loader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                worker_init_fn=worker_init,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY,
                                drop_last=True,
                                )
        valid_loader = DataLoader(valid_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                worker_init_fn=worker_init,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY,
                                drop_last=True,
                                )

        data_loader={'train':train_loader,'valid':valid_loader}

    elif TASK_NAME=='Synapse':

        train_dataset=Synapse_dataset(split='train', joint_transform=train_tf)
        valid_dataset=Synapse_dataset(split='val', joint_transform=val_tf)

        train_loader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
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

    elif TASK_NAME=='ACDC':
        # index = np.load(file='index.npy')
        # train_dataset=Synapse_dataset(split='train',index=index[0:2000],joint_transform=train_tf)
        # valid_dataset=Synapse_dataset(split='val',index=index[2000:],joint_transform=val_tf)

        train_dataset=ACDC(split='train', joint_transform=train_tf)
        valid_dataset=ACDC(split='test', joint_transform=val_tf)

        # g = torch.Generator()
        # g.manual_seed(0)

        train_loader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
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

    elif TASK_NAME=='CT-1K':

        train_dataset=CT_1K(split='train', joint_transform=train_tf)
        valid_dataset=CT_1K(split='test', joint_transform=val_tf)

        train_loader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
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
        # data_loader={'train':train_loader,'valid':train_loader}

    elif TASK_NAME=='Standford40':

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        trainset = torchvision.datasets.ImageFolder(root='/content/StanfordAction/train/',
                                        transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        testset = torchvision.datasets.ImageFolder(root='/content/StanfordAction/test/',
                                        transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        data_loader={'train':train_loader,'valid':test_loader}

    elif TASK_NAME=='FER2013':

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((48, 48)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        trainset = torchvision.datasets.ImageFolder(root='/content/FER2013/train/', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        testset = torchvision.datasets.ImageFolder(root='/content/FER2013/test/', transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        data_loader={'train':train_loader,'valid':test_loader}

    if SAVE_MODEL:
        checkpoint = Save_Checkpoint(CKPT_NAME,current_num_epoch,last_num_epoch,initial_best_acc,initial_best_epoch)
    else:
        checkpoint = None

    if args.train=='True':
        logger.info(50*'*')
        logger.info('Training Phase')
        logger.info(50*'*')
        loss_function = collect(start_epoch=56)
        for epoch in range(start_epoch,end_epoch+1):
            trainer(
                end_epoch=end_epoch,
                epoch_num=epoch,
                model=model,
                teacher_model = teacher_model,
                dataloader=data_loader['train'],
                optimizer=optimizer,
                device=DEVICE,
                ckpt=checkpoint,                
                num_class=NUM_CLASS,
                lr_scheduler=lr_scheduler,
                writer=writer,
                logger=logger,
                loss_function=loss_function)
            
            if epoch==end_epoch:
                if SAVE_MODEL and 0 < checkpoint.best_accuracy():
                    pretrained_model_path = '/content/drive/MyDrive/checkpoint/' + CKPT_NAME + '_best.pth'
                    loaded_data = torch.load(pretrained_model_path, map_location='cuda')
                    pretrained = loaded_data['net']
                    model2_dict = model.state_dict()
                    state_dict = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}
                    # logger.info(state_dict.keys())
                    model2_dict.update(state_dict)
                    model.load_state_dict(model2_dict)

                    acc=loaded_data['acc']
                    # acc_per_class=loaded_data['acc_per_class'].tolist()
                    # acc_per_class=[round(x,2) for x in acc_per_class]
                    best_epoch=loaded_data['best_epoch']

                    logger.info(50*'*')
                    logger.info(f'Best Accuracy over training: {acc:.2f}')
                    # logger.info(f'Best Accuracy Per Class over training: {acc_per_class}')
                    logger.info(f'Epoch Number: {best_epoch}')

                    if args.inference=='True':
                        logger.info(50*'*')
                        logger.info('Inference Phase')
                        # logger.info(50*'*')
                        # inference(model=model,logger=logger)
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
        # loss_function.save()
    if tensorboard:
        writer.close()

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

    if os.path.isfile('/content/drive/MyDrive/checkpoint/numpy_state.pickle'):
        with open('/content/drive/MyDrive/checkpoint/numpy_state.pickle', 'rb') as f:
            numpy_state = pickle.load(f) 
        np.random.set_state(numpy_state)
   
    if os.path.isfile('/content/drive/MyDrive/checkpoint/random_state.pickle'):
        with open('/content/drive/MyDrive/checkpoint/random_state.pickle', 'rb') as f:
            random_state = pickle.load(f) 
        random.setstate(random_state)
    

    if os.path.isfile('/content/drive/MyDrive/checkpoint/torch_state.pth'):
        torch_state = torch.load('/content/drive/MyDrive/checkpoint/torch_state.pth')
        torch.set_rng_state(torch_state)

    if os.path.isfile('/content/drive/MyDrive/checkpoint/cuda_state.pth'):
        cuda_state = torch.load('/content/drive/MyDrive/checkpoint/cuda_state.pth')
        torch.cuda.set_rng_state(cuda_state)

    main(args)
    
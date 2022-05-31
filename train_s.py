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
from models.multi_res_unet_loss import MultiResUnet_loss
from models.UCTransNet import UCTransNet
from models.GT_UNet import GT_U_Net
from models.ENet import ENet
from models.DABNet import DABNet
from models.DABNet_loss import DABNet_loss
from models.ENet_loss import ENet_loss
from models.UCTransNet_GT import UCTransNet_GT
from models.GT_CTrans import GT_CTrans
# from models.original_UNet import original_UNet
import utils
from utils import color
from utils import Save_Checkpoint
from trainer_s import trainer_s
from tester_s import tester_s
from dataset import COVID_19,Synapse_dataset,RandomGenerator,ValGenerator,ACDC,CT_1K
from utils import DiceLoss,atten_loss,prototype_loss,prototype_loss_kd
from config import *
from tabulate import tabulate
from tensorboardX import SummaryWriter
# from testing import inference
# from testingV2 import inferenceV2
import warnings
warnings.filterwarnings('ignore')


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
    # train_tf = ValGenerator(output_size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    val_tf = ValGenerator(output_size=[IMAGE_HEIGHT, IMAGE_WIDTH])

    # transform = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         ToTensorV2()
    #     ]
    # )

# LOAD_MODEL

    if MODEL_NAME=='UNet':
        model = UNet(n_channels=1, n_classes=NUM_CLASS).to(DEVICE)
        # model = original_UNet().to(DEVICE)

    elif MODEL_NAME=='UNet_loss':
        model = UNet_loss(n_channels=1, n_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME=='U':
        model = U(bilinear=False).to(DEVICE)

    elif MODEL_NAME=='U_loss':
        model = U_loss(bilinear=False).to(DEVICE)

    elif MODEL_NAME=='UCTransNet':
        config_vit = get_CTranS_config()
        model = UCTransNet(config_vit,n_channels=n_channels,n_classes=n_labels,img_size=IMAGE_HEIGHT).to(DEVICE)

    elif MODEL_NAME=='UCTransNet_GT':
        config_vit = get_CTranS_config()
        model = UCTransNet_GT(config_vit,n_channels=n_channels,n_classes=n_labels,img_size=IMAGE_HEIGHT).to(DEVICE)

    elif MODEL_NAME=='GT_UNet':
        model = GT_U_Net(img_ch=1,output_ch=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME=='GT_CTrans':
        config_vit = get_CTranS_config()
        model = GT_CTrans(config_vit,img_ch=1,output_ch=NUM_CLASS,img_size=256).to(DEVICE)

    elif MODEL_NAME == 'AttUNet':
        model = AttentionUNet(img_ch=1, output_ch=NUM_CLASS).to(DEVICE)

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

    elif MODEL_NAME == 'DABNet':
        model = DABNet(classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'DABNet_loss':
        model = DABNet_loss(classes=NUM_CLASS).to(DEVICE)
        
    else: 
        raise TypeError('Please enter a valid name for the model type')


    num_parameters = utils.count_parameters(model)

    model_table = tabulate(
        tabular_data=[[MODEL_NAME, f'{num_parameters:.2f} M', DEVICE]],
        headers=['Builded Model', '#Parameters', 'Device'],
        tablefmt="fancy_grid"
        )
    logger.info(model_table)

    # loss_fn = utils.DiceLoss(n_classes=NUM_CLASS)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
 
    if COSINE_LR is True:
        lr_scheduler = utils.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler =  None     


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
            # lr_scheduler.load_state_dict(loaded_data['lr_scheduler'])

            # if last_num_epoch-initial_best_epoch < early_stopping:
            #     count_es = last_num_epoch-initial_best_epoch
            #     es = False
            # elif early_stopping < last_num_epoch-initial_best_epoch:
            #     count_es = early_stopping + 1
            #     es = True
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
        # index = np.load(file='index.npy')
        # train_dataset=Synapse_dataset(split='train',index=index[0:2000],joint_transform=train_tf)
        # valid_dataset=Synapse_dataset(split='val',index=index[2000:],joint_transform=val_tf)

        train_dataset=Synapse_dataset(split='train', joint_transform=train_tf)
        valid_dataset=Synapse_dataset(split='val', joint_transform=val_tf)

        # g = torch.Generator()
        # g.manual_seed(0)

        train_loader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                worker_init_fn=worker_init,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY,
                                drop_last=True,
                                # generator=g
                                )
        valid_loader = DataLoader(valid_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                worker_init_fn=worker_init,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY,
                                drop_last=True,
                                # generator=g
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
        valid_dataset=CT_1K(split='valid', joint_transform=val_tf)
        test_dataset=CT_1K(split='test', joint_transform=val_tf)

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
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                worker_init_fn=worker_init,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY,
                                drop_last=True,
                                )
        test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                worker_init_fn=worker_init,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY,
                                drop_last=True,
                                )

        data_loader={'train':train_loader,'valid':valid_loader,'test':test_loader}

    if SAVE_MODEL:
        checkpoint = Save_Checkpoint(CKPT_NAME,current_num_epoch,last_num_epoch,initial_best_acc,initial_best_epoch)
    else:
        checkpoint = None

    if args.train=='True':
        logger.info(50*'*')
        logger.info('Training Phase')
        logger.info(50*'*')
        loss_function = prototype_loss()
        # loss_function = prototype_loss_kd()
        for epoch in range(start_epoch,end_epoch+1):
            # set_epoch(epoch,g)
            trainer_s(
                end_epoch=end_epoch,
                epoch_num=epoch,
                model=model,
                dataloader=data_loader,
                optimizer=optimizer,
                device=DEVICE,
                ckpt=checkpoint,                
                num_class=NUM_CLASS,
                lr_scheduler=lr_scheduler,
                writer=writer,
                logger=logger,
                loss_function=loss_function)
            # tester_s(
            #     end_epoch=end_epoch,
            #     epoch_num=epoch,
            #     model=model,
            #     dataloader=data_loader['valid'],
            #     device=DEVICE,
            #     ckpt=checkpoint,
            #     num_class=NUM_CLASS,
            #     writer=writer,
            #     logger=logger,
            #     optimizer=optimizer,
            #     lr_scheduler=lr_scheduler,
            #     early_stopping=early_stopping)

            if epoch==end_epoch:
                if SAVE_MODEL and 0 < checkpoint.best_accuracy():
                    # pretrained_model_path = os.path.join(os.path.abspath('checkpoint'), CKPT_NAME + '_best.pth')
                    # pretrained_model_path = '/content/drive/MyDrive/checkpoint/' + CKPT_NAME + '_best.pth'
                    pretrained_model_path = '/content/drive/MyDrive/checkpoint/' + CKPT_NAME + '_best.pth'
                    loaded_data = torch.load(pretrained_model_path, map_location='cuda')
                    pretrained = loaded_data['net']
                    model2_dict = model.state_dict()
                    state_dict = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}
                    model2_dict.update(state_dict)
                    model.load_state_dict(model2_dict)

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
                        logger.info(50*'*')
                        inference(model=model,logger=logger)
                        tester_s(
                            end_epoch=1,
                            epoch_num=1,
                            model=copy.deepcopy(model),
                            dataloader=data_loader['test'],
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
    if tensorboard:
        writer.close()

    # numpy_state = np.random.get_state()
    # with open('./checkpoint/numpy_state.pickle', 'wb') as f:
    #     pickle.dump(numpy_state, f)

    # random_state = random.getstate()
    # with open('./checkpoint/random_state.pickle', 'wb') as f:
    #     pickle.dump(random_state, f)

    # torch_state = torch.get_rng_state()
    # torch.save(torch_state, './checkpoint/torch_state.pth')

    # cuda_state = torch.cuda.get_rng_state()
    # torch.save(cuda_state, './checkpoint/cuda_state.pth')

parser = argparse.ArgumentParser()
parser.add_argument('--inference', type=str,default='False')
parser.add_argument('--train', type=str,default='True')
args = parser.parse_args()

def worker_init(worker_id):
    random.seed(SEED + worker_id)

# def set_epoch(epoch,g):
#     g.manual_seed(5728479885 + epoch)   

# def worker_init(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

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
    
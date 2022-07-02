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
from models.DABNet import DABNet
from models.DABNet_loss import DABNet_loss
from models.ENet_loss import ENet_loss
from models.UCTransNet_GT import UCTransNet_GT
from models.GT_CTrans import GT_CTrans
import ml_collections
import utils
from utils import color, print_progress
from utils import Save_Checkpoint
from trainer import trainer
from tester import tester
from dataset import COVID_19,Synapse_dataset,RandomGenerator,ValGenerator,ACDC,CT_1K
from utils import DiceLoss,atten_loss,prototype_loss,prototype_loss_kd
from tabulate import tabulate
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA  
from sklearn.manifold import TSNE

BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = True
SEED = 666

def worker_init(worker_id):
    random.seed(SEED + worker_id)

def extract_prototype(model,dataloader,device='cuda',des_shapes=[16, 64, 128, 128], method='TSNE'):
    model.train()
    model.to(device)
    # down_scales = [1.0,0.5,0.25,0.125]
    down_scales = [0.5,0.25,0.125,0.125]
    num_class = 8
    loader = dataloader['train']
    total_batchs = len(loader)

    # ENet
    proto_des_1 = torch.zeros(num_class, 16 )
    proto_des_2 = torch.zeros(num_class, 64 )
    proto_des_3 = torch.zeros(num_class, 128)
    proto_des_4 = torch.zeros(num_class, 128)
    protos_des = [proto_des_1, proto_des_2, proto_des_3, proto_des_4]
    protos_out = []
    with torch.no_grad():
        for k in range(4):
            protos=[]
            labels=[]
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.float()
                outputs, up4, up3, up2, up1 = model(inputs)

                print_progress(
                    iteration=batch_idx+1,
                    total=total_batchs,
                    prefix=f'Extract Proto / Level {k} : Batch {batch_idx+1}/{total_batchs} ',
                    suffix=f'',
                    bar_length=45
                )  
                masks=targets.clone()
                up = [up1, up2, up3, up4]

                B,C,H,W = up[k].shape
                
                temp_masks = nn.functional.interpolate(masks.unsqueeze(dim=1), scale_factor=down_scales[k], mode='nearest')
                temp_masks = temp_masks.squeeze(dim=1)

                mask_unique_value = torch.unique(temp_masks)
                mask_unique_value = mask_unique_value[1:]
                unique_num = len(mask_unique_value)
                
                if unique_num<1:
                    continue

                for count,p in enumerate(mask_unique_value):
                    p = p.long()
                    bin_mask = torch.tensor(temp_masks==p,dtype=torch.int8)
                    bin_mask = bin_mask.unsqueeze(dim=1).expand_as(up[k])
                    temp = 0.0
                    batch_counter = 0
                    for t in range(B):
                        if torch.sum(bin_mask[t])!=0:
                            v = torch.sum(bin_mask[t]*up[k][t],dim=[1,2])/torch.sum(bin_mask[t],dim=[1,2])
                            temp = temp + temp + nn.functional.normalize(v, p=2.0, dim=0, eps=1e-12, out=None)
                            batch_counter = batch_counter + 1
                    temp = temp / batch_counter
                    protos.append(np.array(temp.detach().cpu()))
                    labels.append(p.item()) 
    
            protos = np.array(protos) 
            labels = np.array(labels)

            if method=='PCA':
                # pca = PCA(n_components = des_shapes[k])
                pca = PCA(n_components = 2)
                pca.fit(protos)
                protos = pca.transform(protos)
                protos = torch.tensor(protos)

                protos = torch.tensor(protos) 
                labels = torch.tensor(labels)
                protos_out.append([protos,labels])
                # for i in range(1, num_class+1):
                #     indexs = (labels==i)
                #     protos_des[k][i-1] = protos[indexs].mean(dim=0)
            elif method=='TSNE':
                protos = TSNE(n_components=2, learning_rate='auto', init='random', random_state='int').fit_transform(protos)
            else:
                assert f"{method} method hasn't been implemented."
    
    # torch.save(protos_des, '/content/UNet_V2/protos_file.pth')
    torch.save(protos_out, '/content/UNet_V2/protos_out_file.pth')
    

def get_CTranS_config(NUM_CLASS=None):
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    # config.transformer.embeddings_dropout_rate = 0.3
    # config.transformer.attention_dropout_rate  = 0.3
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate  = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    # config.patch_sizes = [8,4,2,1]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = NUM_CLASS
    return config

def main(args):
    MODEL_NAME = args.model_name
    NUM_CLASS = args.num_class
    IMAGE_HEIGHT = args.image_size
    IMAGE_WIDTH = args.image_size
    DEVICE = args.device

    train_tf = transforms.Compose([RandomGenerator(output_size=[IMAGE_HEIGHT, IMAGE_WIDTH])])
    val_tf = ValGenerator(output_size=[IMAGE_HEIGHT, IMAGE_WIDTH])
# LOAD_MODEL

    if MODEL_NAME=='UNet':
        model = UNet(n_channels=1, n_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME=='UNet_loss':
        model = UNet_loss(n_channels=1, n_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME=='U':
        model = U(bilinear=False).to(DEVICE)

    elif MODEL_NAME=='U_loss':
        model = U_loss(bilinear=False).to(DEVICE)

    elif MODEL_NAME=='UCTransNet':
        config_vit = get_CTranS_config(NUM_CLASS=NUM_CLASS)
        model = UCTransNet(config_vit,n_channels=n_channels,n_classes=n_labels,img_size=IMAGE_HEIGHT).to(DEVICE)

    elif MODEL_NAME=='UCTransNet_GT':
        config_vit = get_CTranS_config(NUM_CLASS=NUM_CLASS)
        model = UCTransNet_GT(config_vit,n_channels=n_channels,n_classes=n_labels,img_size=IMAGE_HEIGHT).to(DEVICE)

    elif MODEL_NAME=='GT_UNet':
        model = GT_U_Net(img_ch=1,output_ch=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME=='GT_CTrans':
        config_vit = get_CTranS_config(NUM_CLASS=NUM_CLASS)
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

    elif MODEL_NAME == 'ERFNet':
        model = ERFNet(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'ERFNet_loss':
        model = ERFNet_loss(num_classes=NUM_CLASS).to(DEVICE)
        
    else: 
        raise TypeError('Please enter a valid name for the model type')

    TASK_NAME = args.task_name

    num_parameters = utils.count_parameters(model)

    model_table = tabulate(
        tabular_data=[[MODEL_NAME, f'{num_parameters:.2f} M', DEVICE]],
        headers=['Builded Model', '#Parameters', 'Device'],
        tablefmt="fancy_grid"
        )


    CKPT_NAME = MODEL_NAME + '_' + TASK_NAME

    # checkpoint_path = '/content/drive/MyDrive/checkpoint_1/'+CKPT_NAME+'_best.pth'
    checkpoint_path = '/content/drive/MyDrive/checkpoint/'+CKPT_NAME+'_best.pth'

    print('Loading Checkpoint...')
    if os.path.isfile(checkpoint_path):
        pretrained_model_path = checkpoint_path
        loaded_data = torch.load(pretrained_model_path, map_location='cuda')
        pretrained = loaded_data['net']
        model2_dict = model.state_dict()
        state_dict = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)

        loaded_acc=loaded_data['acc']
        initial_best_epoch=loaded_data['best_epoch']

        table = tabulate(
                        tabular_data=[[loaded_acc, initial_best_epoch]],
                        headers=['Loaded Model Acc', 'Best Epoch Number'],
                        tablefmt="fancy_grid"
                        )
        print(table)
    else:
        print(f'No Such file : {checkpoint_path}')


    if TASK_NAME=='Synapse':

        train_dataset=Synapse_dataset(split='train', joint_transform=train_tf)

        train_loader = DataLoader(
                                train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                worker_init_fn=worker_init,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY,
                                drop_last=True,
                                )
        data_loader={'train':train_loader}

    elif TASK_NAME=='ACDC':

        train_dataset=ACDC(split='train', joint_transform=train_tf)

        train_loader = DataLoader(
                                train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                worker_init_fn=worker_init,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY,
                                drop_last=True,
                                )

        data_loader={'train':train_loader}

    elif TASK_NAME=='CT-1K':

        train_dataset=CT_1K(split='train', joint_transform=train_tf)

        train_loader = DataLoader(
                                train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                worker_init_fn=worker_init,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY,
                                drop_last=True,
                                )

        data_loader={'train':train_loader}
    
    extract_prototype(model,data_loader)


            



parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='AttUNet_loss')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--image_size', default=256)
parser.add_argument('--task_name',type=str, default='Synapse')
parser.add_argument('--num_class', default=9)
args = parser.parse_args()


if __name__ == "__main__":
    
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(666)    
    np.random.seed(666)  
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.cuda.manual_seed_all(666) 

    main(args)
    
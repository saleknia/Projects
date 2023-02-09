import argparse
import os
import numpy as np
import random
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from dataset import Synapse_dataset,ValGenerator
from torch.utils.data import Dataset
from models.UNet import UNet
from models.SEUNet import SEUNet
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm
import math
import h5py
import torchvision
import ml_collections


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/test_vol_h5', help='root dir for validation volume data')  
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
args = parser.parse_args()



class Synapse_dataset(Dataset):
    def __init__(self, split, joint_transform: Callable = None):

        if split == 'val_test':
            base_dir = '/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/test_vol_h5'        

        self.joint_transform = joint_transform

        if self.joint_transform:
            self.transform = joint_transform
        elif split=='val_test':
            self.transform = lambda x, y: (torch.tensor(x), torch.tensor(y))

        self.split = split
        self.sample_list = os.listdir(path=base_dir)
        self.sample_list.sort()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        
        vol_name = self.sample_list[idx].strip('\n')
        filepath = self.data_dir + "/{}".format(vol_name)
        data = h5py.File(filepath)
        image, mask = data['image'][:], data['label'][:]

        # image = zoom(image, (1, 224 / 512, 224 / 512), order=3)
        # mask  = zoom(mask , (1, 224 / 512, 224 / 512), order=0)

        sample = {'image': image, 'label': mask}

        # Data Augmentation
        if self.joint_transform:
            sample = self.transform(sample) 
        else:
            sample['image'],sample['label'] = self.transform(sample['image'],sample['label'])

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[224, 224], case=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list

# def test_single_volume(image, label, net, classes, patch_size=[224, 224], case=None):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))

#     return metric_list

# DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

def inference(args, model):
    db_test = Synapse_dataset(split='val_test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size], case=case_name)
        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":


    cudnn.benchmark = False
    cudnn.deterministic = True
    seed = 666

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '/content/UNet/MICCAI_2015_Multi_Atlas_Abdomen/test_vol_h5',
            'num_classes': 9,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.is_pretrain = True

    # net = UNet(n_channels=1, n_classes=9).cuda()
    # state=torch.load('/content/drive/MyDrive/checkpoint/UNet_Synapse_best.pth', map_location=torch.device('cuda'))
    # net.load_state_dict(state['net'])
    # acc=state['acc']
    # epoch=state['best_epoch']

    net = SEUNet(n_channels=1, n_classes=9).to('cuda')
    state=torch.load('/content/drive/MyDrive/checkpoint/SEUNet_Synapse_best.pth', map_location=torch.device('cuda'))
    net.load_state_dict(state['net'])
    acc=state['acc']
    epoch=state['best_epoch']

    print(f'Loaded Model Accuracy: {acc:.2f} , Epoch: {epoch}')    

    inference(args, net)

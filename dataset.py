import os
import sys
import numpy as np
from google.colab import files
import nibabel as nb
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
import zipfile
from zipfile import ZipFile as ZipFile
import utils
from utils import color
from utils import print_progress
from torchvision import transforms as T
from torchvision.transforms import functional as F
from scipy import ndimage
from typing import Callable
from scipy.ndimage.interpolation import zoom
import h5py
import os.path as osp
import numpy as np
import random
import cv2
from torch.utils import data
import pickle
from torch.utils.data import Dataset, DataLoader
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation
import skimage
import skimage.transform
from skimage.color import rgb2gray

# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)

    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized
       

class ISIC2017(Dataset):
    def __init__(self, path_Data='/content/drive/MyDrive/ISIC2017_dataset/', split='train'):
        super(ISIC2017, self)
        if split=='train':
            self.train = True
            self.data   = np.load(path_Data+'data_train.npy')
            self.mask   = np.load(path_Data+'mask_train.npy')
        elif split=='test':
            self.train = False
            self.data   = np.load(path_Data+'data_test.npy')
            self.mask   = np.load(path_Data+'mask_test.npy')
        elif split=='valid':
            self.train = False
            self.data   = np.load(path_Data+'data_val.npy')
            self.mask   = np.load(path_Data+'mask_val.npy')          
          
        # self.data   = dataset_normalized(self.data)
        self.mask   = np.expand_dims(self.mask, axis=3)
        self.mask   = self.mask /255.0
         
    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        # (256, 256, 3)
        if self.train:
            img, seg = self.apply_augmentation(img, seg)
        
        img, seg = self.resize(img, seg)
        
        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute(2, 0, 1)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        seg = seg.permute(2, 0, 1)
        
        return img, seg[0]
               
    # def apply_augmentation(self, img, seg):
    #     if random.random() < 0.5:
    #         img  = np.flip(img,  axis=1)
    #         seg  = np.flip(seg,  axis=1)
    #     return img, seg

    def apply_augmentation(self, img, seg):
        if random.random() > 0.5:
            img, seg = random_rot_flip(img, seg)
        if random.random() > 0.5:
            img, seg = random_rotate(img, seg)
        if random.random() > 0.5:
            img, seg = random_gray(img, seg)

        return img, seg

    def resize(self, img, seg):
        size = 224
        img = skimage.transform.resize(img, (size, size, 3))
        seg = skimage.transform.resize(seg, (size, size, 1))
        return img, seg

    def __len__(self):
        return len(self.data)
    


def shift_2d_replace(data, dx, dy, constant=0.0):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    :param data: The 2d numpy array to be shifted
    :param dx: The shift in x
    :param dy: The shift in y
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data

def random_shift(image, label):

    np_image = np.array(image)
    np_label = np.array(label)
    x, y = np_image.shape

    x_Shift = int(x * (np.random.rand()-0.5))
    y_Shift = int(y * (np.random.rand()-0.5))

    image = shift_2d_replace(data=np_image, dx=x_Shift, dy=y_Shift) 
    label = shift_2d_replace(data=np_label, dx=x_Shift, dy=y_Shift) 
    
    return image, label


def random_erase(image, label):

    np_image = np.array(image)
    np_label = np.array(label)


    w = np_image.shape[1] // 2
    h = np_image.shape[0] // 2

    x0 = random.randint(0, w)
    y0 = random.randint(0, h)

    np_image[y0:y0+h, x0:x0+w] = 0.0
    np_label[y0:y0+h, x0:x0+w] = 0.0

    return np_image, np_label

def random_crop(image, label):

    np_image = np.array(image)
    x, y = np_image.shape
    temp_image = np.zeros((x, y))
    temp_label = np.zeros((x, y))

    w = np_image.shape[1] // 2
    h = np_image.shape[0] // 2

    x0 = random.randint(0, w)
    y0 = random.randint(0, h)

    temp_image[y0:y0+h, x0:x0+w] = np_image[y0:y0+h, x0:x0+w]
    temp_label[y0:y0+h, x0:x0+w] = np_image[y0:y0+h, x0:x0+w]

    return image, label

def random_scale(image, label):

    np_image = np.array(image)
    np_label = np.array(label)
    x, y = np_image.shape

    scale = 0.5 + (0.5 * np.random.rand())

    image = zoom(image, scale, order=0) 
    label = zoom(label, scale, order=0)
    xp, yp = image.shape

    b_x = (x - xp)//2
    a_x = x - xp - b_x

    b_y = (y - yp)//2
    a_y = y - yp - b_y

    image = np.pad(image, ((b_y, a_y), (b_x, a_x)))
    label = np.pad(label, ((b_y, a_y), (b_x, a_x)))
    
    return image, label

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def random_gray(image, label):
    gray_image = rgb2gray(image)
    arrays = [gray_image for _ in range(3)]
    gray_image = np.stack(arrays, axis=-1)
    return gray_image, label

def crop_image(img,label,tol=0):
    img = np.array(img)
    label = np.array(label)
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    image, label = img[np.ix_(mask.any(1),mask.any(0))], label[np.ix_(mask.any(1),mask.any(0))]
    image = zoom(image, (256.0 / image.shape[0], 256.0 / image.shape[1]), order=0)  # why not 3?
    label = zoom(label, (256.0 / label.shape[0], 256.0 / label.shape[1]), order=0)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)


        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

class COVID_19(Dataset):
    def __init__(self,joint_transform: Callable = None,split='train'):

        self.split = split
        self.root = '/content/UNet/COVID-19'

        self.joint_transform = joint_transform

        if self.joint_transform:
            self.transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.transform = lambda x, y: (to_tensor(x), to_tensor(y))

        if self.split=='train':
            self.image_dir = os.path.join(self.root,'train','ct_scans')
            self.mask_dir = os.path.join(self.root,'train','ct_masks')

        if self.split=='test':
            self.image_dir = os.path.join(self.root,'test','ct_scans')
            self.mask_dir = os.path.join(self.root,'test','ct_masks')

    def __len__(self):
        return len(os.listdir(path=self.image_dir))
    
    def __getitem__(self,index):
        images_path = self.image_dir
        images_name = os.listdir(images_path)
        images_name.sort()
        image = np.load(file=os.path.join(images_path,images_name[index]))

        masks_path = self.mask_dir
        masks_name = os.listdir(masks_path)
        masks_name.sort()
        mask = np.load(file=os.path.join(masks_path,masks_name[index]))
        

        sample = {'image': image, 'label': mask}

        # Data Augmentation
        if self.joint_transform:
            sample = self.transform(sample) 
        else:
            sample['image'],sample['label'] = self.transform(sample['image'],sample['label'])

        image,mask = sample['image'],sample['label'] 

        return image,mask

# class COVID_19(Dataset):
#     '''
#     if Download=False you sholud prepare root directory with the
#     root/train & root/train_masks.
#     if Download=True you should upload your kaggle.json to download
#     dataset to the root/train & root/masks folders. 
#     '''
#     def __init__(self,download=True,transform=None):
#         if not os.path.isdir('/content/UNet/COVID-19'):
#             os.system('mkdir -p /content/UNet/COVID-19')
#         self.root = '/content/UNet/COVID-19'
#         self.transform=transform
#         self.scaler = MinMaxScaler(feature_range=(0,1))

#         if download:
#             print(color.BOLD,color.RED)
#             print('\rDownloading Dataset...',color.END)
#             self.ct_download_path,self.mask_download_path = self.download_metadata()
#             self.ct_download_path.sort()
#             self.mask_download_path.sort()
#             self.ct_path,self.mask_path = self.download_data(self.ct_download_path[0:10],self.mask_download_path[0:10])
#             print(color.BOLD,color.RED)
#             print('\rDataset Downloaded.',color.END)

#         self.image_dir = os.path.join(self.root,'ct_scans')
#         self.mask_dir = os.path.join(self.root,'ct_masks')

#     def download_metadata(self):
#         os.system('mkdir ~/.kaggle')
#         os.system('cp kaggle.json ~/.kaggle/')
#         os.system('chmod 600 ~/.kaggle/kaggle.json')
#         os.system(f'kaggle datasets download -d andrewmvd/covid19-ct-scans -f metadata.csv -p {self.root}')
#         metadata_path = os.path.join(self.root,'metadata.csv')
#         csv_file = pd.read_csv(metadata_path)

#         ct_download = np.array(csv_file['ct_scan'])
#         ct_download_path = [x.split('ct_scans/')[1] for x in ct_download]

#         mask_download = np.array(csv_file['lung_and_infection_mask'])
#         mask_download_path = [x.split('lung_and_infection_mask/')[1] for x in mask_download]
#         return ct_download_path,mask_download_path

#     def download_data(self,ct_download_path,mask_download_path):
#         pwd = os.getcwd()
#         ct_path = os.path.join(self.root,'ct_scans')
#         mask_path = os.path.join(self.root,'ct_masks')
#         os.makedirs(name=ct_path,exist_ok=True)
#         os.makedirs(name=mask_path,exist_ok=True)

#         os.chdir(path=ct_path)
#         for case_num,ct in enumerate(ct_download_path):
#             os.system(f'kaggle datasets download andrewmvd/covid19-ct-scans -f ct_scans/{ct}')
#             zip_path = ct + '.zip'
#             with ZipFile(zip_path, 'r') as myzip:
#                 myzip.extractall(path=None, members=None, pwd=None) 
#             os.remove(path=zip_path)
#             sample_path = ct
#             sample = nb.load(filename=sample_path).get_fdata()
#             sample = np.clip(a=sample,a_min=-650,a_max=250)
#             sample = self.scaler.fit_transform(sample.reshape(-1,sample.shape[-1])).reshape(sample.shape)
#             sample = sample.astype(dtype=np.float32)
#             num_slices = sample.shape[2]
#             for s in range(num_slices):
#                 slice_name = ct.split('.')[0]+'_'+'slice_'+str(s)
#                 slice_name = slice_name.split('org_')[0]+slice_name.split('org_')[1]
#                 np.save(slice_name,arr=sample[:,:,s]) 
#                 print_progress(
#                     iteration=s+1,
#                     total=num_slices,
#                     prefix=f'CT Case {case_num+1}',
#                     suffix=f'Slice {s+1}',
#                     bar_length=70
#                 )  
#             os.remove(path=ct) 
#         os.chdir(pwd) 
#         os.chdir(path=mask_path)
#         for mask_num,mask in enumerate(mask_download_path):
#             os.system(f'kaggle datasets download andrewmvd/covid19-ct-scans -f lung_and_infection_mask/{mask}')
#             zip_path = mask + '.zip'
#             with ZipFile(zip_path, 'r') as myzip:
#                 myzip.extractall(path=None, members=None, pwd=None) 
#             os.remove(path=zip_path)            
#             sample_path = mask 
#             sample = nb.load(filename=sample_path).get_fdata()
#             sample = sample.astype(dtype=np.float32)
#             sample[sample==2.0]=1.0
#             sample[sample==3.0]=2.0
#             num_slices = sample.shape[2]
#             for s in range(num_slices):
#                 slice_name = mask.split('.')[0]+'_'+'slice_'+str(s)
#                 np.save(slice_name,arr=sample[:,:,s])  
#                 print_progress(
#                     iteration=s+1,
#                     total=num_slices,
#                     prefix=f'Mask Case {mask_num+1}',
#                     suffix=f'Slice {s+1}',
#                     bar_length=70
#                 )  
#             os.remove(mask)        
#         os.chdir(path=pwd)        
#         return ct_path,mask_path

#     def __len__(self):
#         return len(os.listdir(path=self.image_dir))
    
#     def __getitem__(self,index):
#         images_path = self.image_dir
#         images_name = os.listdir(images_path)
#         images_name.sort()
#         image = np.load(file=os.path.join(images_path,images_name[index]))
#         image = image.transpose()

#         masks_path = self.mask_dir
#         masks_name = os.listdir(masks_path)
#         masks_name.sort()
#         mask = np.load(file=os.path.join(masks_path,masks_name[index]))
#         mask = mask.transpose()
        
#         if self.transform is not None: 
#             # Data Augmentation
#             augmentation=self.transform(image=image,mask=mask)
#             image=augmentation['image']
#             mask=augmentation['mask']
 
#         return image,mask


class Synapse_dataset(Dataset):
    def __init__(self, split, joint_transform: Callable = None):
        if split == 'train': 
            base_dir = '/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/train_npz'
        if split == 'val':
            base_dir = '/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/test_npz'
        if split == 'val_test':
            base_dir = '/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/test_vol_h5'        

        self.joint_transform = joint_transform

        if self.joint_transform:
            self.transform = joint_transform
        elif split=='val' or split=='train':
            to_tensor = T.ToTensor()
            self.transform = lambda x, y: (to_tensor(x), to_tensor(y))
        elif split=='val_test':
            self.transform = lambda x, y: (torch.tensor(x), torch.tensor(y))

        self.split = split
        self.sample_list = os.listdir(path=base_dir)
        self.sample_list.sort()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, slice_name)
        data = np.load(data_path)
        image, mask = data['image'], data['label']
        
        if self.split == 'train':
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)

            image, mask = data['image'], data['label']

        elif self.split == 'val':
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)

            image, mask = data['image'], data['label']

        elif self.split == 'val_test':
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}".format(vol_name)
            data = h5py.File(filepath)
            image, mask = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': mask}

        # Data Augmentation
        if self.joint_transform:
            sample = self.transform(sample) 
        else:
            sample['image'],sample['label'] = self.transform(sample['image'],sample['label'])

        image,mask = sample['image'],sample['label'] 

        if self.split == 'train' or self.split == 'val':
            return image,mask
        elif self.split == 'val_test':
            sample['case_name'] = self.sample_list[idx].strip('\n')
            return sample


class CamVidDataSet(data.Dataset):
    """ 
       CamVidDataSet is employed to load train set
       Args:
        root: the CamVid dataset path, 
        list_path: camvid_train_list.txt, include partial path
    """

    def __init__(self, root='', list_path='', max_iters=None, crop_size=(360, 360),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=11):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            # print(img_file)
            label_file = osp.join(self.root, name.split()[1])
            # print(label_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        print("length of train set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            scale = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  # random resize between 0.5 and 2
            f_scale = scale[random.randint(0, 5)]
            # f_scale = 0.5 + random.randint(0, 15) / 10.0  #random resize between 0.5 and 2
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)

        image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))  # NHWC -> NCHW

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name



class ACDC(Dataset):
    def __init__(self, split, joint_transform: Callable = None):
        if split == 'train': 
            base_dir = '/content/UNet_V2/ACDC/train'
        if split == 'test':
            base_dir = '/content/UNet_V2/ACDC/test'


        self.joint_transform = joint_transform

        if self.joint_transform:
            self.transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.transform = lambda x, y: (to_tensor(x), to_tensor(y))

        self.split = split
        self.sample_list = os.listdir(path=base_dir)
        self.sample_list.sort()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        slice_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, slice_name)
        data = np.load(data_path)
        image, mask = data['image'], data['label']

        image = zoom(image, (256.0 / image.shape[0], 256.0 / image.shape[1]))
        mask = zoom(mask, (256.0 / mask.shape[0], 256.0 / mask.shape[1]))

        sample = {'image': image, 'label': mask}

        # Data Augmentation
        if self.joint_transform:
            sample = self.transform(sample) 
        else:
            sample['image'],sample['label'] = self.transform(sample['image'],sample['label'])

        image,mask = sample['image'],sample['label'] 


        return image,mask


class CT_1K(Dataset):
    def __init__(self, split, joint_transform: Callable = None):
        if split == 'train': 
            base_dir = '/content/UNet_V2/CT-1K/train'
        if split == 'valid': 
            base_dir = '/content/UNet_V2/CT-1K/valid'
        if split == 'test':
            base_dir = '/content/UNet_V2/CT-1K/test'


        self.joint_transform = joint_transform

        if self.joint_transform:
            self.transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.transform = lambda x, y: (to_tensor(x), to_tensor(y))

        self.split = split
        self.sample_list = os.listdir(path=base_dir)
        self.sample_list.sort()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        slice_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, slice_name)
        data = np.load(data_path)
        image, mask = data['image'], data['label']

        sample = {'image': image, 'label': mask}

        # Data Augmentation
        if self.joint_transform:
            sample = self.transform(sample) 
        else:
            sample['image'],sample['label'] = self.transform(sample['image'],sample['label'])

        image,mask = sample['image'],sample['label'] 


        return image,mask

class TCIA(Dataset):
    def __init__(self, split, joint_transform: Callable = None):
        if split == 'train': 
            base_dir = '/content/UNet_V2/TCIA/train'
        if split == 'valid': 
            base_dir = '/content/UNet_V2/TCIA/valid'
        if split == 'test':
            base_dir = '/content/UNet_V2/TCIA/test'

        self.joint_transform = joint_transform

        if self.joint_transform:
            self.transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.transform = lambda x, y: (to_tensor(x), to_tensor(y))

        self.split = split
        self.sample_list = os.listdir(path=base_dir)
        self.sample_list.sort()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        slice_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, slice_name)
        data = np.load(data_path)
        image, mask = data['image'], data['label']

        sample = {'image': image, 'label': mask}

        # Data Augmentation
        if self.joint_transform:
            sample = self.transform(sample) 
        else:
            sample['image'],sample['label'] = self.transform(sample['image'],sample['label'])

        image,mask = sample['image'],sample['label'] 


        return image,mask


class SSL(Dataset):
    def __init__(self, joint_transform: Callable = None):

        base_dir = '/content/UNet_V2/SSL'

        self.joint_transform = joint_transform

        if self.joint_transform:
            self.transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.transform = lambda x, y: (to_tensor(x), to_tensor(y))

        self.sample_list = os.listdir(path=base_dir)
        self.sample_list.sort()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        slice_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, slice_name)
        data = np.load(data_path)
        image, mask = data['image'], data['label']

        sample = {'image': image, 'label': mask}

        # Data Augmentation
        if self.joint_transform:
            sample = self.transform(sample) 
        else:
            sample['image'],sample['label'] = self.transform(sample['image'],sample['label'])

        image,mask = sample['image'],sample['label'] 


        return image,mask


# class Synapse_dataset(Dataset):
#     def __init__(self, split, index, joint_transform: Callable = None):
#         if split == 'train' or split == 'val': 
#             base_dir = '/content/UNet/MICCAI_2015_Multi_Atlas_Abdomen/train_npz'
#         # if split == 'val':
#         #     base_dir = '/content/UNet/MICCAI_2015_Multi_Atlas_Abdomen/test_npz'
#         if split == 'val_test':
#             base_dir = '/content/UNet/MICCAI_2015_Multi_Atlas_Abdomen/test_vol_h5'        

#         self.joint_transform = joint_transform

#         if self.joint_transform:
#             self.transform = joint_transform
#         elif split=='val' or split=='train':
#             to_tensor = T.ToTensor()
#             self.transform = lambda x, y: (to_tensor(x), to_tensor(y))
#         elif split=='val_test':
#             self.transform = lambda x, y: (torch.tensor(x), torch.tensor(y))

#         self.split = split
#         self.sample_list = os.listdir(path=base_dir)
#         self.sample_list.sort()
#         self.index = index
#         self.data_dir = base_dir

#     def __len__(self):
#         # return len(self.sample_list)
#         return len(self.index)

#     def __getitem__(self, idx):
#         index = self.index[idx]
#         slice_name = self.sample_list[index]
#         data_path = os.path.join(self.data_dir, slice_name)
#         data = np.load(data_path)
#         image, mask = data['image'], data['label']
        
#         # if self.split == 'train':
#         #     slice_name = self.sample_list[idx]
#         #     data_path = os.path.join(self.data_dir, slice_name)
#         #     data = np.load(data_path)

#         #     image, mask = data['image'], data['label']
#         #     # image = np.flip(m=image,axis=0)
#         #     # mask = np.flip(m=mask,axis=0)
#         #     # image = np.rot90(m=image,k=1)
#         #     # mask = np.rot90(m=mask,k=1)

#         # elif self.split == 'val':
#         #     slice_name = self.sample_list[idx]
#         #     data_path = os.path.join(self.data_dir, slice_name)
#         #     data = np.load(data_path)

#         #     image, mask = data['image'], data['label']
#         #     # image = np.flip(m=image,axis=0)
#         #     # mask = np.flip(m=mask,axis=0)
#         #     # image = np.rot90(m=image,k=1)
#         #     # mask = np.rot90(m=mask,k=1) 

#         # elif self.split == 'val_test':
#         #     vol_name = self.sample_list[idx].strip('\n')
#         #     filepath = self.data_dir + "/{}".format(vol_name)
#         #     data = h5py.File(filepath)
#         #     image, mask = data['image'][:], data['label'][:]

#         sample = {'image': image, 'label': mask}

#         # Data Augmentation
#         if self.joint_transform:
#             sample = self.transform(sample) 
#         else:
#             sample['image'],sample['label'] = self.transform(sample['image'],sample['label'])

#         image,mask = sample['image'],sample['label'] 

#         if self.split == 'train' or self.split == 'val':
#             return image,mask
#         elif self.split == 'val_test':
#             sample['case_name'] = self.sample_list[idx].strip('\n')
#             return sample


# class Synapse_dataset(Dataset):
#     def __init__(self, split, joint_transform: Callable = None):
#         if split == 'train': 
#             base_dir = '/content/UNet/MICCAI_2015_Multi_Atlas_Abdomen/train_npz'
#         if split == 'val':
#             base_dir = '/content/UNet/MICCAI_2015_Multi_Atlas_Abdomen/test_npz'

#         if joint_transform:
#             self.joint_transform = joint_transform
#         else:
#             to_tensor = T.ToTensor()
#             self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

#         self.split = split
#         self.sample_list = os.listdir(path=base_dir)
#         self.sample_list.sort()
#         self.data_dir = base_dir

#     def __len__(self):
#         return len(self.sample_list)

#     def __getitem__(self, idx):
#         if self.split == 'train':
#             slice_name = self.sample_list[idx]
#             data_path = os.path.join(self.data_dir, slice_name)
#             data = np.load(data_path)

#             image, mask = data['image'], data['label']
#             # image = np.flip(m=image,axis=0)
#             # mask = np.flip(m=mask,axis=0)
#             # image = np.rot90(m=image,k=1)
#             # mask = np.rot90(m=mask,k=1)
            
#         elif self.split == 'val':
#             slice_name = self.sample_list[idx]
#             data_path = os.path.join(self.data_dir, slice_name)
#             data = np.load(data_path)

#             image, mask = data['image'], data['label']
#             # image = np.flip(m=image,axis=0)
#             # mask = np.flip(m=mask,axis=0)
#             # image = np.rot90(m=image,k=1)
#             # mask = np.rot90(m=mask,k=1)

#         sample = {'image': image, 'label': mask}
#         # Data Augmentation
#         if self.joint_transform:
#             sample = self.joint_transform(sample) 

#         image,mask = sample['image'],sample['label'] 

#         # if (self.transform is not None): 
#         #     # Data Augmentation
#         #     augmentation=self.transform(image=image,mask=mask)
#         #     image=augmentation['image']
#         #     mask=augmentation['mask']

#         return image , mask

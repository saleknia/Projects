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

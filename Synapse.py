class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[34m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

import os
os.system('pip install synapseclient')
import synapseclient 
import sys
import argparse
import wget
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from os import listdir
from scipy import ndimage
from PIL import Image
# import albumentations as A
import h5py
import zipfile
from zipfile import ZipFile as ZipFile
import shutil

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    # bar = '█' * filled_length + ' ' * (bar_length - filled_length)
    bar = '■' * filled_length + '□' * (bar_length - filled_length)

    sys.stdout.write('\r%s |\033[34m%s\033[0m| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def list_download(download_path,show=True):
    train_list = wget.download(
        url = 'https://raw.githubusercontent.com/Beckschen/TransUNet/main/lists/lists_Synapse/train.txt',
        out = download_path)
    test_vol_list = wget.download(
        url = 'https://raw.githubusercontent.com/Beckschen/TransUNet/main/lists/lists_Synapse/test_vol.txt',
        out = download_path)
    if show:
        print(color.BLUE,color.BOLD,'Train List: ',color.END,train_list)
        print(color.BLUE,color.BOLD,'Test Vol List: ',color.END,test_vol_list)

def synapse_download(user_name, password, path):
    pwd = os.getcwd()
    os.makedirs(name=path,exist_ok=True)
    os.chdir(path=path)
    syn = synapseclient.Synapse() 
    syn.login(user_name,password) 
    # download file into current working directory
    entity = syn.get('syn3379050', downloadLocation='.')
    print('\r',color.BLUE,color.BOLD,'Downloaded File: ',color.END,entity.name)
    print('\r',color.BLUE,color.BOLD,'Download Path: ',color.END,entity.path)
    os.chdir(path=pwd)

def unzip_dataset(zip_path='/content/MICCAI_2015_Multi_Atlas_Abdomen/RawData.zip',
                  extract_path='/content/MICCAI_2015_Multi_Atlas_Abdomen'):

    pwd = os.getcwd()
    os.chdir(path=extract_path)

    with ZipFile(zip_path,'r') as myzip:
        myzip.extractall(path=extract_path, members=None, pwd=None)

    os.chdir(path=pwd)
    os.remove(path=zip_path)
    # shutil.rmtree(os.path.join(extract_path,'RawData','Testing'))


# formal

idx2cls_org =[
          'Background',
          'spleen',
          'right kidney',
          'left kidney',
          'gallbladder',
          'esophagus',
          'liver',
          'stomach',
          'aorta',
          'infereior vena cava',
          'portal vein',
          'pancreas',
          'right adrenal gland',
          'left adrenal gland'
]

idx2cls_made =[
          'Background',
          'aorta',
          'gallbladder',
          'left kidney',
          'right kidney',
          'liver',
          'pancreas',
          'spleen',
          'stomach',
]

idx2cls_org = np.array(idx2cls_org)

idx2cls_made = np.array(idx2cls_made)

palate=[
        (0,0,0),
        (51,153,255),
        (0,255,0),
        (255,0,0),
        (0,255,255),
        (255,0,255),
        (255,255,0),
        (126,0,255),
        (255,126,0)
]
palate = np.array(palate,dtype=np.float32)/255.0

def masking(image,label,palate):
    assert image.shape==label.shape,f'Dimesion Mismatch: label Dim={label.shape}, img Dim={image.shape}'
    row,col = image.shape
    image_expand = np.expand_dims(image,axis=2)
    temp = np.concatenate((image_expand,image_expand,image_expand),axis=2)
    label = label.astype(dtype=np.uint8)
    for r in range(row):
        for c in range(col):
            if label[r,c]!=0:
                temp[r,c] = palate[label[r,c]]
    return temp


def count_img_slice(images_path='/content/MICCAI_2015_Multi_Atlas_Abdomen/RawData/Training/img'):
    from os import listdir
    images_names = listdir(path=images_path)
    images_names.sort()
    sum = 0
    for name in images_names:
        sample_nib = nib.load(filename=os.path.join(images_path,name)).get_fdata()
        sum += sample_nib.shape[2]
        print(name,' ',sample_nib.shape[2])
    print('Number of slices according to the images: ',sum)


def count_label_slice(labels_path='/content/MICCAI_2015_Multi_Atlas_Abdomen/RawData/Training/label'):
    from os import listdir
    labels_names = listdir(path=labels_path)
    labels_names.sort()
    sum = 0
    for name in labels_names:
        sample_nib = nib.load(filename=os.path.join(labels_path,name)).get_fdata()
        sum += sample_nib.shape[2]
        print(name,' ',sample_nib.shape[2])
    print('Number of slices according to the labels: ',sum)

# all_indexes = [1,2,3,4,5,6,7,8,9,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]

# train_index=[5,6,7,9,10,21,23,24,26,27,28,30,31,33,34,37,39,40]
# test_index=[1,2,3,4,8,22,25,29,32,35,36,38]

train_index = [1,2,3,4,5,6,7,8,9,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
lookup_table = [0, ]

def formal(index,lenght):
    index=str(index)
    index='0'*(lenght-len(index))+index
    return index

def making_train_npz(path='/content/data/Synapse/train_npz',
                     samples_path = '/content/MICCAI_2015_Multi_Atlas_Abdomen/RawData/Training',
                     lookup_table = [0,7,4,3,2,0,5,8,1,0,0,6,0,0]):
    
    train_index=[5,6,7,9,10,21,23,24,26,27,28,30,31,33,34,37,39,40]
    scaler = MinMaxScaler(feature_range=(0,1))

    training_samples_path = os.path.join(samples_path,'img')
    training_samples_names = listdir(path=training_samples_path)
    training_samples_names.sort()

    training_labels_path = os.path.join(samples_path,'label')
    training_labels_names = listdir(path=training_labels_path)
    training_labels_names.sort()

    os.makedirs(name=path,exist_ok=True)
    pwd = os.getcwd()
    os.chdir(path=path)

    for (sample_name,label_name) in zip(training_samples_names,training_labels_names):
        sample_index = int(sample_name.split('.')[0].split('img')[1])
        label_index = int(label_name.split('.')[0].split('label')[1])
        
        if sample_index in train_index:
            sample_path = os.path.join(training_samples_path,sample_name)
            label_path = os.path.join(training_labels_path,label_name)

            sample = nib.load(sample_path).get_fdata()
            sample = np.clip(a=sample,a_min=-125,a_max=175)
            sample = scaler.fit_transform(sample.reshape(-1,sample.shape[-1])).reshape(sample.shape)


            label = nib.load(label_path).get_fdata()

            num_sample_slices = sample.shape[2]
            num_label_slices = label.shape[2]

            assert num_sample_slices==num_label_slices,f'Dimesion Mismatch: {sample_name} , label Dim={num_label_slices}, img Dim={num_sample_slices}'

            for index in range(num_sample_slices):
                slice_name='case'+formal(sample_index,lenght=4)+'_slice'+formal(index,lenght=3)+'.npz'

                print_progress(iteration=index+1,
                               total=num_sample_slices,
                               prefix='Case '+formal(sample_index,lenght=4),
                               suffix='Slice '+formal(index,lenght=3),
                               decimals=1, 
                               bar_length=50)

                slice_2d = sample[:,:,index]
                slice_2d = slice_2d.astype(dtype=np.float32)

                label_2d = label[:,:,index]
                label_2d = label_2d.astype(dtype=np.float32)

                # slice_2d = np.rot90(m=slice_2d,k=1)
                # label_2d = np.rot90(m=label_2d,k=1)

                # augmentations = horizontal_flip(image=slice_2d, mask=label_2d)
                # slice_2d = augmentations["image"]
                # label_2d = augmentations["mask"]

                # augmentations = Resize(image=slice_2d, mask=label_2d)
                # slice_2d = augmentations["image"]
                # label_2d = augmentations["mask"]

                # lookup_table=[0,7,4,3,2,0,5,8,1,0,0,6,0,0]

                label_2d = label_2d.astype(dtype=np.uint8)
                row,col = label_2d.shape
                for r in range(row):
                    for c in range(col):
                        label_2d[r,c] = lookup_table[label_2d[r,c]]
                label_2d = label_2d.astype(dtype=np.float32)

                # slice_2d = np.flip(m=slice_2d,axis=1)
                # label_2d = np.flip(m=label_2d,axis=1)

                np.savez(file=slice_name,image=slice_2d,label=label_2d)
        # if sample_index in train_index:
        #     break
    os.chdir(path=pwd)


def making_test_vol_h5(path='/content/data/Synapse/test_vol_h5',
                     samples_path = '/content/MICCAI_2015_Multi_Atlas_Abdomen/RawData/Training',
                     lookup_table = [0,7,4,3,2,0,5,8,1,0,0,6,0,0]):

    scaler = MinMaxScaler(feature_range=(0,1))
    test_index=[1,2,3,4,8,22,25,29,32,35,36,38]

    testing_samples_path = os.path.join(samples_path,'img')
    testing_samples_names = listdir(path=testing_samples_path)
    testing_samples_names.sort()

    testing_labels_path = os.path.join(samples_path,'label')
    testing_labels_names = listdir(path=testing_labels_path)
    testing_labels_names.sort()

    os.makedirs(name=path,exist_ok=True)
    pwd = os.getcwd()
    os.chdir(path=path)
    counter=0
    for (sample_name,label_name) in zip(testing_samples_names,testing_labels_names):
        sample_index = int(sample_name.split('.')[0].split('img')[1])
        label_index = int(label_name.split('.')[0].split('label')[1])
        if sample_index in test_index:
            counter+=1
            sample_path = os.path.join(testing_samples_path,sample_name)
            label_path = os.path.join(testing_labels_path,label_name)

            sample = nib.load(sample_path).get_fdata()
            sample = np.clip(a=sample,a_min=-125,a_max=175)
            sample = scaler.fit_transform(sample.reshape(-1,sample.shape[-1])).reshape(sample.shape)

            label = nib.load(label_path).get_fdata()

            num_sample_slices = sample.shape[2]
            num_label_slices = label.shape[2]

            assert num_sample_slices==num_label_slices,f'Dimesion Mismatch: {sample_name} , label Dim={num_label_slices}, img Dim={num_sample_slices}'


            slice_3d_name='case'+formal(sample_index,lenght=4)+'.npy.h5'

            print_progress(iteration=counter,
                           total=len(test_index),
                           prefix='Case '+formal(sample_index,lenght=4),
                           decimals=1, 
                           bar_length=50)

            slice_3d = sample.astype(dtype=np.float32)
            label_3d = label.astype(dtype=np.float32)

            # augmentations = Resize(image=slice_3d, mask=label_3d)
            # slice_3d = augmentations["image"]
            # label_3d = augmentations["mask"]

            # lookup_table=[0,7,4,3,2,0,5,8,1,0,0,6,0,0]

            # label_3d = label_3d.astype(dtype=np.uint8)
            # row,col,dim = label_3d.shape
            # for r in range(row):
            #     for c in range(col):
            #         for d in range(dim):
            #             label_3d[r,c,d] = lookup_table[label_3d[r,c,d]]

            # label_3d = label_3d.astype(dtype=np.float32)

            # slice_3d = slice_3d.transpose((2,0,1))
            # label_3d = label_3d.transpose((2,0,1))

            # slice_3d = np.flip(m=slice_3d,axis=2)
            # label_3d = np.flip(m=label_3d,axis=2)

            h5f = h5py.File(slice_3d_name, 'w')
            h5f.create_dataset('image', data=slice_3d)
            h5f.create_dataset('label', data=label_3d)
            h5f.close()
    os.chdir(path=pwd)

def making_test_npz(path='/content/data/Synapse/test_vol_h5',
                     samples_path = '/content/MICCAI_2015_Multi_Atlas_Abdomen/RawData/Training',
                     lookup_table = [0,7,4,3,2,0,5,8,1,0,0,6,0,0]):

    scaler = MinMaxScaler(feature_range=(0,1))
    test_index=[1,2,3,4,8,22,25,29,32,35,36,38]

    testing_samples_path = os.path.join(samples_path,'img')
    testing_samples_names = listdir(path=testing_samples_path)
    testing_samples_names.sort()

    testing_labels_path = os.path.join(samples_path,'label')
    testing_labels_names = listdir(path=testing_labels_path)
    testing_labels_names.sort()

    os.makedirs(name=path,exist_ok=True)
    pwd = os.getcwd()
    os.chdir(path=path)


    for (sample_name,label_name) in zip(testing_samples_names,testing_labels_names):
        sample_index = int(sample_name.split('.')[0].split('img')[1])
        label_index = int(label_name.split('.')[0].split('label')[1])
        
        if sample_index in test_index:
            sample_path = os.path.join(testing_samples_path,sample_name)
            label_path = os.path.join(testing_labels_path,label_name)

            sample = nib.load(sample_path).get_fdata()
            sample = np.clip(a=sample,a_min=-125,a_max=175)
            sample = scaler.fit_transform(sample.reshape(-1,sample.shape[-1])).reshape(sample.shape)


            label = nib.load(label_path).get_fdata()

            num_sample_slices = sample.shape[2]
            num_label_slices = label.shape[2]

            assert num_sample_slices==num_label_slices,f'Dimesion Mismatch: {sample_name} , label Dim={num_label_slices}, img Dim={num_sample_slices}'

            for index in range(num_sample_slices):
                slice_name='case'+formal(sample_index,lenght=4)+'_slice'+formal(index,lenght=3)+'.npz'

                print_progress(iteration=index+1,
                               total=num_sample_slices,
                               prefix='Case '+formal(sample_index,lenght=4),
                               suffix='Slice '+formal(index,lenght=3),
                               decimals=1, 
                               bar_length=50)

                slice_2d = sample[:,:,index]
                slice_2d = slice_2d.astype(dtype=np.float32)

                label_2d = label[:,:,index]
                label_2d = label_2d.astype(dtype=np.float32)

                # slice_2d = np.rot90(m=slice_2d,k=1)
                # label_2d = np.rot90(m=label_2d,k=1)

                # augmentations = horizontal_flip(image=slice_2d, mask=label_2d)
                # slice_2d = augmentations["image"]
                # label_2d = augmentations["mask"]

                # augmentations = Resize(image=slice_2d, mask=label_2d)
                # slice_2d = augmentations["image"]
                # label_2d = augmentations["mask"]

                # lookup_table=[0,7,4,3,2,0,5,8,1,0,0,6,0,0]

                # label_2d = label_2d.astype(dtype=np.uint8)
                # row,col = label_2d.shape
                # for r in range(row):
                #     for c in range(col):
                #         label_2d[r,c] = lookup_table[label_2d[r,c]]
                # label_2d = label_2d.astype(dtype=np.float32)

                # slice_2d = np.flip(m=slice_2d,axis=1)
                # label_2d = np.flip(m=label_2d,axis=1)

                np.savez(file=slice_name,image=slice_2d,label=label_2d)
        # if sample_index in train_index:
        #     break
    os.chdir(path=pwd)

def synapse_prepare(combine=False, test_index=[1,2,3,4,8,22,25,29,32,35,36,38]):
    if os.path.isdir(s='/content/project_TransUNet')==False:
        os.system('unzip -qq /content/drive/MyDrive/project_TransUNet.zip -d /content')
    os.system('mkdir /content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen')
    os.system('cp -r /content/project_TransUNet/data/Synapse/train_npz /content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen')
    os.system('cp -r /content/project_TransUNet/data/Synapse/test_vol_h5 /content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen')

    pwd = os.getcwd()
    
    if combine:
        os.chdir(path='/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/train_npz')
    else:
        os.system('mkdir /content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/test_npz')
        os.chdir(path='/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/test_npz')

    files_path = '/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/test_vol_h5'
    files = os.listdir(path=files_path)

    for file_name in files:
        file_index = int(file_name.split('.npy.h5')[0].split('case')[1])
        file_path = os.path.join(files_path,file_name)
        f = h5py.File(name=file_path,mode='r')
        num_sample_slices = f['image'].shape[0]
        sample = np.array(f['image'])
        label = np.array(f['label'])

        for index in range(num_sample_slices):
            slice_name='case'+formal(file_index,lenght=4)+'_slice'+formal(index,lenght=3)+'.npz'

            slice_2d = sample[index,:,:]
            slice_2d = slice_2d.astype(dtype=np.float32)

            label_2d = label[index,:,:]
            label_2d = label_2d.astype(dtype=np.float32)

            np.savez(file=slice_name,image=slice_2d,label=label_2d)
    if combine:
        os.system('mkdir /content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/test_npz')
        train_files_path = '/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/train_npz'
        test_files_path = '/content/UNet_V2/MICCAI_2015_Multi_Atlas_Abdomen/test_npz'
        all_files = os.listdir(path=train_files_path)
        for f in all_files:
            index = int(f.split('_slice')[0].split('case')[1])
            if index in test_index:
                shutil.move(f'{train_files_path+"/"+f}', test_files_path)

    os.chdir(path=pwd)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='.', help='root dir for data')
parser.add_argument('--download', type=str,
                    default='True', help='download or copy from drive')
parser.add_argument('--combine', type=str,
                    default='False', help='Combine Samples')
parser.add_argument('--fold', type=int,
                    default=1, help='Cross fold Validation')
parser.add_argument('--user_name', type=str,
                    default='amirsaleknia', help='Synapse Account Username')
parser.add_argument('--password', type=str,
                    default='28103725a@A', help='Synapse Account Password')
args = parser.parse_args()

if __name__ == "__main__":
    if args.root_path == '.':
        path = os.getcwd()
    else:
        path = args.root_path

    user_name = args.user_name
    password = args.password
    fold = args.fold

    if args.combine=='False':
        combine = False
    else:
        combine = True

    if args.download=='True':
        dataset_name = 'MICCAI_2015_Multi_Atlas_Abdomen'
        download_path = os.path.join(path,dataset_name)
        os.makedirs(download_path,exist_ok=True)

        zip_path = os.path.join(download_path,'RawData.zip')
        extract_path = os.path.join(download_path)

        list_path = os.path.join(download_path,'lists')
        os.makedirs(list_path,exist_ok=True)

        print(color.BOLD+color.RED+'Downloading Train and Test Lists... '+color.END)   
        list_download(download_path=list_path,show=True)

        print(color.BOLD+color.RED+'Downloading Dataset... '+color.END)   
        synapse_download(user_name=user_name, password=password, path=download_path)

        print(color.BOLD+color.RED+'Extracting Dataset... '+color.END)   
        unzip_dataset(zip_path=zip_path,extract_path=extract_path)

        print(color.BOLD+color.RED+'Making train_npz dirctory... '+color.END) 
        making_train_npz(path = os.path.join(download_path,'train_npz'),
                        samples_path = os.path.join(download_path,'RawData','Training'),
                        lookup_table = [0,3,2,2,8,9,1,7,5,6,0,4,0,0])

        # print(color.BOLD+color.RED+'Making train_npz dirctory... '+color.END) 
        # making_train_npz(path = os.path.join(download_path,'train_npz'),
        #                 samples_path = os.path.join(download_path,'RawData','Training'),
        #                 lookup_table = [0,7,4,3,2,0,5,8,1,0,0,6,0,0])
        
        # print(color.BOLD+color.RED+'Making test_npz dirctory... '+color.END)                      
        # making_test_npz(path = os.path.join(download_path,'test_npz'),
        #                 samples_path = os.path.join(download_path,'RawData','Training'),
        #                 lookup_table = [0,7,4,3,2,0,5,8,1,0,0,6,0,0])

        # print(color.BOLD+color.RED+'Making test_vol_h5 dirctory... '+color.END)                      
        # making_test_vol_h5(path = os.path.join(download_path,'test_vol_h5'),
        #                 samples_path = os.path.join(download_path,'RawData','Training'),
        #                 lookup_table = [0,7,4,3,2,0,5,8,1,0,0,6,0,0])
    elif args.download=='False':
        test_index = test_indexs[fold-1]
        synapse_prepare(combine=combine, test_index=test_index)









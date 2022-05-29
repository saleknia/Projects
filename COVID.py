import os
import sys
import argparse
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
from torchvision import transforms as T
from torchvision.transforms import functional as F
from scipy import ndimage
from typing import Callable
from scipy.ndimage.interpolation import zoom
import h5py
import shutil

class color():
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

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

# transpose

def formal(index,lenght):
    index=str(index)
    index='0'*(lenght-len(index))+index
    return index

def download_data(root,ct_download_path,mask_download_path):
    if not os.path.isdir('/content/UNet/COVID-19'):
        os.system('mkdir -p /content/UNet/COVID-19')
    scaler = MinMaxScaler(feature_range=(0,1))
    pwd = os.getcwd()
    ct_path = os.path.join(root,'ct_scans')
    mask_path = os.path.join(root,'ct_masks')
    os.makedirs(name=ct_path,exist_ok=True)
    os.makedirs(name=mask_path,exist_ok=True)
    test_index = [7, 8, 9]

    os.chdir(path=ct_path)
    for case_num,ct in enumerate(ct_download_path):
        os.system(f'kaggle datasets download andrewmvd/covid19-ct-scans -f ct_scans/{ct}')
        zip_path = ct + '.zip'
        with ZipFile(zip_path, 'r') as myzip:
            myzip.extractall(path=None, members=None, pwd=None) 
        os.remove(path=zip_path)
        sample_path = ct
        sample = nb.load(filename=sample_path).get_fdata()
        # sample = np.clip(a=sample,a_min=-650,a_max=250)
        sample = scaler.fit_transform(sample.reshape(-1,sample.shape[-1])).reshape(sample.shape)
        sample = sample.astype(dtype=np.float32)
        num_slices = sample.shape[2]
        for s in range(num_slices):
            slice_name = 'case_' + formal(index=case_num, lenght=3) + 'slice' + formal(index=s, lenght=3)
            # slice_name = slice_name.split('org_')[0]+slice_name.split('org_')[1]
            np.save(slice_name,arr=sample[:,:,s].transpose()) 
            print_progress(
                iteration=s+1,
                total=num_slices,
                prefix=f'CT Case {case_num+1}',
                suffix=f'Slice {s+1}',
                bar_length=70
            )  
        os.remove(path=ct) 
    ct_files = os.listdir(ct_path)
    ct_files.sort()
    os.system('mkdir -p /content/UNet/COVID-19/train/ct_scans')
    os.system('mkdir -p /content/UNet/COVID-19/test/ct_scans')
    for f in ct_files:
        if int(f.split('case_')[1][0:3]) in test_index:
            shutil.move(src=f'/content/UNet/COVID-19/ct_scans/{f}', dst='/content/UNet/COVID-19/test/ct_scans')
        else:
            shutil.move(src=f'/content/UNet/COVID-19/ct_scans/{f}', dst='/content/UNet/COVID-19/train/ct_scans')
    os.system('rm -r /content/UNet/COVID-19/ct_scans')
    os.chdir(pwd) 
    os.chdir(path=mask_path)
    for mask_num,mask in enumerate(mask_download_path):
        os.system(f'kaggle datasets download andrewmvd/covid19-ct-scans -f lung_and_infection_mask/{mask}')
        zip_path = mask + '.zip'
        with ZipFile(zip_path, 'r') as myzip:
            myzip.extractall(path=None, members=None, pwd=None) 
        os.remove(path=zip_path)            
        sample_path = mask 
        sample = nb.load(filename=sample_path).get_fdata()
        sample = sample.astype(dtype=np.float32)
        sample[sample==2.0]=1.0
        sample[sample==3.0]=2.0
        # sample[sample==1.0]=0.0
        # sample[sample==2.0]=0.0
        # sample[sample==3.0]=1.0
        num_slices = sample.shape[2]
        for s in range(num_slices):
            slice_name = 'case_' + formal(index=mask_num, lenght=3) + 'slice' + formal(index=s, lenght=3)
            np.save(slice_name,arr=sample[:,:,s].transpose())  
            print_progress(
                iteration=s+1,
                total=num_slices,
                prefix=f'Mask Case {mask_num+1}',
                suffix=f'Slice {s+1}',
                bar_length=70
            )  
        os.remove(mask)    
    mask_files = os.listdir(mask_path)
    mask_files.sort()
    os.system('mkdir -p /content/UNet/COVID-19/train/ct_masks')
    os.system('mkdir -p /content/UNet/COVID-19/test/ct_masks')
    for f in mask_files:
        if int(f.split('case_')[1][0:3]) in test_index:
            shutil.move(src=f'/content/UNet/COVID-19/ct_masks/{f}', dst='/content/UNet/COVID-19/test/ct_masks')
        else:
            shutil.move(src=f'/content/UNet/COVID-19/ct_masks/{f}', dst='/content/UNet/COVID-19/train/ct_masks')
    os.system('rm -r /content/UNet/COVID-19/ct_masks')    
    os.chdir(path=pwd)        
    return ct_path,mask_path

def download_metadata(root):
    os.system('mkdir ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    os.system(f'kaggle datasets download -d andrewmvd/covid19-ct-scans -f metadata.csv -p {root}')
    metadata_path = os.path.join(root,'metadata.csv')
    csv_file = pd.read_csv(metadata_path)

    ct_download = np.array(csv_file['ct_scan'])
    ct_download_path = [x.split('ct_scans/')[1] for x in ct_download]

    mask_download = np.array(csv_file['lung_and_infection_mask'])
    mask_download_path = [x.split('lung_and_infection_mask/')[1] for x in mask_download]
    return ct_download_path,mask_download_path

def prepare_covid_data(root='/content/UNet/COVID-19'):
    print(color.BOLD,color.RED)
    print('\rDownloading Dataset...',color.END)
    ct_download_path,mask_download_path = download_metadata(root=root)
    ct_download_path.sort()
    mask_download_path.sort()
    ct_path,mask_path = download_data(root, ct_download_path[0:10], mask_download_path[0:10])
    # ct_path,mask_path = download_data(root, ct_download_path, mask_download_path)
    print(color.BOLD,color.RED)
    print('\rDataset Downloaded.',color.END)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='.', help='root dir for data')
args = parser.parse_args()

if __name__ == "__main__":
    if args.root_path == '.':
        path = '/content/UNet/COVID-19'
    else:
        path = args.root_path

    prepare_covid_data(root=path)


import os
from sklearn.preprocessing import MinMaxScaler
import argparse
from google.colab import files
import numpy as np



def formal(index,lenght):
    index=str(index)
    index='0'*(lenght-len(index))+index
    return index

def install_kaggle_cli():
    os.system('pip install --upgrade --force-reinstall --no-deps kaggle')

def TCIA_kaggle_download(supervised=True):
    pwd = os.getcwd()
    os.chdir(path='/content/UNet_V2')
    if not os.path.isdir('~/.kaggle'):
        os.system('mkdir ~/.kaggle')
    if not os.path.isfile('/content/UNet_V2/kaggle.json'):
        print('Upload Your Kaggle JSON: ')
        uploaded = files.upload()
    os.system('cp kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    os.system('kaggle datasets download -d tahsin/pancreasct-dataset')
    os.system('mkdir /content/UNet_V2/TCIA')   
    os.system('unzip -q /content/UNet_V2/pancreasct-dataset.zip -d /content/UNet_V2/TCIA')
    os.system('rm /content/UNet_V2/pancreasct-dataset.zip')
    if supervised==False:
        os.system('rm -r /content/UNet_V2/TCIA/labels')
    os.chdir(path=pwd)

def TCIA_extract(train_index, valid_index, test_index, supervised=True):

    if supervised:
        os.system('mkdir /content/UNet_V2/TCIA/train')
        os.system('mkdir /content/UNet_V2/TCIA/valid')
        os.system('mkdir /content/UNet_V2/TCIA/test')
    else:
        os.system('mkdir /content/UNet_V2/TCIA/train')

    files = os.listdir('/content/UNet_V2/TCIA/images')
    files.remove('lists')
    files.sort()
    scaler = MinMaxScaler(feature_range=(0,1))
    for count,f in enumerate(files):
        sample = np.load('/content/UNet_V2/TCIA/images/' + f)
        sample = np.clip(a=sample,a_min=-125,a_max=275)
        sample = scaler.fit_transform(sample.reshape(-1,sample.shape[-1])).reshape(sample.shape)

        num_sample_slices = sample.shape[2]

        if supervised:
            label = np.load('/content/UNet_V2/TCIA/labels/' + f)
            for index in range(num_sample_slices):
                if count in train_index:
                    pwd = os.getcwd()
                    os.chdir(path='/content/UNet_V2/TCIA/train')
                    slice_name = 'case'+f'_{formal(count,2)}'+'_slice'+f'{formal(index,3)}'+'.npz'
                    slice_2d = sample[:,:,index]
                    slice_2d = slice_2d.astype(dtype=np.float32)
                    label_2d = label[:,:,index]
                    label_2d = label_2d.astype(dtype=np.float32)
                    np.savez(file=slice_name,image=slice_2d,label=label_2d)
                    os.chdir(path=pwd)
                elif count in valid_index:
                    pwd = os.getcwd()
                    os.chdir(path='/content/UNet_V2/TCIA/valid')
                    slice_name = 'case'+f'_{formal(count,2)}'+'_slice'+f'{formal(index,3)}'+'.npz'
                    slice_2d = sample[:,:,index]
                    slice_2d = slice_2d.astype(dtype=np.float32)
                    label_2d = label[:,:,index]
                    label_2d = label_2d.astype(dtype=np.float32)
                    np.savez(file=slice_name,image=slice_2d,label=label_2d)
                    os.chdir(path=pwd)
                elif count in test_index:
                    pwd = os.getcwd()
                    os.chdir(path='/content/UNet_V2/TCIA/test')
                    slice_name = 'case'+f'_{formal(count,2)}'+'_slice'+f'{formal(index,3)}'+'.npz'
                    slice_2d = sample[:,:,index]
                    slice_2d = slice_2d.astype(dtype=np.float32)
                    label_2d = label[:,:,index]
                    label_2d = label_2d.astype(dtype=np.float32)
                    np.savez(file=slice_name,image=slice_2d,label=label_2d)
                    os.chdir(path=pwd)
        else:
            pwd = os.getcwd()
            os.chdir(path='/content/UNet_V2/TCIA/train')
            for index in range(num_sample_slices):
                slice_name = 'case'+f'_{formal(count,2)}'+'_slice'+f'{formal(index,3)}'+'.npz'
                slice_2d = sample[:,:,index]
                slice_2d = slice_2d.astype(dtype=np.float32)
                np.savez(file=slice_name,image=slice_2d)
            os.chdir(path=pwd)

    if os.path.isdir('/content/UNet_V2/TCIA/images'):
        os.system('rm -r /content/UNet_V2/TCIA/images')

    if os.path.isdir('/content/UNet_V2/TCIA/labels'):
        os.system('rm -r /content/UNet_V2/TCIA/labels')



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='.', help='root dir for data')
parser.add_argument('--supervised', type=str, default='True', help='Providing Labels or not.')

args = parser.parse_args()

if __name__ == "__main__":

    if args.root_path == '.':
        path = os.getcwd()
    else:
        path = args.root_path

    if args.supervised == 'True':
        supervised = True
    else:
        supervised = False

    train_index = np.arange(50)
    valid_index = np.arange(50,65)
    test_index  = np.arange(65,80)

    TCIA_kaggle_download(supervised=supervised)
    TCIA_extract(train_index=train_index, valid_index=valid_index, test_index=test_index, supervised=supervised)




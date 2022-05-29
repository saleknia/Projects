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
    os.chdir(path='/content/UNet')
    if not os.path.isdir('~/.kaggle'):
        os.system('mkdir ~/.kaggle')
    if not os.path.isfile('/content/UNet/kaggle.json'):
        print('Upload Your Kaggle JSON: ')
        uploaded = files.upload()
    os.system('cp kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    os.system('kaggle datasets download -d tahsin/pancreasct-dataset')
    os.system('mkdir /content/UNet/TCIA')   
    os.system('unzip -q /content/UNet/pancreasct-dataset.zip -d /content/UNet/TCIA')
    os.system('rm /content/UNet/pancreasct-dataset.zip')
    if supervised==False:
        os.system('rm -r /content/UNet/TCIA/labels')
    os.chdir(path=pwd)

def TCIA_extract(test_index, supervised=True):

    if supervised:
        os.system('mkdir /content/UNet/TCIA/train')
        os.system('mkdir /content/UNet/TCIA/test')
    else:
        os.system('mkdir /content/UNet/TCIA/train')

    files = os.listdir('/content/UNet/TCIA/images')
    files.remove('lists')
    files.sort()
    scaler = MinMaxScaler(feature_range=(0,1))
    for count,f in enumerate(files):
        sample = np.load('/content/UNet/TCIA/images/' + f)
        sample = np.clip(a=sample,a_min=-125,a_max=275)
        sample = scaler.fit_transform(sample.reshape(-1,sample.shape[-1])).reshape(sample.shape)

        num_sample_slices = sample.shape[2]

        if supervised:
            label = np.load('/content/UNet/TCIA/labels/' + f)
            for index in range(num_sample_slices):
                if count in test_index:
                    pwd = os.getcwd()
                    os.chdir(path='/content/UNet/TCIA/test')
                    slice_name = 'case'+f'_{formal(count,2)}'+'_slice'+f'{formal(index,3)}'+'.npz'
                    slice_2d = sample[:,:,index]
                    slice_2d = slice_2d.astype(dtype=np.float32)
                    label_2d = label[:,:,index]
                    label_2d = label_2d.astype(dtype=np.float32)
                    np.savez(file=slice_name,image=slice_2d,label=label_2d)
                    os.chdir(path=pwd)
                else:
                    pwd = os.getcwd()
                    os.chdir(path='/content/UNet/TCIA/train')
                    slice_name = 'case'+f'_{formal(count,2)}'+'_slice'+f'{formal(index,3)}'+'.npz'
                    slice_2d = sample[:,:,index]
                    slice_2d = slice_2d.astype(dtype=np.float32)
                    label_2d = label[:,:,index]
                    label_2d = label_2d.astype(dtype=np.float32)
                    np.savez(file=slice_name,image=slice_2d,label=label_2d)
                    os.chdir(path=pwd)
        else:
            pwd = os.getcwd()
            os.chdir(path='/content/UNet/TCIA/train')
            for index in range(num_sample_slices):
                slice_name = 'case'+f'_{formal(count,2)}'+'_slice'+f'{formal(index,3)}'+'.npz'
                slice_2d = sample[:,:,index]
                slice_2d = slice_2d.astype(dtype=np.float32)
                np.savez(file=slice_name,image=slice_2d)
            os.chdir(path=pwd)

    if os.path.isdir('/content/UNet/TCIA/images'):
        os.system('rm -r /content/UNet/TCIA/images')

    if os.path.isdir('/content/UNet/TCIA/labels'):
        os.system('rm -r /content/UNet/TCIA/labels')



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
    test_index = np.arange(10)

    TCIA_kaggle_download(supervised=supervised)
    TCIA_extract(test_index=test_index, supervised=supervised)




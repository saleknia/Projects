import os
import numpy as np
os.system('pip install opencv-python==4.5.5.64')
import cv2


if not os.path.isdir('/content/UNet_V2/SSL'):
    os.mkdir('/content/UNet_V2/SSL')
os.system('cd /content/UNet_V2/SSL')
os.system('cp /content/UNet_V2/CT-1K/train/* /content/UNet_V2/SSL')
os.system('cp /content/UNet_V2/CT-1K/test/* /content/UNet_V2/SSL')
os.system('cp /content/UNet_V2/CT-1K/valid/* /content/UNet_V2/SSL')

kernel_size = [7, 3, 3, 1, 3, 3, 3]
list_dir = os.listdir('content/UNet_V2/SSL')

for f in list_dir:
    data_path = '/content/UNet_V2/SSL/' + f
    data = np.load(data_path)
    image, mask = data['image'], data['label']   
    mask[mask==6] = 0
    mask[mask==9] = 0
    mask[mask==10] = 0
    mask[mask==11] = 0
    mask[mask==12] = 0
    mask[mask==7] = 6
    mask[mask==8] = 7

    unique = np.unique(mask)[1:].astype('uint8')
    mask_temp = np.zeros(mask.shape)
    for i in unique:
        if kernel_size[i-1]!=0:
            kernel = np.ones((kernel_size[i-1], kernel_size[i-1]), np.uint8)
            opening_gt_temp = cv2.erode((mask==i).astype('uint8'), kernel, iterations=1)
            mask_temp = mask_temp + (opening_gt_temp * i)
        else:
            mask_temp = mask_temp + ((mask==i).astype('uint8') * i)
    opening_gt = mask_temp
    np.savez(file=f,image=image,label=mask)

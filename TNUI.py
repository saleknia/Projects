import numpy as np
import cv2
import torch
import torch.utils.data as data
import os
import random
from scipy import ndimage

def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


###################### get image path  ######################
def get_image_paths(dataroot):

    paths = []
    if dataroot is not None:
        paths_img = os.listdir(dataroot)
        for _ in sorted(paths_img):
            path = os.path.join(dataroot, _)
            paths.append(path)
    return paths

###################### read images ######################

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.

    if img.ndim == 2:
        # img = np.expand_dims(img, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1)
    return img

###################### read labels ######################

def read_nodule_label(label_path):

    img_label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    img_label = img_label / 255
    return img_label

def read_cell_label(label_path):
    img_label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    img_label = np.where(img_label == 128, 1, img_label)
    img_label = np.where(img_label == 255, 2, img_label)
    return img_label

#----------------------------------------------------------------------------------------------------#
#       数据增广----Z
#----------------------------------------------------------------------------------------------------#

img_w = 512
img_h = 512

# def gamma_transform(img, gamma):
#     gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
#     gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
#     return cv2.LUT(img, gamma_table)
#
#
# def random_gamma_transform(img, gamma_vari):
#     log_gamma_vari = np.log(gamma_vari)
#     alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
#     gamma = np.exp(alpha)
#     return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb

def img_blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img

def channel_change(img):
    channels = [0, 1, 2]
    random.shuffle(channels)
    img = img[:, :, channels]
    return img

def data_augment(img_list, rot=True, flip=True, blur=True, noise=True, channel_ch=True):
    xb, yb = img_list
    if rot:
        if np.random.random() < 0.5:
            xb, yb = rotate(xb, yb, 90)
        if np.random.random() < 0.5:
            xb, yb = rotate(xb, yb, 180)
        if np.random.random() < 0.5:
            xb, yb = rotate(xb, yb, 270)
    if flip:
        if np.random.random() < 0.5:
            xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
            yb = cv2.flip(yb, 1)
        if np.random.random() < 0.5:
            xb = cv2.flip(xb, 0)  # flipcode > 0：沿x
            yb = cv2.flip(yb, 0)
        if np.random.random() < 0.5:
            xb = cv2.flip(xb, -1)  # flipcode > 0：沿x,y
            yb = cv2.flip(yb, -1)

    # if blur:
    #     if np.random.random() < 0.25:
    #         xb = img_blur(xb)
    # if noise:
    #     if np.random.random() < 0.2:
    #         xb = add_noise(xb)

    # if channel_ch and xb.shape[2] == 3:
    #     if np.random.random() < 0.2:
    #         xb = channel_change(xb)

    # if np.random.random() < 0.25:
    #     xb = random_gamma_transform(xb, 1.0)

    return [xb, yb]

#-----------------------------------------------------------------------------------------------------#
# image processing
# process on numpy image
#-----------------------------------------------------------------------------------------------------#

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-30, 30)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def apply_augmentation(image, label):
    if random.random() > 0.0:
        image, label = random_rot_flip(image, label)
    if random.random() > 0.0:
        image, label = random_rotate(image, label)

    return image, label

def augment(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot   and random.random() < 0.5
    rot90 = rot   and random.random() < 0.5

    # print(img_list[0].shape)
    # print(img_list[1].shape)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

class CreateDataset(data.Dataset):
    def __init__(self, img_paths, label_paths, resize, phase, aug=False):
        super(CreateDataset, self).__init__()
        self.phase = phase
        self.resize = resize
        self.aug = aug

        self.paths_imgs = get_image_paths(img_paths)
        assert self.paths_imgs, 'Error: imgs paths are empty.'

        self.paths_label = get_image_paths(label_paths)
        assert self.paths_label, 'Error: paths_label are empty.'

    def __getitem__(self, index):

        img_path = self.paths_imgs[index]

        # print(img_path)

        img = read_img(img_path)  # h w c

        lable_path = self.paths_label[index]

        label = read_nodule_label(lable_path)  # h w

        img = cv2.resize(img, (self.resize, self.resize))
        label = cv2.resize(label, (self.resize, self.resize))

        if self.phase == 'train':
            if self.aug:
                img, label = augment([img, np.expand_dims(label, axis=2)], hflip=True, rot=True)
                # img, label= apply_augmentation(img, np.expand_dims(label, axis=2))

                label = label.squeeze(2) #HW
                img = img[:, :, [2, 1, 0]] # bgr -> rgb

        elif self.phase == 'val':
            if img.shape[2] == 3:
                img = img[:, :, [2, 1, 0]]

            # if label.shape[2] == 3:
            #     label = label[:,:, 1]
        # print(label.shape)
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

        label = torch.from_numpy(np.ascontiguousarray(label)).long()


        return img, label

    def __len__(self):
        return len(self.paths_imgs)
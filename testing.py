import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from dataset import Synapse_dataset,ValGenerator
from torch.utils.data import DataLoader
import tqdm
import math
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        # return 1, 0
        return 0,0
    elif pred.sum() == 0 and gt.sum()==0:
        return math.nan,math.nan
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cpu()
            net.eval()
            # print(input.is_cuda)
            # print(next(net.parameters()).is_cuda)
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
        metrics = calculate_metric_percase(prediction == i, label == i)
        if metrics==(math.nan,math.nan):
            pass
        else:
            metric_list.append(metrics)

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     if len(image.shape) == 3:
#         prediction = np.zeros(shape=(label.shape[0],patch_size[0],patch_size[0]))
#         masks = np.zeros(shape=(label.shape[0],patch_size[0],patch_size[0]))
#         for ind in range(image.shape[0]):
#             sample = image[ind]
#             mask = label[ind]
#             x, y = sample.shape[0], sample.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 sample = zoom(sample, (patch_size[0] / x, patch_size[1] / y), order=0)  # why not 3?
#                 mask = zoom(mask, (patch_size[0] / x, patch_size[1] / y), order=0)

#             # sample = np.flip(m=sample,axis=0)
#             # mask = np.flip(m=mask,axis=0)
#             # sample = np.rot90(m=sample,k=1)
#             # mask = np.rot90(m=mask,k=1)

#             input = torch.from_numpy(sample.copy()).unsqueeze(0).unsqueeze(0).float().cuda()

#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 pred = out
#                 masks[ind] = mask
#                 prediction[ind] = pred
#         label = masks
#     else:
#         input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))

#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     return metric_list

# DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

def inference(model,logger,test_save_path=None,num_classes=9,img_size=256):
    # model = model.cpu()
    db_test = Synapse_dataset(split='val_test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logger.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=num_classes, patch_size=[img_size,img_size],test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        logger.info('Index %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, num_classes):
        logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logger.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


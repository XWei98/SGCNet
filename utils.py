import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import nibabel as nib

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)

        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes




class DiceLosssamtopo(nn.Module):
    def __init__(self, n_classes):
        super(DiceLosssamtopo, self).__init__()
        self.n_classes = n_classes


    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target,weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # 将所有类别的目标标签合并为一个类别
        one_hot_target = self._one_hot_encoder(target)
        one_hot_target[one_hot_target <= 0] = 0
        one_hot_target[one_hot_target > 0] = 1
        re_hot_target = one_hot_target.clone()
        re_hot_target[re_hot_target <= 0] = 1
        re_hot_target[re_hot_target > 0] = 0

        target_combined = (target > 0).float().unsqueeze(1).repeat(1, 9, 1, 1)  # 将所有非背景类别合并为一个类别
        relabel = re_hot_target*target_combined
        relabel[relabel < 1] = 0
        relabel[relabel >= 1] = 1
        sinputs = inputs*relabel
        if weight is None:
            weight = [1] * self.n_classes
        assert sinputs.size() == target_combined.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(sinputs[:, i]+target_combined[:, i], target_combined[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class DiceLosssam(nn.Module):
    def __init__(self, n_classes):
        super(DiceLosssam, self).__init__()
        self.n_classes = n_classes


    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target,weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        # 将所有类别的目标标签合并为一个类别
        target_combined = (target > 0).float()  # 将所有非背景类别合并为一个类别

        # 将所有类别的预测结果合并为一个通道
        inputs_combined = torch.sum(inputs, dim=1)  # 在类别维度上进行sum，得到一个通道

        assert inputs_combined.size() == target_combined.size(), 'predict {} & target {} shape do not match'.format(
            inputs_combined.size(), target_combined.size())

        # 计算单类别的Dice损失
        dice_loss = self._dice_loss(inputs_combined, target_combined)
        return dice_loss
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


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path="/data1/Code/zhaoxiaowei/TransUNet-main/vision3d", case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        save_img = torch.zeros(image.shape)
        # save_img = save_img.expand(-1, 9, -1, -1)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)[0].unsqueeze(0)
                #print(outputs.shape)
                # print(outputs.shape)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                temp = F.interpolate(torch.unsqueeze(torch.unsqueeze(out,dim=0),dim=0).float(), size=(512, 512), mode='nearest')
                save_img[ind] = temp.squeeze()
                # transform = transforms.ToPILImage()
                # for i in range(outputs.size(0)):#batch
                #     for f in range(outputs.size(1)):#类别
                #         img = torch.zeros(out.shape)
                #         img[out==f]=1
                #         img = torch.unsqueeze(img,dim=0)
                #         img = transforms.ToPILImage()((img * 255).byte())
                #         img.save(f"./vision/{f}/image_{case[i]}_slice{ind}.png")
                    
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
               # print(pred.shape)#1,9,448,448
                prediction[ind] = pred
        save_img = save_img.cpu().numpy()
        toNii_gz(image,save_img,case)    
        # image = image.cpu().numpy()  # 如果在 GPU 上，先移动到 CPU
        # save_img = save_img.cpu().numpy()
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

        # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        # img_itk.SetSpacing((1, 1, z_spacing))
        # prd_itk.SetSpacing((1, 1, z_spacing))
        # lab_itk.SetSpacing((1, 1, z_spacing))
        # sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        # sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        # sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def toNii_gz(original_img,segmentation_map,name):

    original_img = original_img*255
    original_img_nifti = nib.Nifti1Image(original_img, affine=np.eye(4))
    seg_img_nifti = nib.Nifti1Image(segmentation_map, affine=np.eye(4))
    
    nib.save(original_img_nifti, f'./vision3d/{name}image.nii.gz')
    nib.save(seg_img_nifti, f'./vision3d/{name}seg.nii.gz')

    print("name已保存nii.gz 文件")

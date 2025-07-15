
import sys
import random
sys.path.append(".")
from guided_diffusion.utils import staple

import numpy
import numpy as np
import jittor as jt
import math
from PIL import Image
import matplotlib.pyplot as plt
from guided_diffusion.utils import staple
import argparse

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz


def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((0, 1))
    union = (outputs | labels).sum((0, 1))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(jt.Function):
    """Dice系数计算（Jittor适配版）"""
    def forward(self, input, target):
        # 移除 save_for_backward，直接计算并保存所需变量
        eps = 0.0001
        self.inter = jt.matmul(input.view(-1), target.view(-1))
        self.union = jt.sum(input) + jt.sum(target) + eps
        return (2 * self.inter.float() + eps) / self.union.float()

    def backward(self, grad_output):
        # 直接使用 forward 中保存的变量，无需从 saved_variables 获取
        input = self.input  # 从 forward 中显式保存的变量
        target = self.target
        grad_input = grad_output * 2 * (target * self.union - self.inter) / (self.union * self.union)
        return grad_input, None


def dice_coeff(input, target):
    """Dice coeff for batches"""
    # s = torch.FloatTensor(1).to(device = input.device).zero_()
    s = jt.zeros(1, dtype="float32").stop_grad()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_seg(pred,true_mask_p,threshold = (0.1, 0.3, 0.5, 0.7, 0.9)):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()

            # 修正：使用 squeeze() 自动挤压所有大小为1的维度
            disc_pred = vpred[:, 0, :, :].squeeze().cpu().numpy().astype('int32')
            disc_mask = gt_vmask_p[:, 0, :, :].squeeze().cpu().numpy().astype('int32')

            eiou += iou(disc_pred, disc_mask)
            edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
            
        return eiou / len(threshold), edice / len(threshold)

def main():
    argParser = argparse.ArgumentParser()
    image_size = 64
    # argParser.add_argument("./results")
    # argParser.add_argument("../data\ISIC\Test\ISBI2016_ISIC_Part1_Test_GroundTruth")
    args = argParser.parse_args()
    mix_res = (0,0)
    num = 0
    pred_path = "./results-torch"
    gt_path = "../data\ISIC\Test\ISBI2016_ISIC_Part1_Test_GroundTruth"
    for root, dirs, files in os.walk(pred_path, topdown=False):
        for name in files:
            if 'ens' in name:
                num += 1
                ind = name.split('_')[0]
                pred = Image.open(os.path.join(root, name)).convert('L')
                gt_name = "ISIC_" + ind + "_Segmentation.png"
                gt = Image.open(os.path.join(gt_path, gt_name)).convert('L')
                # pred = torchvision.transforms.PILToTensor()(pred)
                # pred = torch.unsqueeze(pred,0).float()
                pred = jt.array(np.array(pred))  # 替代torchvision.transforms.PILToTensor()
                pred = pred.unsqueeze(0).unsqueeze(0).float()  # 增加批次维度并转换为float类型
                pred = pred / pred.max()
                # if args.debug:
                #     print('pred max is', pred.max())
                #     vutils.save_image(pred, fp = os.path.join('./results/' + str(ind)+'pred.jpg'), nrow = 1, padding = 10)
                # gt = torchvision.transforms.PILToTensor()(gt)
                # gt = torchvision.transforms.Resize((256,256))(gt)
                # gt = torch.unsqueeze(gt,0).float() / 255.0

                # 1. 将PIL图像转换为Jittor张量（保持0-255）
                gt = jt.array(np.array(gt))  # 转换为Jittor张量，形状为(H, W, C)
                gt = gt.unsqueeze(0).unsqueeze(0).float()


                # 2. 调整图像大小为256x256（使用jt.nn.interpolate）
                gt = jt.nn.interpolate(gt, size=(64, 64), mode='bilinear', align_corners=False)

                # 3. 归一化到[0,1]并移除批次维度（如果需要）
                gt = gt.float() / 255.0

                # gt = jt.squeeze(gt, 0)  # 如果需要移除批次维度，可以取消注释这一行
                # if args.debug:
                #     vutils.save_image(gt, fp = os.path.join('./results/' + str(ind)+'gt.jpg'), nrow = 1, padding = 10)
                # print(pred.shape)
                # print(gt.shape)
                # print(pred)
                # print(gt)
                temp = eval_seg(pred, gt)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
    iou, dice = tuple([a/num for a in mix_res])
    print('iou is',iou)
    print('dice is', dice)

if __name__ == "__main__":
    main()

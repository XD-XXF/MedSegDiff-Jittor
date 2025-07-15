import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import jittor as jt  # 替换torch为jittor
from jittor.dataset import Dataset  # Jittor的Dataset基类
from PIL import Image
import jittor.transform as transforms  # Jittor的transform模块
import pandas as pd
from skimage.transform import rotate


class ISICDataset(Dataset):  # 继承Jittor的Dataset
    def __init__(self, args, data_path, transform=None, mode='Training', plane=False):
        super().__init__()

        # 读取CSV文件（注意：Jittor对编码的处理与PyTorch一致）
        df = pd.read_csv(
            os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'),
            encoding='gbk'
        )
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform

        # 关键：设置数据集长度（Jittor需要显式声明）
        self.set_attrs(total_len=len(self.name_list))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """获取图像和掩码，返回Jittor张量"""
        name = self.name_list[index]
        # 注意：路径中的"Train"可能需要根据mode动态调整（如验证集用"Test"）
        if self.mode == "Train":
            img_path = os.path.join(self.data_path, "ISIC", "Train", name)
            mask_name = self.label_list[index]
            msk_path = os.path.join(self.data_path, "ISIC", "Train", mask_name)
        else:
            img_path = os.path.join(self.data_path, "ISIC", "Test", name)
            mask_name = self.label_list[index]
            msk_path = os.path.join(self.data_path, "ISIC", "Test", mask_name)

        # 读取图像和掩码
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')  # 掩码为灰度图

        # 应用数据增强（Jittor的transform与PyTorch类似，但随机状态管理不同）
        import random

        if self.transform:
            # 保存当前Python随机状态
            state = random.getstate()

            # 应用图像增强（确保transform中只使用Python的random模块）
            random.setstate(state)
            img = self.transform(img)

            # 应用相同的随机状态到掩码
            random.setstate(state)
            mask = self.transform(mask)


        return (img, mask, name)
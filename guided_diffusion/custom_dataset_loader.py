import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import jittor as jt  # 替换torch为jittor
from jittor.dataset import Dataset  # Jittor数据集基类
from PIL import Image
import jittor.transform as transforms  # Jittor数据增强工具
import pandas as pd
from skimage.transform import rotate
from glob import glob
from sklearn.model_selection import train_test_split
import nibabel


### 2D数据集（CustomDataset）
class CustomDataset(Dataset):  # 继承Jittor的Dataset
    def __init__(self, args, data_path, transform=None, mode="Training", plane=False):
        super().__init__()  # 初始化父类
        print("loading data from the directory :", data_path)
        path = data_path
        # 加载图像和掩码路径
        images = sorted(glob(os.path.join(path, "images/*.png")))
        masks = sorted(glob(os.path.join(path, "masks/*.png")))

        self.name_list = images
        self.label_list = masks
        self.data_path = path
        self.mode = mode
        self.transform = transform

        # 显式设置数据集长度（Jittor必需）
        self.set_attrs(total_len=len(self.name_list))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # 获取图像和掩码路径
        name = self.name_list[index]
        img_path = name  # 已为完整路径，无需额外拼接
        mask_name = self.label_list[index]
        msk_path = mask_name

        # 读取图像（RGB）和掩码（灰度图）
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # 应用数据增强（保证图像和掩码增强一致）
        if self.transform:
            # Jittor用jt.random管理随机状态
            state = jt.random.get_state()
            img = self.transform(img)
            jt.random.set_state(state)  # 同步掩码的随机状态
            mask = self.transform(mask)

        return (img, mask, name)


### 3D数据集（CustomDataset3D）
class CustomDataset3D(Dataset):  # 继承Jittor的Dataset
    def __init__(self, data_path, transform):
        super().__init__()  # 初始化父类
        print("loading data from the directory :", data_path)
        path = data_path
        # 加载3D图像和掩码（nii.gz格式）
        images = sorted(glob(os.path.join(path, "images/*.nii.gz")))
        masks = sorted(glob(os.path.join(path, "masks/*.nii.gz")))

        # 校验图像和掩码数量一致
        assert len(images) == len(masks), "Number of images and masks must be the same"
        self.valid_cases = list(zip(images, masks))  # 存储（图像路径，掩码路径）对

        # 解析所有切片信息（case索引 + 切片索引）
        self.all_slices = []
        for case_idx, (img_path, seg_path) in enumerate(self.valid_cases):
            # 加载nii文件获取形状
            seg_vol = nibabel.load(seg_path)
            img = nibabel.load(img_path)
            assert img.shape == seg_vol.shape, \
                f"Image and segmentation shape mismatch: {img.shape} vs {seg_vol.shape}, Files: {img_path}, {seg_path}"
            num_slices = img.shape[-1]  # 切片数量（假设最后一维度为切片）
            self.all_slices.extend([(case_idx, slice_idx) for slice_idx in range(num_slices)])

        self.data_path = path
        self.transform = transform
        # 显式设置数据集长度
        self.set_attrs(total_len=len(self.all_slices))

    def __len__(self):
        return len(self.all_slices)

    def __getitem__(self, index):
        # 获取当前切片的case索引和切片索引
        case_idx, slice_idx = self.all_slices[index]
        img_path, seg_path = self.valid_cases[case_idx]

        # 加载3D体积并提取对应切片
        nib_img = nibabel.load(img_path)
        nib_seg = nibabel.load(seg_path)

        # 提取切片并转换为Jittor张量（替换torch.tensor为jt.array）
        # 形状：(1, 1, H, W) -> （通道数，批次维度？不，Jittor无需额外批次维，此处保持与原代码维度一致）
        image = jt.array(nib_img.get_fdata(), dtype=jt.float32)[:, :, slice_idx].unsqueeze(0).unsqueeze(0)
        label = jt.array(nib_seg.get_fdata(), dtype=jt.float32)[:, :, slice_idx].unsqueeze(0).unsqueeze(0)
        # 将掩码二值化（合并所有肿瘤类别）
        label = jt.where(label > 0, 1, 0).float()

        # 应用数据增强（同步图像和掩码的随机状态）
        if self.transform:
            state = jt.random.get_state()
            image = self.transform(image)
            jt.random.set_state(state)
            label = self.transform(label)

        # 返回图像、掩码和虚拟路径（用于保存结果）
        virtual_path = f"{img_path.split('.nii')[0]}_slice{slice_idx}.nii"
        return (image, label, virtual_path)
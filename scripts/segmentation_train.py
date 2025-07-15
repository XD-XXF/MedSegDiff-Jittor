import argparse
import os
from pathlib import Path
import random
import jittor
import sys

sys.path.append(".")
import numpy as np
import time

# 替换PyTorch为Jittor
import jittor as jt
from jittor import dataset
from jittor import nn
import jittor.transform as transforms

from guided_diffusion.dataaaa import DataLoader
from guided_diffusion import logger
from guided_diffusion.resample import create_named_schedule_sampler
# from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset, CustomDataset3D
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from guided_diffusion.train_util import TrainLoop



def main():
    args = create_argparser().parse_args()
    args.image_size = 256
    # Jittor设备设置（默认使用GPU，无需显式指定device，自动适配）
    jt.flags.use_cuda = 1

    # 禁用多GPU，强制单卡训练
    args.multi_gpu = None
    args.gpu_dev = "0"

    # 配置日志目录
    logger.configure(dir=args.out_dir)
    logger.log("creating data loader...")

    # 数据加载（适配Jittor的Dataset和transforms）
    if args.data_name == 'ISIC':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()  # Jittor的ToTensor会将数据转为jittor.Var
        ]
        transform_train = transforms.Compose(tran_list)
        ds = ISICDataset(args, args.data_dir, transform_train, mode="Train")
        args.in_ch = 4

    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size, args.image_size))]
        transform_train = transforms.Compose(tran_list)
        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5

    elif any(Path(args.data_dir).glob("*\*.nii.gz")):
        tran_list = [transforms.Resize((args.image_size, args.image_size))]
        transform_train = transforms.Compose(tran_list)
        print(f"Loading 3D custom dataset from {args.data_dir}")
        ds = CustomDataset3D(args, args.data_dir, transform_train)
        args.in_ch = 4

    else:
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ]
        transform_train = transforms.Compose(tran_list)
        print(f"Loading 2D custom dataset from {args.data_dir}")
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4

    # Jittor数据加载器（替换PyTorch的DataLoader）
    datal = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4  # Jittor的num_workers设置
    )
    data = iter(datal)

    logger.log("creating model and diffusion...")

    # 创建模型和扩散过程（需确保内部使用Jittor的nn模块）
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )


    # 调度采样器（适配Jittor）
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=args.diffusion_steps)

    logger.log("training...")
    # 训练循环（需确保TrainLoop内部使用Jittor的API）
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name='ISIC',
        data_dir="../data",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=1000,
        batch_size=1,
        microbatch=-1,  # -1表示禁用微批次
        ema_rate="0.9999",  # EMA衰减率（逗号分隔的列表）
        log_interval=1,
        save_interval=1,
        resume_checkpoint=None,  # 恢复训练的检查点路径
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev="0",
        multi_gpu=None,  # 单卡训练
        out_dir='./results/',
        image_size=256,
        num_channels=2,
        class_cond=False,
        num_res_blocks=1,
        num_heads=1,
        learn_sigma=True,
        use_scale_shift_norm=False,
        attention_resolutions=1,
        diffusion_steps=1,
        noise_schedule='linear',
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
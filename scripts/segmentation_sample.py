

import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import jittor as jt
from PIL import Image
from guided_diffusion.dataaaa import DataLoader
import jittor.transform as transforms
from guided_diffusion import dist_util, logger
# from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img




def main():
    jt.flags.use_cuda = 1
    args = create_argparser().parse_args()
    args.image_size = 64


    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)


    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4

    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir,transform_test)
        args.in_ch = 5
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)

        ds = CustomDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4

    datal = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    # data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = jt.load(args.model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    # model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # for _ in range(len(data)):
    #     b, m, path = next(data)  #should return an image from the dataloader "data"
    #     c = jt.randn_like(b[:, :1, ...])
    #     img = jt.concat((b, c), dim=1)     #add a noise channel$
    #     if args.data_name == 'ISIC':
    #         slice_ID=path[0].split("_")[-1].split('.')[0]
    #     elif args.data_name == 'BRATS':
    #         # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
    #         slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
    #
    #     logger.log("sampling...")



    # 修改为：
    for idx, (b, m, path) in enumerate(datal):  # 直接迭代 DataLoader

        c = jt.randn_like(b[:, :1, ...])
        img = jt.concat((b, c), dim=1)  # 添加噪声通道

        # 后续代码保持不变...
        if args.data_name == 'ISIC':
            slice_ID = path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            slice_ID = path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]

        logger.log("sampling...")
        # ... 其余代码 ...


        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            print(slice_ID)
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )



            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )




            co = jt.array(cal_out)
            if args.version == 'new':
                enslist.append(sample[:,-1,:,:])
            else:
                enslist.append(co)

            if args.debug:
                # print('sample size is',sample.size())
                # print('org size is',org.size())
                # print('cal size is',cal.size())
                if args.data_name == 'ISIC':
                    # s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = jt.array(org)[:,:-1,:,:]
                    c = jt.array(cal).repeat(1, 3, 1, 1)
                    # co = co.repeat(1, 3, 1, 1)

                    s = sample[:,-1,:,:]
                    b,h,w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                    tup = (ss,o,c)
                elif args.data_name == 'BRATS':
                    s = jt.array(sample)[:,-1,:,:].unsqueeze(1)
                    m = jt.array(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = jt.array(org)[:,0,:,:].unsqueeze(1)
                    o2 = jt.array(org)[:,1,:,:].unsqueeze(1)
                    o3 = jt.array(org)[:,2,:,:].unsqueeze(1)
                    o4 = jt.array(org)[:,3,:,:].unsqueeze(1)
                    c = jt.array(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                compose = jt.concat(tup, 0)
                jt.save_image(compose, os.path.join(args.out_dir, str(slice_ID) + '_output' + str(i) + ".jpg"),
                                  nrow=1, padding=10)


            ensres = staple(jt.stack(enslist, dim=0)).squeeze(0)

            ensres_np = ensres.numpy()
            print(ensres_np.shape)
            # 调整维度顺序，从 (C, H, W) 到 (H, W, C)
            ensres_np = np.transpose(ensres_np, (1, 2, 0))
            # 归一化到 [0, 255] 范围
            ensres_np = (ensres_np - ensres_np.min()) / (ensres_np.max() - ensres_np.min()) * 255
            ensres_np = ensres_np.astype(np.uint8)
            # 如果是单通道图像，去掉多余的维度
            if ensres_np.shape[-1] == 1:
                ensres_np = ensres_np.squeeze(-1)
            # 创建 PIL 图像对象
            ensres_img = Image.fromarray(ensres_np)
            # 保存图像
            ensres_img.save(os.path.join(args.out_dir, str(slice_ID) + '_output_ens' + ".jpg"))

def create_argparser():
    defaults = dict(
        data_name='ISIC',
        data_dir="../data",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./results/emasavedmodel_0.9999_140000.pkl",  # path to pretrain model
        num_ensemble=1,  # number of samples in the ensemble
        gpu_dev="0",
        multi_gpu=None,  # "0,1,2"
        debug=False,
        resume_checkpoint=None,  # "/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        out_dir='./results/',
        image_size=256,
        num_channels=128,
        class_cond=False,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=True,
        use_scale_shift_norm=False,
        attention_resolutions=16,
        diffusion_steps=100,
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

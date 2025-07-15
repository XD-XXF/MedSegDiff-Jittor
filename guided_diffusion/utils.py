
import numpy as np
import jittor
import jittor as jt
import jittor.nn as nn


softmax_helper = lambda x: jt.nn.softmax(x, 1)
sigmoid_helper = lambda x: jt.sigmoid(x)


class TransposeConv2d(jt.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = jt.nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, transpose=True
        )

    def execute(self, x):
        return self.conv(x)

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, TransposeConv2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def maybe_to_jittor(d):
    if isinstance(d, list):
        d = [maybe_to_jittor(i) if not isinstance(i, jt.Var) else i for i in d]
    elif not isinstance(d, jt.Var):
        d = jt.array(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def staple(a):
    # a: n,c,h,w detach tensor
    mvres = mv(a)
    gap = 0.4
    if gap > 0.02:
        for i, s in enumerate(a):

            r = s * mvres
            res = r if i == 0 else jt.concat((res,r),0)
        nres = mv(res)
        gap = jt.mean(jt.abs(mvres - nres))
        mvres = nres
        a = res
    return mvres

def allone(disc,cup):
    disc = np.array(disc) / 255
    cup = np.array(cup) / 255
    res = np.clip(disc * 0.5 + cup,0,1) * 255
    res = 255 - res
    res = Image.fromarray(np.uint8(res))
    return res

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def mv(a):
    b = a.shape[0]  # 获取第一维的大小
    return jt.sum(a, dim=0, keepdims=True) / b  # 沿第0维求和并保持维度

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def export(tar, img_path=None):
    # 获取通道数
    c = tar.shape[1]

    if c == 3:
        # 直接保存RGB图像
        vutils.save_image(tar, img_path)
    else:
        # 提取最后一个通道并复制为三通道
        s = tar[:, -1, :, :].unsqueeze(1)  # 提取最后一个通道
        s = jt.concat([s, s, s], dim=1)  # 复制为RGB三通道
        vutils.save_image(s, img_path)


def norm(t):
    # 计算均值、标准差和方差
    m = jt.mean(t)
    s = jt.std(t)
    # v = jt.var(t)  # 计算了但未使用，可以省略

    # 标准化处理
    return (t - m) / s

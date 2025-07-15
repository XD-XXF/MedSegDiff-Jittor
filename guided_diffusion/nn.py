"""
Various utilities for neural networks.
"""

import math

import jittor as jt


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(jt.nn.Module):
    def forward(self, x):
        return x * jt.sigmoid(x)


# class GroupNorm32(jt.nn.GroupNorm):
#     def forward(self, x):
#         return super().forward(x.float()).type(x.dtype)
#
#     def execute(self, x):
#         return super().execute(x.float()).astype(x.dtype)


class GroupNorm32(jt.nn.GroupNorm):
    def execute(self, x):
        # 保存输入形状，确保输出形状一致
        input_shape = x.shape
        # 调用父类归一化（先转float32计算，再转回原类型）
        out = super().execute(x.float()).astype(x.dtype)
        # 强制恢复原始形状（防止维度挤压）
        return out.reshape(input_shape)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return jt.nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return jt.nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return jt.nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def layer_norm(shape, *args, **kwargs):

    return jt.nn.LayerNorm(shape, *args, **kwargs)

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return jt.nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return jt.nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return jt.nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return jt.nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    # for targ, src in zip(target_params, source_params):
    #     targ.detach().mul_(rate).add_(src, alpha=1 - rate)
    with jt.no_grad():  # 不需要计算梯度
        for targ, src in zip(target_params, source_params):
            # 直接在原始targ Tensor上进行原地操作
            # 注意: Jittor中不需要detach()，因为所有操作默认不跟踪梯度
            targ.update(targ * rate + src * (1 - rate))

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dims=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建正弦时间步嵌入 (Jittor版本)
    """
    half = dim // 2

    # 计算频率
    freqs = jt.exp(
        -math.log(max_period) * jt.arange(start=0, end=half, dtype=jt.float32) / half
    )

    # 计算参数
    args = timesteps[:, None].float() * freqs[None]

    # 拼接余弦和正弦嵌入
    embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)

    # 处理维度为奇数的情况
    if dim % 2:
        embedding = jt.concat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)

    return embedding




import jittor as jt


def checkpoint(func, inputs, params, flag):
    """
    Jittor版本的梯度检查点函数，用于在节省内存的同时保持梯度计算能力。

    :param func: 待执行的函数
    :param inputs: 传递给func的输入参数序列
    :param params: func依赖的参数序列（非输入，但需要计算梯度）
    :param flag: 是否启用梯度检查点（True启用，False直接执行）
    """
    if flag:
        # 启用检查点模式：正向不缓存中间结果，反向重新计算
        return CheckpointFunction.apply(func, len(inputs), *inputs, *params)
    else:
        # 不启用检查点，直接执行函数
        return func(*inputs)


class CheckpointFunction:
    @staticmethod
    def apply(func, input_len, *args):
        """
        自定义检查点实现，替代PyTorch的autograd.Function

        :param func: 待执行的函数
        :param input_len: 输入参数的数量（用于从args中拆分inputs和params）
        :param args: 合并的输入参数和依赖参数（前input_len个是inputs，剩余是params）
        """
        # 拆分输入和参数（与原代码逻辑一致）
        inputs = args[:input_len]
        params = args[input_len:]

        # 正向传播：不记录中间梯度，节省内存
        with jt.no_grad():
            output = func(*inputs)

        # 定义反向传播回调：重新计算正向过程以获取梯度
        def backward(output_grads):
            # 重新创建带梯度的输入（Jittor变量默认可求导，无需显式设置requires_grad）
            inputs_with_grad = [x.detach() for x in inputs]
            # 重新执行正向传播（此时需要记录中间结果用于求导）
            output = func(*inputs_with_grad)
            # 计算输入和参数的梯度（output对inputs和params的梯度）
            grads = jt.grad(output, inputs_with_grad + params, output_grads)
            # 拆分输入的梯度和参数的梯度
            input_grads = grads[:input_len]
            param_grads = grads[input_len:]
            return input_grads + param_grads

        # 为输出绑定反向传播回调（Jittor的梯度触发机制）
        output.register_hook(backward)
        return output
"""
Helpers to train with 16-bit precision.
"""

import numpy as np
from . import logger
import jittor as jt

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """
    将基础模块转换为float16精度（Jittor版本）
    """
    if isinstance(l, (jt.nn.Conv1d, jt.nn.Conv2d, jt.nn.Conv3d)):
        # 将权重转换为float16
        l.weight = l.weight.astype(jt.float16)
        # 若存在偏置，也转换为float16
        if l.bias is not None:
            l.bias = l.bias.astype(jt.float16)


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (jt.nn.Conv1d, jt.nn.Conv2d, jt.nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()

def _flatten_dense_tensors(tensors):
    """展平并拼接多个张量（Jittor版本）"""
    return jt.cat([t.reshape(-1) for t in tensors])

def make_master_params(param_groups_and_shapes):
    """
    将模型参数复制到全精度参数列表中（Jittor版本）
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        # 提取参数组中的参数并转换为float32
        params_float = [param.detach().astype(jt.float32) for (_, param) in param_group]

        # 展平并拼接所有参数（需确保_flatten_dense_tensors已适配Jittor）
        flattened = _flatten_dense_tensors(params_float)

        # 重塑为目标形状并创建可训练参数
        master_param = jt.nn.Parameter(flattened.view(shape))
        master_param.requires_grad = True  # 启用梯度计算
        master_params.append(master_param)
    return master_params


def make_master_params(param_groups_and_shapes):
    """
    将模型参数复制到全精度参数列表中（Jittor版本）
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        # 提取参数组中的参数并转换为float32
        params_float = [param.detach().astype(jt.float32) for (_, param) in param_group]

        # 展平并拼接所有参数（需确保_flatten_dense_tensors已适配Jittor）
        flattened = _flatten_dense_tensors(params_float)

        # 重塑为目标形状并创建可训练参数
        master_param = jt.nn.Parameter(flattened.view(shape))
        master_param.requires_grad = True  # 启用梯度计算
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)

def _unflatten_dense_tensors(flattened, tensors):
    """
    将展平的张量恢复为原始形状的多个张量（对应torch._utils._unflatten_dense_tensors）
    Args:
        flattened: 展平的一维张量（由_flatten_dense_tensors生成）
        tensors: 原始张量列表（用于获取形状信息）
    Returns:
        恢复形状后的张量列表，与输入tensors的形状一一对应
    """
    # 计算每个原始张量展平后的长度
    sizes = [t.numel() for t in tensors]
    # 分割展平的张量为对应长度的片段
    chunks = jt.split(flattened, sizes)
    # 恢复每个片段为原始形状
    return [chunk.reshape(t.shape) for chunk, t in zip(chunks, tensors)]


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params, optimizer):
    for param in model_params:
        if param.requires_grad:
            # 获取优化器对应的梯度
            grad = param.opt_grad(optimizer)
            if grad is not None:
                # 分离梯度并清零
                grad.detach_()
                grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return jt.zeros_like(param)


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
        opt
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale
        self.optimizer = opt

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self, opt):
        opt.zero_grad()
        # zero_grad(self.model_params, opt)


    def backward(self, loss):
        """
        反向传播函数 (Jittor版本)
        """
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            # 缩放损失并执行反向传播
            (loss * loss_scale).backward()
        else:
            # 直接执行反向传播
            # loss.backward()
            self.optimizer.backward(loss)




    def optimize(self, opt):
        """
        优化步骤 (Jittor版本)
        """
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt):
        """
        FP16优化步骤 (Jittor版本)
        """
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)

        # 计算梯度和参数范数
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)

        # 检查梯度是否溢出（NaN/Inf）
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_grad(self.model_params, opt)  # 清零梯度
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        # 执行优化步骤
        opt.step()

        # 清零梯度
        zero_grad(self.model_params, opt)

        return True

    def _optimize_normal(self, opt):
        """
        标准精度优化步骤 (Jittor版本)
        """
        # 计算梯度和参数范数
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        # 执行优化步骤
        opt.step()

        return True



    def _compute_norms(self, grad_scale=1.0):
        """
        计算梯度和参数的L2范数 (Jittor版本)
        """
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            # 使用Jittor的no_grad上下文
            with jt.no_grad():
                # 计算参数范数的平方
                # 将张量展平为一维向量，再计算范数
                global_l2_norm = jt.norm(p.reshape(-1), p=2)  # reshape(-1) 将张量展平为一维
                param_norm += global_l2_norm.item() ** 2
                # 计算梯度范数的平方（如果存在）
                if p.opt_grad(self.optimizer) is not None:
                    ggg = p.opt_grad(self.optimizer).reshape(-1)
                    grad_norm += jt.norm(ggg, p=2).item() ** 2
        # 返回平方根并应用梯度缩放
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)


    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)

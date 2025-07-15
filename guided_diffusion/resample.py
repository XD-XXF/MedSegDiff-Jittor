from abc import ABC, abstractmethod

import numpy as np
import jittor as jt


def create_named_schedule_sampler(name, diffusion, maxt):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion, maxt)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")





class ScheduleSampler(ABC):
    """
    扩散过程中时间步的采样器，用于减少目标函数的方差 (Jittor版本)
    """

    @abstractmethod
    def weights(self):
        """
        获取每个扩散步骤的权重数组

        权重无需归一化，但必须为正数
        """

    def sample(self, batch_size, device):
        """
        为批次重要性采样时间步

        :param batch_size: 时间步数量
        :param device: 存储张量的设备
        :return: 元组 (timesteps, weights):
                 - timesteps: 时间步索引张量
                 - weights: 用于缩放损失的权重张量
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = jt.array(indices_np).long()
        weights_np = 1 / (len(p) * p[indices_np])
        weights = jt.array(weights_np).float()
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion, maxt):
        self.diffusion = diffusion
        self._weights = np.ones([maxt])

    def weights(self):
        return self._weights





class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        使用模型损失更新重加权策略 (Jittor版本)
        """
        # 获取分布式环境信息
        world_size = jt.world_size
        rank = jt.rank

        # 收集所有进程的批次大小
        batch_sizes = [
            jt.zeros(1, dtype=jt.int32).to(local_ts.device)
            for _ in range(world_size)
        ]
        jt.all_gather(
            batch_sizes,
            jt.array([len(local_ts)], dtype=jt.int32).to(local_ts.device)
        )

        # 填充所有批次到最大批次大小
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        # 收集所有进程的时间步和损失
        timestep_batches = [jt.zeros(max_bs).to(local_ts) for _ in batch_sizes]
        loss_batches = [jt.zeros(max_bs).to(local_losses) for _ in batch_sizes]
        jt.all_gather(timestep_batches, local_ts)
        jt.all_gather(loss_batches, local_losses)

        # 展平收集的结果
        timesteps = [
            int(x.item()) for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [
            float(x.item()) for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]
        ]

        # 使用所有损失更新采样器
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        使用模型损失更新重加权策略 (由子类实现)
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

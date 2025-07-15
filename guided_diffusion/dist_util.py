"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
#from mpi4py import MPI
import jittor as jt
import jittor.distributions as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(args):
    """
    Setup a distributed process group in Jittor.
    """
    # 如果已设置多GPU，则使用Jittor的多GPU模式
    if args.multi_gpu:
        # 设置可用的GPU设备
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev

        # 启用Jittor的多GPU支持
        jt.flags.use_cuda = 1
        jt.flags.nccl_fusion_threshold_mb = 16
        jt.flags.nccl_fusion_max_ops = 24

        # 打印分布式训练信息
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print(f"Running distributed training with local rank {local_rank} and world size {world_size}")

        # 设置当前设备
        jt.set_device(local_rank)
    else:
        # 单GPU模式
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev
        jt.flags.use_cuda = 1


def dev():
    """
    Get the device to use for Jittor.
    """
    if jt.has_cuda:
        return f"cuda:{jt.rank}" if jt.flags.use_cuda else "cpu"
    return "cpu"


def load_state_dict(path, **kwargs):

    return jt.load(path)


def sync_params(params):
    """
    Synchronize model parameters across devices in Jittor.
    """
    # Jittor在分布式训练时会自动同步模型参数
    # 因此这个函数在Jittor中可以简化为空实现
    # 或者根据实际需求使用jt.sync_all()确保所有操作完成
    jt.sync_all()


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

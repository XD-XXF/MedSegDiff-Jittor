import copy
import functools
import os

import blobfile as bf
import jittor
import jittor as jt

from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
# from visdom import Visdom
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.device = jt.flags.use_cuda and "cuda" or "cpu"
        self.dataloader=dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * 1

        self.sync_cuda = 1

        self._load_and_sync_parameters()

        opt = jt.optim.AdamW(
            model.parameters(),  # Jittor通过model.parameters()获取参数
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.opt = opt

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            opt=self.opt
        )


        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]


        self.use_ddp = True
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)

            # 使用Jittor加载检查点
            checkpoint = jt.load(resume_checkpoint)

            # 加载部分状态字典（处理可能的键不匹配）
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)



    def _load_ema_parameters(self, rate):
        # 复制主参数作为初始值
        ema_params = [p.clone() for p in self.mp_trainer.parameters()]

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            print(f"loading EMA from checkpoint: {ema_checkpoint}...")
            # 使用Jittor加载检查点
            state_dict = jt.load(ema_checkpoint)

            # 将状态字典应用到EMA参数
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)

            # 更新EMA参数
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if name in pretrained_dict:
                    ema_params[i] = pretrained_dict[name]

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if not main_checkpoint:  # 若没有主检查点，直接返回
            return

        # 构建优化器检查点路径
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )

        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")

            # 使用Jittor加载检查点
            state_dict = jt.load(opt_checkpoint)

            # 加载优化器状态
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):


            try:
                    batch, cond, name = next(data_iter)
            except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(self.dataloader)
                    batch, cond, name = next(data_iter)



            self.run_step(batch, cond)


            i += 1

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        batch = jt.concat((batch, cond), dim=1)
        cond = {}
        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()  # 确保梯度清零

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch]
            micro_cond = {
                k: v[i: i + self.microbatch].to(self.device)
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1[0]["loss"].detach()
                )

            losses = losses1[0]
            sample = losses1[1]

            loss = (losses["loss"] * weights + losses['loss_cal'] * 10).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            # 使用Jittor优化器直接更新参数
            self.opt.backward(loss=loss, retain_graph=False)



        return sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            # 将主参数转换为状态字典
            state_dict = {}

            if rate == 0:  # 主模型保存
                # 使用模型的命名参数
                named_params = list(self.model.named_parameters())
                for i, param in enumerate(params):
                    if i < len(named_params):
                        name = named_params[i][0]
                        state_dict[name] = param
                    else:
                        state_dict[f'param_{i}'] = param
            else:  # EMA模型保存
                # 尝试保持与主模型相同的参数名称
                named_params = list(self.model.named_parameters())
                for i, (name, _) in enumerate(named_params):
                    if i < len(params):
                        state_dict[name] = params[i]
                    else:
                        break
                # 如果EMA参数比命名参数多，可以添加额外参数
                for i in range(len(named_params), len(params)):
                    state_dict[f'ema_param_{i}'] = params[i]

            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"savedmodel{(self.step + self.resume_step):06d}.pkl"  # 改为.pkl
            else:
                filename = f"emasavedmodel_{rate}_{(self.step + self.resume_step):06d}.pkl"  # 改为.pkl

            save_path = bf.join(get_blob_logdir(), filename)
            jt.save(state_dict, save_path)

        # 保存主模型
        save_checkpoint(0, self.mp_trainer.master_params)

        # 保存EMA模型
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # 保存优化器状态
        logger.log("saving optimizer state...")
        opt_save_path = bf.join(
            get_blob_logdir(), f"optsavedmodel{(self.step + self.resume_step):06d}.pkl"  # 改为.pkl
        )
        jt.save(self.opt.state_dict(), opt_save_path)




def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

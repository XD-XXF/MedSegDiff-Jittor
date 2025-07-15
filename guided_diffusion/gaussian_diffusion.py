"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import enum
import math
import os
# from visdom import Visdom
# viz = Visdom(port=8850)
import numpy as np
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from scipy import ndimage
from .utils import staple, dice_score, norm
from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
import string
import random
import jittor as jt

def standardize(img):
    mean = jt.mean(img)
    std = jt.std(img)
    img = (img - mean) / std
    return img


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    BCE_DICE = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        dpm_solver,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.dpm_solver = dpm_solver

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = jt.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        C=1
        cal = 0
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, cal = model_output
        x=x[:,-1:,...]  #loss is only calculated on the last channel, not on the input brain MR image
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = jt.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = jt.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = jt.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'cal': cal,
        }



    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:

            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, org, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        a, gradient = cond_fn(x, self._scale_timesteps(t),org,  **model_kwargs)


        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return a, new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t,  model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])

        eps = eps.detach() - (1 - alpha_bar).sqrt() *p_mean_var["update"]*0

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x.detach(), t.detach(), eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out, eps


    def sample_known(self, img, batch_size = 1):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop_known(model,(batch_size, channels, image_size, image_size), img)

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = jt.randn_like(x[:, -1:,...])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * jt.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"], "cal": out["cal"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,

    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_known(
            self,
            model,
            shape,
            img,
            step=1000,
            org=None,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            conditioner=None,
            classifier=None
    ):

        assert isinstance(shape, (tuple, list))
        noise = jt.randn_like(img[:, :1, ...])
        x_noisy = jt.concat((img[:, :-1, ...], noise), dim=1)  # add noise as the last channel

        if self.dpm_solver:
            final = {}
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=jt.array(self.betas))

            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs=model_kwargs,
            )

            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                                    correcting_x0_fn="dynamic_thresholding", img=img[:, :-1, ...])

            ## Steps in [20, 30] can generate quite good samples.
            sample, cal = dpm_solver.sample(
                noise.to(dtype=jt.float),
                steps=step,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            sample = sample.detach()  ### MODIFIED: for DPM-Solver OOM issue
            sample[:, -1, :, :] = norm(sample[:, -1, :, :])
            final["sample"] = sample
            final["cal"] = cal

            cal_out = jt.clamp(final["cal"] + 0.25 * final["sample"][:, -1, :, :].unsqueeze(1), 0, 1)
        else:
            print('no dpm-solver')
            i = 0
            letters = string.ascii_lowercase
            name = ''.join(random.choice(letters) for i in range(10))
            for sample in self.p_sample_loop_progressive(
                    model,
                    shape,
                    time=step,
                    noise=x_noisy,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    org=org,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
            ):
                final = sample
                # i += 1
                # '''vis each step sample'''
                # if i % 5 == 0:
                #
                #     o1 = jt.array(img)[:,0,:,:].unsqueeze(1)
                #     o2 = jt.array(img)[:,1,:,:].unsqueeze(1)
                #     o3 = jt.array(img)[:,2,:,:].unsqueeze(1)
                #     o4 = jt.array(img)[:,3,:,:].unsqueeze(1)
                #     s = jt.array(final["sample"])[:,-1,:,:].unsqueeze(1)
                #     tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),s)
                #     compose = jt.concat(tup,0)
                #     print(s)
                #     jt.save_image(s, os.path.join('../res_temp_norm_6000_100', name+str(i)+".jpg"), nrow = 1, padding = 10)

            if dice_score(final["sample"][:, -1, :, :].unsqueeze(1), final["cal"]) < 0.65:
                cal_out = jt.clamp(final["cal"] + 0.25 * final["sample"][:, -1, :, :].unsqueeze(1), 0, 1)
            else:
                cal_out = jt.clamp(final["cal"] * 0.5 + 0.5 * final["sample"][:, -1, :, :].unsqueeze(1), 0, 1)

        return final["sample"], x_noisy, img, final["cal"], cal_out

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            time=1000,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            org=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        逐步生成扩散过程中的中间样本 (Jittor版本)
        """

        assert isinstance(shape, (tuple, list))

        # 初始化噪声或使用提供的噪声
        if noise is not None:
            img = noise
        else:
            img = jt.randn(*shape)

        # 逆序时间步
        indices = list(range(time))[::-1]
        org_c = img.size(1)
        org_MRI = img[:, :-1, ...]  # 原始脑部MR图像

        # 进度条处理
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # 逐步采样
        for i in indices:
            t = jt.array([i] * shape[0])

            with jt.no_grad():
                # 确保图像包含原始MRI和当前分割掩码
                if img.size(1) != org_c:
                    img = jt.concat((org_MRI, img), dim=1)

                # 执行单步采样
                out = self.p_sample(
                    model,
                    img.float(),
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )

                yield out
                img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        使用DDIM从模型采样x_{t-1} (Jittor版本)
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # 从预测的x_start重新推导epsilon
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        # 提取累积alpha值
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)

        # 计算噪声系数sigma
        sigma = (
                eta
                * jt.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * jt.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # 生成随机噪声
        noise = jt.randn_like(x[:, -1:, ...])

        # 计算预测均值
        mean_pred = (
                out["pred_xstart"] * jt.sqrt(alpha_bar_prev)
                + jt.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        # 当t=0时不添加噪声
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        # 计算最终采样结果
        sample = mean_pred + nonzero_mask * sigma * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        使用DDIM反向ODE从模型采样x_{t+1} (Jittor版本)
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # 从预测的x_start重新推导epsilon
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        # 提取下一时间步的累积alpha值
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # 反向ODE的预测均值计算
        mean_pred = (
                out["pred_xstart"] * jt.sqrt(alpha_bar_next)
                + jt.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}



    def ddim_sample_loop_interpolation(
        self,
        model,
        shape,
        img1,
        img2,
        lambdaint,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = jt.randint(499, 500, (b,)).long().to(device)

        img1 = jt.array(img1).to(device)
        img2 = jt.array(img2).to(device)

        noise = jt.randn_like(img1).to(device)
        x_noisy1 = self.q_sample(x_start=img1, t=t, noise=noise).to(device)
        x_noisy2 = self.q_sample(x_start=img2, t=t, noise=noise).to(device)
        interpol = lambdaint * x_noisy1 + (1 - lambdaint) * x_noisy2

        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=interpol,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"], interpol, img1, img2

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        使用DDIM从模型生成样本 (Jittor版本)
        """
        final = None
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = jt.randint(99, 100, (b,)).long().to(device)

        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]



    def ddim_sample_loop_known(
        self,
        model,
        shape,
        img,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]

        img = img.to(device)

        t = jt.randint(499, 500, (b,)).long().to(device)
        noise = jt.randn_like(img[:, :1, ...]).to(device)

        x_noisy = jt.concat((img[:, :-1, ...], noise), dim=1).float()
        img = img.to(device)

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=x_noisy,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample

        return final["sample"], x_noisy, img

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            time=1000,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
    ):
        """
        使用DDIM从模型采样并逐步生成中间样本 (Jittor版本)
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        # 初始化噪声或使用提供的噪声
        if noise is not None:
            img = noise
        else:
            img = jt.randn(*shape, device=device)

        # 逆序时间步（从time-1到0）
        indices = list(range(time - 1))[::-1]
        orghigh = img[:, :-1, ...]  # 保留原始高分辨率部分

        # 进度条处理
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # 逐步采样
        for i in indices:
            t = jt.array([i] * shape[0], device=device)

            with jt.no_grad():
                # 确保图像维度正确
                if img.shape != (1, 5, 224, 224):
                    img = jt.concat((orghigh, img), dim=1).float()

                # 执行DDIM单步采样
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )

            yield out
            img = out["sample"]  # 更新当前样本

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        获取变分下界的项 (Jittor版本)
        """
        # 计算真实后验的均值和方差
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )

        # 使用模型预测均值和方差
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        # 计算KL散度（以比特为单位）
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        # 计算解码器负对数似然（以比特为单位）
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # 在第一个时间步返回解码器NLL，否则返回KL散度
        output = jt.where((t == 0), decoder_nll, kl)

        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses_segmentation(self, model, classifier, x_start, t, model_kwargs=None, noise=None):
        """
        计算单个时间步的训练损失 (Jittor版本)
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = jt.randn_like(x_start[:, -1:, ...])

        mask = x_start[:, -1:, ...]
        res = (mask > 0).float()

        res_t = self.q_sample(res, t, noise=noise)  # 向分割通道添加噪声
        x_t = x_start.float()
        x_t[:, -1:, ...] = res_t.float()
        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.BCE_DICE or self.loss_type == LossType.RESCALED_MSE:
            model_output, cal = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                C = 1
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = jt.split(model_output, C, dim=1)

                # 使用变分下界学习方差，但不影响均值预测
                frozen_out = jt.concat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=res,
                    x_t=res_t,
                    t=t,
                    clip_denoised=False,
                )["output"]

                if self.loss_type == LossType.RESCALED_MSE:
                    # 除以1000以等价于初始实现
                    terms["vb"] *= self.num_timesteps / 1000.0

            # 根据模型类型确定目标值
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=res, x_t=res_t, t=t
                )[0],
                ModelMeanType.START_X: res,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            # 计算损失项
            terms["loss_diff"] = mean_flat((target - model_output) ** 2)
            terms["loss_cal"] = mean_flat((res - cal) ** 2)

            # 合并损失
            if "vb" in terms:
                terms["loss"] = terms["loss_diff"] + terms["vb"]
            else:
                terms["loss"] = terms["loss_diff"]
        else:
            raise NotImplementedError(self.loss_type)

        return (terms, model_output)

    def _prior_bpd(self, x_start):
        """
        计算变分下界的先验KL项（以比特每维度为单位）(Jittor版本)
        """
        batch_size = x_start.shape[0]
        t = jt.array([self.num_timesteps - 1] * batch_size, device=x_start.device)

        # 计算q(x_T|x_0)的均值和方差
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)

        # 计算与标准正态分布的KL散度
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )

        # 转换为比特单位
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        计算完整的变分下界（以比特每维度为单位）及相关指标 (Jittor版本)
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []

        # 逆序遍历所有时间步
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = jt.array([t] * batch_size, device=device)
            noise = jt.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            # 计算当前时间步的变分下界项
            with jt.no_grad():
                out = self._vb_terms_bptimestepsd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )

            # 记录各项指标
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        # 堆叠所有时间步的结果
        vb = jt.stack(vb, dim=1)
        xstart_mse = jt.stack(xstart_mse, dim=1)
        mse = jt.stack(mse, dim=1)

        # 计算先验BPD和总BPD
        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd

        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    从一维numpy数组中提取值以用于一批索引 (Jittor版本)
    """
    # 将numpy数组转换为Jittor张量并提取对应时间步的值
    res = jt.array(arr)[timesteps].float()

    # 添加维度直到与广播形状的维度数匹配
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)  # 在末尾添加维度

    # 扩展张量以匹配广播形状
    return res.broadcast(broadcast_shape)

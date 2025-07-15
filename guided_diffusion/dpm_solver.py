import jittor as jt
import math


class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=jt.float32,
    ):
        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError(
                f"Unsupported noise schedule {schedule}. The schedule needs to be 'discrete' or 'linear' or 'cosine'")

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * jt.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * jt.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = jt.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).astype(dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).astype(dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device),
                                  self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: jt.log(jt.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return jt.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return jt.sqrt(1. - jt.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * jt.log(1. - jt.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * jt.logaddexp(-2. * lamb, jt.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (jt.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * jt.logaddexp(jt.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), jt.flip(self.log_alpha_array.to(lamb.device), [1]),
                               jt.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * jt.logaddexp(-2. * lamb, jt.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: jt.arccos(jt.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


# 辅助函数，需要确保Jittor中有此功能的实现
def interpolate_fn(x, xp, yp):
    """
    Jittor版本的线性插值函数
    """
    x = x.float()
    xp = xp.float()
    yp = yp.float()

    # 找到每个x在xp中的位置
    indices = jt.searchsorted(xp[0], x[:, 0])

    # 处理边界情况
    left = jt.clamp(indices - 1, 0, xp.shape[1] - 1)
    right = jt.clamp(indices, 0, xp.shape[1] - 1)

    # 计算插值权重
    weight = (x[:, 0] - xp[0, left]) / (xp[0, right] - xp[0, left] + 1e-10)
    weight = jt.clamp(weight, 0, 1)

    # 执行插值
    result = weight * yp[0, right] + (1 - weight) * yp[0, left]
    return result.reshape(-1, 1)


import jittor as jt

def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        x_in = x.detach()
        x_in.requires_grad = True
        log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
        return jt.grad(log_prob.sum(), x_in)

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * sigma_t * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = jt.concat([x] * 2)
                t_in = jt.concat([t_continuous] * 2)
                c_in = jt.concat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class DPM_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
        img = None,
    ):
        self.img = img
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method (Jittor版本).
        """
        dims = x0.dim()  # 获取张量维度
        p = self.dynamic_thresholding_ratio  # 分位数参数

        # 计算p分位数（沿最后一个维度，即特征维度）
        # 将x0展平为(batch_size, -1)，然后计算每个样本的p分位数
        s = jt.quantile(jt.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)

        # 确保阈值s不小于最小阈值thresholding_max_val
        # 生成与s同形状的1s张量，乘以thresholding_max_val后与s取最大值
        s = jt.maximum(s, self.thresholding_max_val * jt.ones_like(s).to(s.device))

        # 扩展维度以匹配x0的维度（用于广播操作）
        s = expand_dims(s, dims)

        # 取标量值（Jittor中用item()方法与PyTorch一致）
        s = s.item()

        # 对x0进行裁剪并归一化到[-1, 1]
        x0 = jt.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model (Jittor版本).
        """
        # 拼接图像和输入张量（沿通道维度dim=1），并转换为float32类型
        # Jittor中用jt.cat替换torch.cat，dtype指定为jt.float32
        combined = jt.cat((self.img, x), dim=1).astype(jt.float32)

        # 调用模型进行预测
        out = self.model(combined, t)

        # 如果模型输出是元组，返回第一个元素（通常为噪声预测结果）
        if isinstance(out, tuple):
            return out[0]
        return out

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise[:,0:1,:,:]) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling (Jittor版本).
        Args:
            skip_type: A `str`. The type for the spacing of the time steps.
            t_T: A `float`. The starting time of the sampling.
            t_0: A `float`. The ending time of the sampling.
            N: A `int`. The total number of the spacing of the time steps.
            device: A Jittor device.
        Returns:
            A Jittor tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            # 计算起始和结束时间的lambda值
            lambda_T = self.noise_schedule.marginal_lambda(jt.array([t_T]).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(jt.array([t_0]).to(device))

            # 均匀采样logSNR，再通过inverse_lambda转换回时间
            logSNR_steps = jt.linspace(lambda_T.item(), lambda_0.item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)

        elif skip_type == 'time_uniform':
            # 直接在时间域上均匀采样
            return jt.linspace(t_T, t_0, N + 1).to(device)

        elif skip_type == 'time_quadratic':
            # 二次时间采样（DDIM低分辨率数据用）
            t_order = 2
            t = jt.linspace(t_T ** (1. / t_order), t_0 ** (1. / t_order), N + 1).pow(t_order).to(device)
            return t

        else:
            raise ValueError(
                f"Unsupported skip_type {skip_type}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'")

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver (Jittor版本).
        """
        # 根据steps和order计算每个步骤的阶数
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3, ] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3, ] * (K - 1) + [1]
            else:
                orders = [3, ] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2, ] * K
            else:
                K = steps // 2 + 1
                orders = [2, ] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1, ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")

        # 计算时间步
        if skip_type == 'logSNR':
            # 按logSNR均匀采样时间步
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            # 计算累积索引，用于从完整时间步中截取所需部分
            # Jittor中用jt.cumsum和jt.array替换PyTorch的对应操作
            indices = jt.cumsum(jt.concat([jt.array([0]), jt.array(orders)]), 0)
            # 获取完整时间步后按索引截取
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[indices]

        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t` (Jittor版本).
        """
        ns = self.noise_schedule
        dims = x.dim()  # 获取张量维度

        # 计算噪声调度相关参数（lambda、log_alpha、sigma等）
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s  # 时间步差（logSNR空间）
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = jt.exp(log_alpha_t)  # 计算alpha_t

        if self.algorithm_type == "dpmsolver++":
            # dpmsolver++ 算法的更新公式
            phi_1 = jt.expm1(-h)  # 计算phi函数（expm1(x) = exp(x) - 1）

            # 若未提供model_s，则通过模型计算
            if model_s is None:
                model_s = self.model_fn(x, s)

            # 计算t时刻的近似解x_t
            x_t = (
                    sigma_t / sigma_s * x
                    - alpha_t * phi_1 * model_s
            )
        else:
            # 标准dpmsolver算法的更新公式
            phi_1 = jt.expm1(h)

            # 若未提供model_s，则通过模型计算
            if model_s is None:
                model_s = self.model_fn(x, s)

            # 计算t时刻的近似解x_t
            x_t = (
                    jt.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
            )

        # 根据需求返回中间结果
        if return_intermediate:
            return x_t, {'model_s': model_s}
        else:
            return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False,
                                            solver_type='dpmsolver'):
        """
        DPM-Solver-2 单步更新函数 (Jittor版本)
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError(f"'solver_type' must be either 'dpmsolver' or 'taylor', got {solver_type}")
        if r1 is None:
            r1 = 0.5

        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)

        # 计算噪声调度相关参数
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s = ns.marginal_std(s)
        sigma_s1 = ns.marginal_std(s1)
        sigma_t = ns.marginal_std(t)
        alpha_s1 = jt.exp(log_alpha_s1)
        alpha_t = jt.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = jt.expm1(-r1 * h)
            phi_1 = jt.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)

            # 计算中间点s1处的x值
            x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    - (alpha_s1 * phi_11) * model_s
            )

            # 计算中间点s1处的模型输出
            model_s1 = self.model_fn(x_s1, s1)

            if solver_type == 'dpmsolver':
                # DPM-Solver-2 标准公式
                x_t = (
                        (sigma_t / sigma_s) * x
                        - (alpha_t * phi_1) * model_s
                        - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                # Taylor展开版本
                x_t = (
                        (sigma_t / sigma_s) * x
                        - (alpha_t * phi_1) * model_s
                        + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s)
                )
        else:
            phi_11 = jt.expm1(r1 * h)
            phi_1 = jt.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)

            # 计算中间点s1处的x值
            x_s1 = (
                    jt.exp(log_alpha_s1 - log_alpha_s) * x
                    - (sigma_s1 * phi_11) * model_s
            )

            # 计算中间点s1处的模型输出
            model_s1 = self.model_fn(x_s1, s1)

            if solver_type == 'dpmsolver':
                # DPM-Solver-2 标准公式
                x_t = (
                        jt.exp(log_alpha_t - log_alpha_s) * x
                        - (sigma_t * phi_1) * model_s
                        - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                # Taylor展开版本
                x_t = (
                        jt.exp(log_alpha_t - log_alpha_s) * x
                        - (sigma_t * phi_1) * model_s
                        - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s)
                )

        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1. / 3., r2=2. / 3., model_s=None, model_s1=None,
                                           return_intermediate=False, solver_type='dpmsolver'):
        """
        三阶单步DPM-Solver更新（Jittor版本）
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError(f"'solver_type' must be either 'dpmsolver' or 'taylor', got {solver_type}")
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.

        ns = self.noise_schedule
        # 计算logSNR相关参数
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s  # logSNR步长
        lambda_s1 = lambda_s + r1 * h  # 中间点1的logSNR
        lambda_s2 = lambda_s + r2 * h  # 中间点2的logSNR
        s1 = ns.inverse_lambda(lambda_s1)  # 中间点1的时间
        s2 = ns.inverse_lambda(lambda_s2)  # 中间点2的时间

        # 计算噪声调度的核心参数
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_s2 = ns.marginal_log_mean_coeff(s2)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s = ns.marginal_std(s)
        sigma_s1 = ns.marginal_std(s1)
        sigma_s2 = ns.marginal_std(s2)
        sigma_t = ns.marginal_std(t)
        alpha_s1 = jt.exp(log_alpha_s1)
        alpha_s2 = jt.exp(log_alpha_s2)
        alpha_t = jt.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            # 计算phi函数（dpmsolver++版本）
            phi_11 = jt.expm1(-r1 * h)
            phi_12 = jt.expm1(-r2 * h)
            phi_1 = jt.expm1(-h)
            phi_22 = jt.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            # 获取s时刻的模型输出（若未提供）
            if model_s is None:
                model_s = self.model_fn(x, s)
            # 获取s1时刻的模型输出（若未提供）
            if model_s1 is None:
                x_s1 = (
                        (sigma_s1 / sigma_s) * x
                        - (alpha_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            # 计算s2时刻的状态和模型输出
            x_s2 = (
                    (sigma_s2 / sigma_s) * x
                    - (alpha_s2 * phi_12) * model_s
                    + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)

            # 根据求解器类型计算t时刻的状态
            if solver_type == 'dpmsolver':
                x_t = (
                        (sigma_t / sigma_s) * x
                        - (alpha_t * phi_1) * model_s
                        + (1. / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                        (sigma_t / sigma_s) * x
                        - (alpha_t * phi_1) * model_s
                        + (alpha_t * phi_2) * D1
                        - (alpha_t * phi_3) * D2
                )
        else:
            # 计算phi函数（标准dpmsolver版本）
            phi_11 = jt.expm1(r1 * h)
            phi_12 = jt.expm1(r2 * h)
            phi_1 = jt.expm1(h)
            phi_22 = jt.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            # 获取s时刻的模型输出（若未提供）
            if model_s is None:
                model_s = self.model_fn(x, s)
            # 获取s1时刻的模型输出（若未提供）
            if model_s1 is None:
                x_s1 = (
                        jt.exp(log_alpha_s1 - log_alpha_s) * x
                        - (sigma_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            # 计算s2时刻的状态和模型输出
            x_s2 = (
                    jt.exp(log_alpha_s2 - log_alpha_s) * x
                    - (sigma_s2 * phi_12) * model_s
                    - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)

            # 根据求解器类型计算t时刻的状态
            if solver_type == 'dpmsolver':
                x_t = (
                        jt.exp(log_alpha_t - log_alpha_s) * x
                        - (sigma_t * phi_1) * model_s
                        - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                        jt.exp(log_alpha_t - log_alpha_s) * x
                        - (sigma_t * phi_1) * model_s
                        - (sigma_t * phi_2) * D1
                        - (sigma_t * phi_3) * D2
                )

        # 返回结果（可选包含中间状态）
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        多步二阶DPM-Solver更新（Jittor版本）
        从时间步`t_prev_list[-1]`更新到时间`t`
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError(f"'solver_type' must be either 'dpmsolver' or 'taylor', got {solver_type}")

        ns = self.noise_schedule
        # 提取前序模型输出和时间步
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]

        # 计算logSNR和噪声调度参数
        lambda_prev_1 = ns.marginal_lambda(t_prev_1)
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        log_alpha_prev_0 = ns.marginal_log_mean_coeff(t_prev_0)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0 = ns.marginal_std(t_prev_0)
        sigma_t = ns.marginal_std(t)
        alpha_t = jt.exp(log_alpha_t)

        # 计算步长和导数估计
        h_0 = lambda_prev_0 - lambda_prev_1  # 前序步长（logSNR空间）
        h = lambda_t - lambda_prev_0  # 当前步长（logSNR空间）
        r0 = h_0 / h  # 步长比例
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)  # 一阶导数估计

        if self.algorithm_type == "dpmsolver++":
            # dpmsolver++ 算法的更新公式
            phi_1 = jt.expm1(-h)

            if solver_type == 'dpmsolver':
                x_t = (
                        (sigma_t / sigma_prev_0) * x
                        - (alpha_t * phi_1) * model_prev_0
                        - 0.5 * (alpha_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                        (sigma_t / sigma_prev_0) * x
                        - (alpha_t * phi_1) * model_prev_0
                        + (alpha_t * (phi_1 / h + 1.)) * D1_0
                )
        else:
            # 标准dpmsolver算法的更新公式
            phi_1 = jt.expm1(h)

            if solver_type == 'dpmsolver':
                x_t = (
                        jt.exp(log_alpha_t - log_alpha_prev_0) * x
                        - (sigma_t * phi_1) * model_prev_0
                        - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                        jt.exp(log_alpha_t - log_alpha_prev_0) * x
                        - (sigma_t * phi_1) * model_prev_0
                        - (sigma_t * (phi_1 / h - 1.)) * D1_0
                )

        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        多步三阶DPM-Solver更新 (Jittor版本)
        """
        ns = self.noise_schedule
        # 提取前三个时间步的模型输出和时间
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list

        # 计算各时间点的logSNR和噪声调度参数
        lambda_prev_2 = ns.marginal_lambda(t_prev_2)
        lambda_prev_1 = ns.marginal_lambda(t_prev_1)
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)

        log_alpha_prev_0 = ns.marginal_log_mean_coeff(t_prev_0)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0 = ns.marginal_std(t_prev_0)
        sigma_t = ns.marginal_std(t)
        alpha_t = jt.exp(log_alpha_t)

        # 计算各时间步之间的logSNR差值（步长）
        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0

        # 计算步长比例
        r0, r1 = h_0 / h, h_1 / h

        # 计算一阶和二阶导数估计
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)

        if self.algorithm_type == "dpmsolver++":
            # dpmsolver++算法的更新公式
            phi_1 = jt.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
            )
        else:
            # 标准dpmsolver算法的更新公式
            phi_1 = jt.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            x_t = (
                    jt.exp(log_alpha_t - log_alpha_prev_0) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
            )

        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5,
                            solver_type='dpmsolver'):
        """
        自适应步长DPM-Solver (Jittor版本)
        """
        ns = self.noise_schedule
        s = jt.full((1,), t_T, dtype=x.dtype).to(x.device)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(jt.full_like(s, t_0))
        h = jt.full_like(s, h_init)
        x_prev = x
        nfe = 0  # 函数评估次数

        # 根据阶数选择对应的低阶和高阶更新函数
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1,
                                                                                               solver_type=solver_type,
                                                                                               **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1,
                                                                                    return_intermediate=True,
                                                                                    solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2,
                                                                                              solver_type=solver_type,
                                                                                              **kwargs)
        else:
            raise ValueError(f"For adaptive step size solver, order must be 2 or 3, got {order}")

        # 自适应步长迭代
        while jt.abs(s - t_0).mean() > t_err:
            # 计算下一时间点
            t = ns.inverse_lambda(lambda_s + h)

            # 低阶方法更新
            x_lower, lower_noise_kwargs = lower_update(x, s, t)

            # 高阶方法更新
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)

            # 计算误差容限
            delta = jt.max(jt.ones_like(x) * atol, rtol * jt.max(jt.abs(x_lower), jt.abs(x_prev)))

            # 计算误差范数
            def norm_fn(v):
                return jt.sqrt(jt.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdims=True))

            E = norm_fn((x_higher - x_lower) / delta).max()

            # 根据误差决定是否接受当前步
            if E <= 1.:
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)

            # 自适应调整步长
            h = jt.min(theta * h * jt.pow(E, -1. / order), lambda_0 - lambda_s)
            nfe += order

        print('adaptive solver nfe', nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise (Jittor版本).
        """
        # 获取噪声调度的alpha和sigma参数
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)

        # 如果没有提供噪声，则生成随机噪声
        if noise is None:
            noise = jt.randn((t.shape[0], *x.shape), dtype=x.dtype, device=x.device)

        # 调整输入x的形状以匹配噪声维度
        x = x.reshape((-1, *x.shape))

        # 扩展alpha_t和sigma_t的维度以支持广播
        # 注意：expand_dims需要确保在Jittor中实现
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise

        # 如果t只有一个元素，则去除额外的维度
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
               method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
               atol=0.0078, rtol=0.05, return_intermediate=False,
               ):
        """
        基于DPM-Solver的采样函数 (Jittor版本)
        """
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"

        if return_intermediate:
            assert method in ['multistep', 'singlestep',
                              'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep',
                              'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"

        device = x.device
        intermediates = []

        with jt.no_grad():
            if method == 'adaptive':
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol,
                                             solver_type=solver_type)

            elif method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps

                # 初始化
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]

                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                # 使用低阶多步DPM-Solver初始化前`order`个值
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step,
                                                         solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))

                # 使用`order`阶多步DPM-Solver计算剩余值
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # 对于steps < 10，使用较低阶数
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order

                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order,
                                                         solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)

                    # 更新历史值列表
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t

                    # 最后一步不需要计算模型值
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)

            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps,
                                                                                                  order=order,
                                                                                                  skip_type=skip_type,
                                                                                                  t_T=t_T, t_0=t_0,
                                                                                                  device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order, ] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)

                # 单步求解
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order,
                                                          device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h

                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)

            else:
                raise ValueError(f"Got wrong method {method}")

            # 去噪到零
            if denoise_to_zero:
                t = jt.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)

        # 最终模型调用
        cal = None
        out = self.model(jt.cat((self.img, x), dim=1).astype(jt.float32), t)
        if isinstance(out, tuple):
            x, cal = out

        if return_intermediate:
            return x, intermediates
        else:
            return x, cal



#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    分段线性插值函数 (Jittor版本)
    """
    N, K = x.shape[0], xp.shape[1]

    # 将输入x与关键点xp合并并排序
    all_x = jt.concat([x.unsqueeze(2), xp.unsqueeze(0).expand(N, -1, -1)], dim=2)
    sorted_all_x, x_indices = jt.argsort(all_x, dim=2)

    # 找到x在排序后的索引位置
    x_idx = jt.argmin(x_indices, dim=2)

    # 计算插值区间的起始索引
    cand_start_idx = x_idx - 1
    start_idx = jt.where(
        jt.eq(x_idx, 0),
        jt.zeros_like(x_idx) + 1,  # 替代torch.tensor(1, device=x.device)
        jt.where(
            jt.eq(x_idx, K), jt.zeros_like(x_idx) + (K - 2), cand_start_idx,
        ),
    )

    # 计算插值区间的结束索引
    end_idx = jt.where(jt.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)

    # 获取区间的x值
    start_x = jt.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = jt.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)

    # 重新计算起始索引以获取y值
    start_idx2 = jt.where(
        jt.eq(x_idx, 0),
        jt.zeros_like(x_idx),  # 替代torch.tensor(0, device=x.device)
        jt.where(
            jt.eq(x_idx, K), jt.zeros_like(x_idx) + (K - 2), cand_start_idx,
        ),
    )

    # 获取区间的y值
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = jt.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = jt.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)

    # 线性插值计算结果
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand



def expand_dims(v, dims):
    """
    扩展张量维度 (Jittor版本)
    """
    # 在末尾添加(dims - 1)个维度，每个维度的大小为1
    return v.reshape(v.shape + (1,) * (dims - v.ndim))


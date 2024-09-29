"""This file contains code for sde.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/baofff/U-ViT/blob/main/sde.py
"""

import torch
import torch.nn as nn
from absl import logging
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F


def check_zip(*args):
    args = [list(arg) for arg in args]
    length = len(args[0])
    for arg in args:
        assert len(arg) == length
    return zip(*args)

def get_sde(name, **kwargs):
    if name == 'vpsde':
        return VPSDE(**kwargs)
    elif name == 'vpsde_cosine':
        return VPSDECosine(**kwargs)
    else:
        raise NotImplementedError


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)


class SDE(object):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def drift(self, x, t):
        raise NotImplementedError

    def diffusion(self, t):
        raise NotImplementedError

    def cum_beta(self, t):  # the variance of xt|x0
        raise NotImplementedError

    def cum_alpha(self, t):
        raise NotImplementedError

    def snr(self, t):  # signal noise ratio
        raise NotImplementedError

    def nsr(self, t):  # noise signal ratio
        raise NotImplementedError

    def marginal_prob(self, x0, t):  # the mean and std of q(xt|x0)
        alpha = self.cum_alpha(t)
        beta = self.cum_beta(t)
        mean = stp(alpha ** 0.5, x0)  # E[xt|x0]
        std = beta ** 0.5  # Cov[xt|x0] ** 0.5
        return mean, std

    def sample(self, x0, t_init=0):  # sample from q(xn|x0), where n is uniform
        t = torch.rand(x0.shape[0], device=x0.device) * (1. - t_init) + t_init
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn_like(x0)
        xt = mean + stp(std, eps)
        return t, eps, xt


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20):
        # 0 <= t <= 1
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def drift(self, x, t):
        return -0.5 * stp(self.squared_diffusion(t), x)

    def diffusion(self, t):
        return self.squared_diffusion(t) ** 0.5

    def squared_diffusion(self, t):  # beta(t)
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def squared_diffusion_integral(self, s, t):  # \int_s^t beta(tau) d tau
        return self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5

    def skip_beta(self, s, t):  # beta_{t|s}, Cov[xt|xs]=beta_{t|s} I
        return 1. - self.skip_alpha(s, t)

    def skip_alpha(self, s, t):  # alpha_{t|s}, E[xt|xs]=alpha_{t|s}**0.5 xs
        x = -self.squared_diffusion_integral(s, t)
        return x.exp()

    def cum_beta(self, t):
        return self.skip_beta(0, t)

    def cum_alpha(self, t):
        return self.skip_alpha(0, t)

    def nsr(self, t):
        nsr = self.squared_diffusion_integral(0, t).expm1()
        nsr = nsr.clamp(max = 1e6, min = 1e-12)
        return nsr

    def snr(self, t):
        snr = 1. / self.nsr(t)
        snr = snr.clamp(max = 1e6, min = 1e-12)
        return snr

    def __str__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'

    def __repr__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'


class VPSDECosine(SDE):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def __init__(self, s=0.008):
        self.s = s
        self.F = lambda t: torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        self.F0 = math.cos(s / (1 + s) * math.pi / 2) ** 2

    def drift(self, x, t):
        ft = - torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi / 2
        return stp(ft, x)

    def diffusion(self, t):
        return (torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi) ** 0.5

    def cum_beta(self, t):  # the variance of xt|x0
        return 1 - self.cum_alpha(t)

    def cum_alpha(self, t):
        return self.F(t) / self.F0

    def snr(self, t):  # signal noise ratio
        Ft = self.F(t)
        snr = Ft / (self.F0 - Ft)
        snr = snr.clamp(max = 1e6, min = 1e-12)
        return snr

    def nsr(self, t):  # noise signal ratio
        Ft = self.F(t)
        nsr = self.F0 / Ft - 1
        nsr = nsr.clamp(max = 1e6, min = 1e-12)
        return nsr

    def __str__(self):
        return 'vpsde_cosine'

    def __repr__(self):
        return 'vpsde_cosine'


class ScoreModel(object):
    r"""
        The forward process is q(x_[0,T])
    """

    def __init__(self, nnet: nn.Module, loss_coeffs:list, sde: SDE, using_cfg: bool = False, T=1):
        assert T == 1
        self.nnet = nnet
        self.loss_coeffs = loss_coeffs
        self.sde = sde
        self.T = T
        self.using_cfg = using_cfg
        print(f'ScoreModel with loss_coeffs={loss_coeffs}, sde={sde}, T={T}')

    def predict(self, xt, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = duplicate(t, xt.size(0))
        log_snr = self.sde.snr(t).log()
        
        return self.nnet(xt, t = t * 999, log_snr = log_snr, **kwargs)  # follow SDE

    def noise_pred(self, xt, t, sampling = True, **kwargs):
        if sampling:
            if self.using_cfg:
                return self.predict(xt, t, **kwargs)
            else:
                return self.predict(xt, t, **kwargs)[-1]
        else:
            return self.predict(xt, t, **kwargs)

    def score(self, xt, t, **kwargs):
        cum_beta = self.sde.cum_beta(t)
        noise_pred = self.noise_pred(xt, t, sampling = True, **kwargs)
        return stp(-cum_beta.rsqrt(), noise_pred)


class ReverseSDE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt + g(t) dw
    """
    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, t, **kwargs)
        return drift - stp(diffusion ** 2, score)

    def diffusion(self, t):
        return self.sde.diffusion(t)


class ODE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt
    """

    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, t, **kwargs)
        return drift - 0.5 * stp(diffusion ** 2, score)

    def diffusion(self, t):
        return 0


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


@ torch.no_grad()
def euler_maruyama(rsde, x_init, sample_steps, eps=1e-3, T=1, trace=None, verbose=False, **kwargs):
    r"""
    The Euler Maruyama sampler for reverse SDE / ODE
    See `Score-Based Generative Modeling through Stochastic Differential Equations`
    """
    assert isinstance(rsde, ReverseSDE) or isinstance(rsde, ODE)
    print(f"euler_maruyama with sample_steps={sample_steps}")
    timesteps = np.append(0., np.linspace(eps, T, sample_steps))
    timesteps = torch.tensor(timesteps).to(x_init)
    x = x_init
    if trace is not None:
        trace.append(x)
    for s, t in tqdm(list(zip(timesteps, timesteps[1:]))[::-1], disable=not verbose, desc='euler_maruyama'):
        drift = rsde.drift(x, t, **kwargs)
        diffusion = rsde.diffusion(t)
        dt = s - t
        mean = x + drift * dt
        sigma = diffusion * (-dt).sqrt()
        x = mean + stp(sigma, torch.randn_like(x)) if s != 0 else mean
        if trace is not None:
            trace.append(x)
        statistics = dict(s=s, t=t, sigma=sigma.item())
        logging.debug(dct2str(statistics))
    return x


def LSimple(score_model: ScoreModel, x0, **kwargs):
    t, noise, xt = score_model.sde.sample(x0)
    prediction = score_model.noise_pred(xt, t, sampling = False, **kwargs)
    target = multi_scale_targets(noise, levels = len(prediction), scale_correction = True)
    loss = 0
    for pred, coeff in check_zip(prediction, score_model.loss_coeffs):
        loss = loss + coeff * mos(pred - target[pred.shape[-1]])
    return loss


def odd_multi_scale_targets(target, levels, scale_correction):
    B, C, H, W = target.shape
    targets = {}
    for l in range(levels):
        ratio = int(2 ** l)
        if ratio == 1:
            targets[target.shape[-1]] = target
            continue
        assert (H - 1) % ratio == 0 and (W - 1) % ratio == 0
        KS = ratio + 1
        scale = KS if scale_correction else KS ** 2
        kernel = torch.ones(C, 1, KS, KS, device = target.device) / scale            
        downsampled = F.conv2d(target, kernel, stride = ratio, padding = KS // 2, groups = C)
        targets[downsampled.shape[-1]] = downsampled
    return targets

def even_multi_scale_targets(target, levels, scale_correction):
    B, C, H, W = target.shape
    targets = {}
    for l in range(levels):
        ratio = int(2 ** l)
        if ratio == 1:
            targets[target.shape[-1]] = target
            continue
        assert H % ratio == 0 and W % ratio == 0
        KS = ratio
        scale = KS if scale_correction else KS ** 2
        kernel = torch.ones(C, 1, KS, KS, device = target.device) / scale            
        downsampled = F.conv2d(target, kernel, stride = ratio, groups = C)
        targets[downsampled.shape[-1]] = downsampled
    return targets
    
def multi_scale_targets(target, levels, scale_correction):
    B, C, H, W = target.shape
    if H % 2 == 0:
        return even_multi_scale_targets(target, levels, scale_correction)
    else:
        return odd_multi_scale_targets(target, levels, scale_correction)
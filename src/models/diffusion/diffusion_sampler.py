from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .denoiser import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    s_cond: float = 0
    warm_start: bool = False  # use previous frame to initialize x
    warm_sigma_cond: float = 0.05  # sigma for warm start conditioning
    warm_sigma_offset_noise: float = 0.05  # offset noise for warm start


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)
        self.is_first_frame = True  
        self.last_frame: Optional[Tensor] = None  # warm start state

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Optional[Tensor], prev_mask: Optional[Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)
        if prev_mask is not None:
            prev_obs = torch.cat((prev_obs, prev_mask), dim=1)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        # init x: default random noise; warm start uses last frame with small noise
        if self.cfg.warm_start and (not self.is_first_frame) and (self.last_frame is not None):  # warm start
            sigma_cond = torch.full((b,), fill_value=self.cfg.warm_sigma_cond, device=device)
            x = self.denoiser.apply_noise(self.last_frame, sigma_cond, sigma_offset_noise=self.cfg.warm_sigma_offset_noise)
        else:
            x = torch.randn(b, c, h, w, device=device)
            self.is_first_frame = False
        trajectory = [x]
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            if self.cfg.s_cond > 0:
                sigma_cond = torch.full((b,), fill_value=self.cfg.s_cond, device=device)
                prev_obs = self.denoiser.apply_noise(prev_obs, sigma_cond, sigma_offset_noise=0)
            else:
                sigma_cond = None
            denoised = self.denoiser.denoise(x, sigma, sigma_cond, prev_obs, prev_act)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, sigma_cond, prev_obs, prev_act)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)
        if self.cfg.warm_start:
            self.last_frame = x
        return x, trajectory


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))

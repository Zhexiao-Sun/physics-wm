from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import cv2

from data import Batch
from .inner_model import InnerModel, InnerModelConfig
from utils import LossAndLogs

from models.mask_utils import build_mask_from_fullres, set_mask_params  

# Utility to broadcast scalars to N-D tensors
def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor
    c_noise_cond: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float
    noise_previous_obs: bool
    mask_gain_train: float = 1.0  
    mask_gain_infer: float = 0.35  
    upsampling_factor: Optional[int] = None
    mask_w_blue: float = 1.0 
    mask_w_green: float = 0.6  
    mask_sigma_global_ratio: float = 0.20 
    mask_sigma_ego_scale: float = 0.50 
    mask_sigma_min_px: float = 2.0  
    mask_downsample_mode: str = "bicubic" 
    mask_dropout_train: float = 0.30  


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_upsampler = cfg.upsampling_factor is not None
        cfg.inner_model.is_upsampler = self.is_upsampler
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None
        if not self.is_upsampler:
            try:
                set_mask_params(
                    w_blue=self.cfg.mask_w_blue,
                    w_green=self.cfg.mask_w_green,
                    sigma_global_ratio=self.cfg.mask_sigma_global_ratio,
                    sigma_ego_scale=self.cfg.mask_sigma_ego_scale,
                    sigma_min_px=self.cfg.mask_sigma_min_px,
                    downsample_mode=self.cfg.mask_downsample_mode,
                )
            except Exception:
                pass  

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma
    
    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor, sigma_cond: Optional[Tensor]) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise, c_noise_cond), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Optional[Tensor], cs: Conditioners) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, cs.c_noise_cond, rescaled_obs, act)
    
    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1) 
        return d    
    
    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, sigma_cond: Optional[Tensor], obs: Tensor, act: Optional[Tensor]) -> Tensor:
        cs = self.compute_conditioners(sigma, sigma_cond)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised
    
    def forward(self, batch: Batch) -> LossAndLogs:
        b, t, c, h, w = batch.obs.size()
        H, W = (self.cfg.upsampling_factor * h, self.cfg.upsampling_factor * w) if self.is_upsampler else (h, w)
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = t - n  

        if self.is_upsampler:
            all_obs = torch.stack([x["full_res"] for x in batch.info]).to(self.device)
            low_res = F.interpolate(batch.obs.reshape(b * t, c, h, w), scale_factor=self.cfg.upsampling_factor, mode="bicubic").reshape(b, t, c, H, W)
            assert all_obs.shape == low_res.shape
        else:
            all_obs = batch.obs.clone()
            full_res_all = None
            if isinstance(batch.info, list) and len(batch.info) > 0 and isinstance(batch.info[0], dict) and ("full_res" in batch.info[0]):
                try:
                    full_res_all = torch.stack([x["full_res"] for x in batch.info]).to(self.device)
                except Exception:
                    full_res_all = None

        loss = 0
        for i in range(seq_length): 
            prev_obs = all_obs[:, i : n + i].reshape(b, n * c, H, W)
            prev_act = None if self.is_upsampler else batch.act[:, i : n + i]
            obs = all_obs[:, n + i]
            mask = batch.mask_padding[:, n + i]

            if (not self.is_upsampler) and getattr(self.cfg.inner_model, "extra_cond_channels", 0) > 0:
                if full_res_all is not None:
                    fr = full_res_all[:, n + i - 1]
                    m = build_mask_from_fullres(fr, H, W, self.cfg.mask_gain_train)  # soft mask from full-res
                else:
                    m = torch.zeros(b, 1, H, W, device=self.device)  
            else:
                m = None

            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(b, self.device)
                prev_obs_noisy = self.apply_noise(prev_obs, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None
                prev_obs_noisy = prev_obs

            if self.is_upsampler:
                prev_obs_input = torch.cat((prev_obs_noisy, low_res[:, n + i]), dim=1)
            else:
                # training-time mask dropout by zeroing mask channel
                if (m is not None) and self.training and getattr(self.cfg, "mask_dropout_train", 0.0) > 0:
                    if float(torch.rand(()).item()) < self.cfg.mask_dropout_train:
                        m = torch.zeros_like(m) 
                prev_obs_input = torch.cat((prev_obs_noisy, m), dim=1) if m is not None else prev_obs_noisy  # concat mask

            sigma = self.sample_sigma_training(b, self.device)
            noisy_obs = self.apply_noise(obs, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma, sigma_cond)
            model_output = self.compute_model_output(noisy_obs, prev_obs_input, prev_act, cs)
            
            target = (obs - cs.c_skip * noisy_obs) / cs.c_out  # obs.shape --> torch.Size([1, 3, 30, 120]), but raw image size: [150, 600]

            loss += F.mse_loss(model_output[mask], target[mask]) 

            denoised = self.wrap_model_output(noisy_obs, model_output, cs)
            all_obs[:, n + i] = denoised

        loss /= seq_length
        return loss, {"loss_denoising": loss.item()}

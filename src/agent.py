from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from utils import extract_state_dict


@dataclass
class AgentConfig:
    denoiser: DenoiserConfig
    upsampler: Optional[DenoiserConfig]
    num_actions: int

    def __post_init__(self) -> None:
        self.denoiser.inner_model.num_actions = self.num_actions
        if self.upsampler is not None:
            self.upsampler.inner_model.num_actions = self.num_actions


class Agent(nn.Module):
    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.denoiser = Denoiser(cfg.denoiser)
        self.upsampler = Denoiser(cfg.upsampler) if cfg.upsampler is not None else None

    @property
    def device(self):
        return self.denoiser.device

    def setup_training(
        self,
        sigma_distribution_cfg: SigmaDistributionConfig,
        sigma_distribution_cfg_upsampler: Optional[SigmaDistributionConfig],
    ) -> None:
        self.denoiser.setup_training(sigma_distribution_cfg)
        if self.upsampler is not None:
            self.upsampler.setup_training(sigma_distribution_cfg_upsampler)

    def load(
        self,
        path_to_ckpt: Path,
        load_denoiser: bool = True,
        load_upsampler: bool = True,
    ) -> None:
        sd = torch.load(Path(path_to_ckpt), map_location=self.device, weights_only=True) 
        if load_denoiser:
            self.denoiser.load_state_dict(extract_state_dict(sd, "denoiser"))
        if load_upsampler:
            self.upsampler.load_state_dict(extract_state_dict(sd, "upsampler"))


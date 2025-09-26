import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from agent import Agent
from envs import WorldModelEnv
from game import Game, PlayEnv


OmegaConf.register_new_resolver("eval", eval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record", action="store_true", help="Record episodes in PlayEnv.")
    parser.add_argument("--store-denoising-trajectory", action="store_true", help="Save denoising steps in info.")
    parser.add_argument("--store-original-obs", action="store_true", help="Save original obs (pre resizing) in info.")
    parser.add_argument("--mouse-multiplier", type=int, default=10, help="Multiplication factor for the mouse movement.")
    parser.add_argument("--size-multiplier", type=int, default=2, help="Multiplication factor for the screen size.")
    parser.add_argument("--compile", action="store_true", help="Turn on model compilation.")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate.")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--warmstart", action="store_true", help="Warm start the diffusion sampler.")  
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if not args.record and (args.store_denoising_trajectory or args.store_original_obs):
        print("Warning: not in recording mode, ignoring --store* options")
    return True


def prepare_play_mode(cfg: DictConfig, args: argparse.Namespace) -> PlayEnv:

    # Hugging Face
    repo_id = "TUM/PIWM_ckpt"
    path_hf = Path(snapshot_download(repo_id=repo_id, allow_patterns=["PIWM/**"]))

    path_ckpt = path_hf / "PIWM/model/piwm.pt"
    spawn_dir = path_hf / "PIWM/spwan"

    # Override config
    cfg.agent = OmegaConf.load(path_hf / "PIWM/config/agent/piwm.yaml") 
    cfg.env   = OmegaConf.load(path_hf / "PIWM/config/env/piwm.yaml")  

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("----------------------------------------------------------------------")
    print(f"Using {device} for rendering.")
    if not torch.cuda.is_available():
        print("If you have a CUDA GPU available and it is not being used, please follow the instructions at https://pytorch.org/get-started/locally/ to reinstall torch with CUDA support and try again.")
    print("----------------------------------------------------------------------")

    assert cfg.env.train.id == "piwm"
    num_actions = cfg.env.num_actions

    # Models
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(path_ckpt)
    
    # World model environment
    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    if args.warmstart:
        wm_env_cfg.diffusion_sampler_next_obs.warm_start = True
        if wm_env_cfg.diffusion_sampler_upsampling is not None:
            wm_env_cfg.diffusion_sampler_upsampling.warm_start = True
    wm_env = WorldModelEnv(agent.denoiser, agent.upsampler, spawn_dir, 1, sl, wm_env_cfg, return_denoising_trajectory=True)
    
    if device.type == "cuda" and args.compile:
        print("Compiling models...")
        wm_env.predict_next_obs = torch.compile(wm_env.predict_next_obs, mode="reduce-overhead")
        wm_env.upsample_next_obs = torch.compile(wm_env.upsample_next_obs, mode="reduce-overhead")

    play_env = PlayEnv(
        agent,
        wm_env,
        args.record,
        args.store_denoising_trajectory,
        args.store_original_obs,
    )

    return play_env


@torch.no_grad()
def main():
    args = parse_args()
    ok = check_args(args)
    if not ok:
        return

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    # window size
    h, w = (cfg.env.train.size,) * 2 if isinstance(cfg.env.train.size, int) else cfg.env.train.size
    size_h, size_w = h * args.size_multiplier, w * args.size_multiplier
    env = prepare_play_mode(cfg, args)
    game = Game(env, (size_h, size_w), args.mouse_multiplier, fps=args.fps, verbose=not args.no_header)
    game.run()
    # game.run_pygame()


if __name__ == "__main__":
    main()

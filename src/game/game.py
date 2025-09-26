"""Game loop for interactive BEV highway visualization and control.

FPS: Added built-in performance metrics collection (avg FPS, p95 FPS, inference-only p95, peak GPU memory).
"""
from typing import Tuple, Union

import numpy as np
import pygame
from PIL import Image
import time
import torch

from controls.action_processing import PIWMAction
from .dataset_env import DatasetEnv
from .play_env import PlayEnv


class Game:
    def __init__(
        self,
        play_env: Union[PlayEnv, DatasetEnv],
        size: Tuple[int, int],
        mouse_multiplier: int,
        fps: int,
        verbose: bool,
    ) -> None:
        self.env = play_env
        self.height, self.width = size
        self.mouse_multiplier = mouse_multiplier
        self.fps = fps
        self.verbose = verbose

        self.env.print_controls()
        print("\nControls:\n")
        print(" m  : switch control (human/replay)")
        print(" .  : pause/unpause")
        print(" e  : step-by-step (when paused)")
        print(" ⏎  : reset env")
        print("Esc : quit")
        print("\n")
        input("Press enter to start")

    def run(self) -> None:
        pygame.init()

        header_height = 110 if self.verbose else 0
        header_width = 540
        font_size = 16
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("mono", font_size)
        x_center, y_center = screen.get_rect().center
        x_header = x_center - header_width // 2
        y_header = y_center - self.height // 2 - header_height - 10
        header_rect = pygame.Rect(x_header, y_header, header_width, header_height)

        def clear_header():
            pygame.draw.rect(screen, pygame.Color("black"), header_rect)
            pygame.draw.rect(screen, pygame.Color("white"), header_rect, 1)

        def draw_text(text, idx_line, idx_column, num_cols):
            x_pos = 5 + idx_column * int(header_width // num_cols)
            y_pos = 5 + idx_line * font_size
            assert (0 <= x_pos <= header_width) and (0 <= y_pos <= header_height)
            screen.blit(font.render(text, True, pygame.Color("white")), (x_header + x_pos, y_header + y_pos))

        def draw_wasd(keys_state):
            if not keys_state or header_height == 0 or header_width == 0:
                return
            margin = 8
            gap = 6
            key_size_v = (header_height - 2 * margin - gap) / 2
            avail_w = header_width - 2 * margin
            key_size_h = (avail_w - 2 * gap) / 3
            key_size = int(max(8, min(key_size_v, key_size_h)))

            total_w = 3 * key_size + 2 * gap
            x0 = x_header + header_width - margin - total_w
            y0 = y_header + margin

            def rect(ix, iy):
                return pygame.Rect(x0 + ix * (key_size + gap), y0 + iy * (key_size + gap), key_size, key_size)

            rects = {
                "w": rect(1, 0),
                "a": rect(0, 1),
                "s": rect(1, 1),
                "d": rect(2, 1),
            }

            base_fill = pygame.Color(40, 40, 40)
            hi_fill = pygame.Color(46, 134, 255)
            outline = pygame.Color("white")

            for k, r in rects.items():
                pressed = bool(keys_state.get(k, False))
                pygame.draw.rect(screen, hi_fill if pressed else base_fill, r, border_radius=int(key_size * 0.2))
                pygame.draw.rect(screen, outline, r, 2, border_radius=int(key_size * 0.2))
                glyph = font.render(k.upper(), True, pygame.Color("white"))
                gx = r.x + (r.w - glyph.get_width()) // 2
                gy = r.y + (r.h - glyph.get_height()) // 2
                screen.blit(glyph, (gx, gy))

        def draw_obs(obs, obs_low_res=None):
            assert obs.ndim == 4 and obs.size(0) == 1
            img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            pygame_image = np.array(img.resize((self.width, self.height), resample=Image.BICUBIC)).transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(pygame_image)
            screen.blit(surface, (x_center - self.width // 2, y_center - self.height // 2))

            if obs_low_res is not None:
                assert obs_low_res.ndim == 4 and obs_low_res.size(0) == 1
                img = Image.fromarray(obs_low_res[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
                h = self.height * obs_low_res.size(2) // obs.size(2)
                w = self.width * obs_low_res.size(3) // obs.size(3)
                pygame_image = np.array(img.resize((w, h), resample=Image.BICUBIC)).transpose((1, 0, 2))
                surface = pygame.surfarray.make_surface(pygame_image)
                screen.blit(surface, (x_header + header_width - w - 5, y_header + 5 + font_size))
                # screen.blit(surface, (x_center - w // 2, y_center + self.height // 2))

        def reset():
            nonlocal obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click
            obs, info = self.env.reset()
            pygame.event.clear()
            do_reset = False
            ep_return = 0
            ep_length = 0
            keys_pressed = []
            l_click = r_click = False

        obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click = (None,) * 8

        reset()
        do_wait = False
        should_stop = False

        # Performance tracking (warm up then record)
        frame_durations_ms = []
        compute_durations_ms = []
        frame_index = 0
        warmup_frames = 5
        if torch and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        while not should_stop:
            do_one_step = False
            mouse_x, mouse_y = 0, 0
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    should_stop = True

                # Mouse motion ignored for 5-action control

                # Mouse buttons ignored for 5-action control

                if event.type == pygame.KEYDOWN:
                    keys_pressed.append(event.key)

                elif event.type == pygame.KEYUP and event.key in keys_pressed:
                    keys_pressed.remove(event.key)

                if event.type != pygame.KEYDOWN:
                    continue

                if event.key == pygame.K_RETURN:
                    do_reset = True

                if event.key == pygame.K_PERIOD:
                    do_wait = not do_wait
                    print("Game paused." if do_wait else "Game resumed.")

                if event.key == pygame.K_e:
                    do_one_step = True

                if event.key == pygame.K_m:
                    do_reset = self.env.next_mode()

                if event.key == pygame.K_UP:
                    do_reset = self.env.next_axis_1()

                if event.key == pygame.K_DOWN:
                    do_reset = self.env.prev_axis_1()

                if event.key == pygame.K_RIGHT:
                    do_reset = self.env.next_axis_2()

                if event.key == pygame.K_LEFT:
                    do_reset = self.env.prev_axis_2()

            if do_reset:
                reset()

            if do_wait and not do_one_step:
                continue

            t0 = time.perf_counter()
            infer_ms = None
            action_input = PIWMAction(keys_pressed)
            if torch and torch.cuda.is_available():  
                start_evt = torch.cuda.Event(enable_timing=True); end_evt = torch.cuda.Event(enable_timing=True)  # FPS
                start_evt.record()  # FPS
            next_obs, rew, end, trunc, info = self.env.step(action_input)
            if torch and torch.cuda.is_available():
                end_evt.record(); torch.cuda.synchronize()  # FPS
                infer_ms = start_evt.elapsed_time(end_evt)  # FPS

            ep_return += rew.item()
            ep_length += 1

            if self.verbose and info is not None:
                clear_header()
                assert isinstance(info, dict) and "header" in info
                header = info["header"]
                num_cols = len(header)
                for j, col in enumerate(header):
                    for i, row in enumerate(col):
                        draw_text(row, idx_line=i, idx_column=j, num_cols=num_cols)
                draw_wasd(info.get("keys_state"))

            draw_low_res = self.verbose and "obs_low_res" in info and self.width == 600
            if draw_low_res:
                draw_obs(obs, info["obs_low_res"])
                draw_text("  Pre-upsampling:", 0, 2, 3)
            else:
                draw_obs(obs, None)

            pygame.display.flip()  # update screen
            t_end = time.perf_counter()
            # Record metrics after warm-up
            if frame_index >= warmup_frames:
                frame_durations_ms.append((t_end - t0) * 1000.0)
                if infer_ms is not None:
                    compute_durations_ms.append(infer_ms)
            frame_index += 1

            clock.tick(self.fps)  # ensures game maintains the given frame rate

            if end or trunc:
                reset()

            else:
                obs = next_obs
                
        # Summarize performance metrics
        if frame_durations_ms:
            ms = np.array(frame_durations_ms, dtype=np.float64)
            avg_fps = 1000.0 / ms.mean()
            p95_fps = 1000.0 / np.percentile(ms, 95)
        else:
            avg_fps = float("nan"); p95_fps = float("nan")
        infer_p95_ms = float("nan")
        if compute_durations_ms:
            infer_p95_ms = float(np.percentile(np.array(compute_durations_ms, dtype=np.float64), 95))
        peak_mem_mb = 0.0
        if torch and torch.cuda.is_available():
            try:
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception:
                peak_mem_mb = 0.0
        print(f"FPS: avg={avg_fps:.2f} | p95={p95_fps:.2f} | infer_p95_ms={infer_p95_ms:.2f} | peak_mem={peak_mem_mb:.1f} MB | samples={len(frame_durations_ms)}")  # FPS
        pygame.quit()
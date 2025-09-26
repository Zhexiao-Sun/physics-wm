"""
51-dim action vector inherited from DIAMOND.
PIWM uses only the first 5 dims (W/A/S/D/Space). Extend the mapping if your dataset requires more actions.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pygame
import torch

from .keymap import PIWM_FORBIDDEN_COMBINATIONS, PIWM_KEYMAP


@dataclass
class PIWMAction:
    keys: List[int]

    def __post_init__(self) -> None:
        self.keys = filter_keys_pressed_forbidden(self.keys)

    @property
    def key_names(self) -> List[str]:
        return [pygame.key.name(key) for key in self.keys]


def print_action(action: PIWMAction) -> Tuple[str]:
    action_names = [PIWM_KEYMAP[k] for k in action.keys] if len(action.keys) > 0 else []
    action_names = [x for x in action_names if not x.startswith("camera_")]
    keys = " + ".join(action_names)
    return keys, "", ""


ACTION_SIZE = 51
# Fixed indices for 5 keys within 51-dim vector (must match dataset mapping)
# 0: w (up), 1: space (medium), 2: s (down), 3: d (right), 4: a (left)
KEY_TO_INDEX = {
    "w": 0,
    "space": 1,
    "s": 2,
    "d": 3,
    "a": 4,
}


def encode_action(action: PIWMAction, device: torch.device) -> torch.Tensor:
    vec = np.zeros(ACTION_SIZE, dtype=np.float32)
    # Priority order consistent with PlayEnv mapping
    key_names = action.key_names
    idx = 1  # default: space/medium
    if ("a" in key_names) or ("left" in key_names):
        idx = 4
    elif ("d" in key_names) or ("right" in key_names):
        idx = 3
    elif ("w" in key_names) or ("up" in key_names):
        idx = 0
    elif ("s" in key_names) or ("down" in key_names):
        idx = 2
    vec[idx] = 1.0
    return torch.tensor(vec, device=device, dtype=torch.float32)


def decode_action(y_preds: torch.Tensor) -> PIWMAction:
    y = y_preds.squeeze()
    first5 = y[:5]
    idx = int(np.argmax(first5))
    idx_to_key = {0: "w", 1: "space", 2: "s", 3: "d", 4: "a"}
    key = idx_to_key.get(idx, "space")
    return PIWMAction([pygame.key.key_code(key)])


def filter_keys_pressed_forbidden(
    keys_pressed: List[int],
    keymap: Dict[int, str] = PIWM_KEYMAP,
    forbidden_combinations: List[Set[str]] = PIWM_FORBIDDEN_COMBINATIONS,
) -> List[int]:
    keys = set()
    names = set()
    for key in keys_pressed:
        if key not in keymap:
            continue
        name = keymap[key]
        keys.add(key)
        names.add(name)
        for forbidden in forbidden_combinations:
            if forbidden.issubset(names):
                keys.remove(key)
                names.remove(name)
                break
    return list(filter(lambda key: key in keys, keys_pressed))



import pygame

PIWM_KEYMAP = {
    pygame.K_w: "left",
    pygame.K_s: "right",
    pygame.K_d: "faster",
    pygame.K_a: "slower",
    pygame.K_SPACE: "medium",
}


PIWM_FORBIDDEN_COMBINATIONS = [
    {"faster", "slower"},
    {"left", "right"},
    {"faster", "medium"},
    {"slower", "medium"},
    {"left", "medium"},
    {"right", "medium"},
    {"left", "faster"},
    {"right", "faster"},
    {"left", "slower"},
    {"right", "slower"},
]



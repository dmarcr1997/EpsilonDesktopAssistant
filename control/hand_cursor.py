from dataclasses import dataclass
from typing import Optional, Tuple
import time

import pyautogui

@dataclass
class CursorConfig:
    smooth_alpha: float = 0.35     # 0..1, higher = snappier
    left_click_thresh: float = 0.70
    left_release_thresh: float = 0.55
    right_click_thresh: float = 0.70
    right_release_thresh: float = 0.55
    click_cooldown_s: float = 0.25
    require_right_hand: bool = True

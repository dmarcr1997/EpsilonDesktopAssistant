from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class HandLandmarks:
    handedness: str
    score: float
    points: List[Tuple[int, int]]
    pinch_index: float   # thumb <-> index
    pinch_middle: float  # thumb <-> middle

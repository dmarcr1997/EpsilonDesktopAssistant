from typing import List, Tuple
import time
import cv2
import numpy as np
import mediapipe as mp
from .hand_types import HandLandmarks

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

class HandTracker:
    def __init__(self, model_path: str = "models/hand_landmarker.task", max_hands: int = 2):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(image, timestamp_ms)

        hands_out: List[HandLandmarks] = []
        if not result.hand_landmarks:
            return hands_out

        for i, landmarks in enumerate(result.hand_landmarks):
            pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]

            handed = result.handedness[i][0].category_name
            score = float(result.handedness[i][0].score)

            pinch_index = self._pinch_metric(pts, w, h, finger_tip_idx=8)
            pinch_middle = self._pinch_metric(pts, w, h, finger_tip_idx=12)

            hands_out.append(
                HandLandmarks(
                    handedness=handed,
                    score=score,
                    points=pts,
                    pinch_index=pinch_index,
                    pinch_middle=pinch_middle,
                )
            )

        return hands_out

    def _pinch_metric(self, pts, w, h, finger_tip_idx: int) -> float:
        # thumb tip = 4
        if len(pts) <= finger_tip_idx:
            return 0.0
        tx, ty = pts[4]
        fx, fy = pts[finger_tip_idx]
        dist = ((tx - fx) ** 2 + (ty - fy) ** 2) ** 0.5
        maxd = 0.25 * min(w, h)
        return float(max(0.0, min(1.0, 1.0 - dist / maxd)))

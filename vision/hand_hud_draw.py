from typing import Tuple, List
import cv2

def draw_hand_skeleton(frame, points: List[Tuple[int, int]], connections, color=(0, 255, 0)):
    # Draw connections (bones)
    if connections is not None:
        for c in connections:
            a, b = c.start, c.end
            ax, ay = points[a]
            bx, by = points[b]
            cv2.line(frame, (ax, ay), (bx, by), color, 2)

    if points is not None:
        # Draw joints (points)
        for (x, y) in points:
            cv2.circle(frame, (x, y), 3, color, -1)

def draw_label(frame, text: str, x: int, y: int, color=(0, 255, 0)):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

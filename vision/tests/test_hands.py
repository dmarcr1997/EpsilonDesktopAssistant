import cv2
from .camera import Camera
from .hand_tracker import HandTracker
from .hand_hud_draw import draw_hand_skeleton, draw_label

if __name__ == "__main__":
    cam = Camera(index=0, width=1280, height=720)
    tracker = HandTracker(max_hands=2)

    while True:
        frame = cam.read()
        if frame is None:
            continue
        hands = tracker.process(frame)
        for hand in hands:
            draw_hand_skeleton(frame, hand.points, tracker.connections)

            wx, wy = hand.points[0]

            draw_label(frame, f"{hand.handedness} {hand.score:.2f} pinch:{hand.pinch:.2f}", wx + 10, wy - 10)
            tx, ty = hand.points[4]
            ix, iy = hand.points[8]
            cv2.circle(frame, (tx, ty), 8, (255, 255, 255), 2)
            cv2.circle(frame, (ix, iy), 8, (255, 255, 255), 2)
        cv2.imshow("hands (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
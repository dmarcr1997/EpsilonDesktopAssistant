import cv2
from .camera import Camera

if __name__ == "__main__":
    cam = Camera()
    while True:
        frame = cam.read()
        if frame is None:
            continue
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
import cv2
from .camera import Camera
from .yoloe_engine import YOLOEEngine

if __name__ == "__main__":
    cam = Camera()
    yolo = YOLOEEngine()

    while True:
        frame = cam.read()
        if frame is None:
            continue
        
        results = yolo.detect(frame, conf=0.25)
        annotated = results[0].plot()

        cv2.imshow("yoloe detect", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()

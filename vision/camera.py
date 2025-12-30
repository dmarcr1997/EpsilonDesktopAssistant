import cv2

class Camera:
    def __init__(self, index:int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index={index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame
    
    def release(self):
        self.cap.release()
        
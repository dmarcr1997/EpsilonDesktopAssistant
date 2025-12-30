from ultralytics import YOLO

class YOLOEEngine:
    def __init__(self, model_path: str = "models/yoloe-11s-seg-pf.pt"):
        self.model = YOLO(model_path)
    
    def detect(self, frame, prompt: str | None = None, conf: float = 0.25):
        kwargs = {"conf": conf, "verbose": False}
        if prompt:
            kwargs["prompt"] = prompt
        results = self.model.predict(source=frame, **kwargs)
        return results
    
    def track(self, frame, conf: float = 0.25, persist: bool = True):
        results = self.model.track(source=frame, persist=persist, verbose=True, conf=conf)
        return results
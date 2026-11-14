from app.core.types import Detection

class NullDetector:
    name = "null"
    ready = True
    detail = "Detection intentionally disabled"
    def detect(self, frame): return []

class UnavailableDetector(NullDetector):
    ready = False
    def __init__(self, detail): self.name="ultralytics"; self.detail=detail

class UltralyticsDetector:
    def __init__(self, model_name):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: python -m pip install ultralytics") from exc
        self.model = YOLO(model_name)
        self.name="ultralytics"; self.ready=True; self.detail=f"Model loaded: {model_name}"
    def detect(self, frame):
        result = self.model(frame, verbose=False)[0]; names=result.names; found=[]
        for box in result.boxes:
            x1,y1,x2,y2=map(float,box.xyxy[0].tolist()); cls=int(box.cls[0]); conf=float(box.conf[0])
            found.append(Detection(names[cls],conf,(x1,y1,x2,y2)))
        return found

def make_detector(name, model):
    if name.lower() != "ultralytics": return NullDetector()
    try: return UltralyticsDetector(model)
    except Exception as exc: return UnavailableDetector(str(exc))

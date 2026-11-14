import asyncio, uuid
from datetime import datetime, timezone
from pathlib import Path
import cv2
from app.pipeline.detectors import make_detector
from app.pipeline.tracker import CentroidTracker
from app.pipeline.rules import RuleEngine
from app.services.camera import CameraReader

class SurveillanceEngine:
    def __init__(self, store, notifier, settings):
        self.store,self.notifier,self.settings=store,notifier,settings; self.readers={}; self.detector=make_detector(settings.detector,settings.yolo_model); self.tracker=CentroidTracker(); self.rules=RuleEngine(); self.task=None
    def start_camera(self,camera):
        reader=self.readers.get(camera["id"])
        if not reader: reader=CameraReader(camera["id"],camera["rtsp_url"]); self.readers[camera["id"]]=reader
        reader.start()
    def status(self,camera):
        r=self.readers.get(camera["id"])
        return {"id":camera["id"],"name":camera["name"],"enabled":camera["enabled"],"online":bool(r and r.online),"last_frame_at":r.last_frame_at if r else None,"error":r.last_error if r else "Camera reader is not running"}
    def detector_status(self):
        return {"name":self.detector.name,"ready":self.detector.ready,"detail":self.detector.detail}
    async def run(self):
        delay=1/max(self.settings.inference_fps,0.1)
        while True:
            for camera in self.store.cameras():
                reader=self.readers.get(camera["id"])
                if not reader or reader.frame is None: continue
                frame=reader.frame.copy(); detections=self.detector.detect(frame); tracks=self.tracker.update(detections)
                reader.annotated_frame=self.annotate(frame, tracks)
                for candidate in self.rules.evaluate(self.store.rules(camera["id"]),tracks,frame.shape[1],frame.shape[0]):
                    await self.record(camera["id"],candidate,frame)
            await asyncio.sleep(delay)
    @staticmethod
    def annotate(frame, tracks):
        for track in tracks:
            x1,y1,x2,y2=map(int,track.box); color=(75,210,115)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            label=f"{track.label} #{track.track_id} {track.confidence:.0%}"
            cv2.putText(frame,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
        return frame
    async def record(self,camera_id,candidate,frame):
        event_id=str(uuid.uuid4()); media=self.settings.data_dir/"media"; media.mkdir(parents=True,exist_ok=True); filename=f"{event_id}.jpg"; cv2.imwrite(str(media/filename),frame)
        event={"id":event_id,"camera_id":camera_id,"rule_id":candidate.rule_id,"event_type":candidate.event_type,"severity":candidate.severity,"track_id":candidate.track_id,"label":candidate.label,"details":candidate.details,"snapshot_path":filename,"occurred_at":datetime.now(timezone.utc).isoformat()}
        self.store.add_event(event)
        try: await self.notifier.send(event)
        except Exception: pass  # delivery retries belong in a durable queue in multi-node deployments

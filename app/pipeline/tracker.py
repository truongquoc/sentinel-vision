from datetime import datetime, timezone
from app.core.types import TrackedObject

class CentroidTracker:
    """Lightweight tracker; replaceable with ByteTrack/DeepSORT without touching rules."""
    def __init__(self, max_distance=80, max_missed=12): self.tracks={}; self.next_id=1; self.max_distance=max_distance; self.max_missed=max_missed
    @staticmethod
    def center(box): return ((box[0]+box[2])/2,(box[1]+box[3])/2)
    def update(self, detections):
        now=datetime.now(timezone.utc); available=set(self.tracks)
        for d in detections:
            cx,cy=self.center(d.box); choices=[(tid,(self.center(t.box)[0]-cx)**2+(self.center(t.box)[1]-cy)**2) for tid,t in self.tracks.items() if tid in available and t.label==d.label]
            tid=min(choices,key=lambda x:x[1])[0] if choices and min(x[1] for x in choices)<self.max_distance**2 else None
            if tid is None:
                tid=self.next_id; self.next_id+=1; self.tracks[tid]=TrackedObject(tid,d.label,d.confidence,d.box,0,now,now)
            else:
                old=self.tracks[tid]; self.tracks[tid]=TrackedObject(tid,d.label,d.confidence,d.box,old.age+1,old.first_seen,now); available.remove(tid)
        for tid in list(available):
            t=self.tracks[tid]; self.tracks[tid]=TrackedObject(t.track_id,t.label,t.confidence,t.box,t.age,t.first_seen,t.last_seen)
            if (now-t.last_seen).total_seconds()>self.max_missed: del self.tracks[tid]
        return list(self.tracks.values())

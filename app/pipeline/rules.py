from datetime import datetime, timezone
from app.core.types import EventCandidate

def inside(point, polygon):
    x,y=point; hit=False
    for i in range(len(polygon)):
        x1,y1=polygon[i]; x2,y2=polygon[(i+1)%len(polygon)]
        if (y1>y)!=(y2>y) and x < (x2-x1)*(y-y1)/(y2-y1+1e-12)+x1: hit=not hit
    return hit

class RuleEngine:
    def __init__(self): self.emitted={}
    def evaluate(self, rules, tracks, width, height):
        now=datetime.now(timezone.utc); out=[]
        for rule in rules:
            if not rule["enabled"]: continue
            cfg=rule["config"]
            for t in tracks:
                if cfg.get("classes") and t.label not in cfg["classes"]: continue
                cx=(t.box[0]+t.box[2])/2/width; cy=(t.box[1]+t.box[3])/2/height
                triggered=(rule["kind"]=="zone" and inside((cx,cy),cfg.get("polygon",[]))) or (rule["kind"]=="loiter" and (now-t.first_seen).total_seconds()>=cfg.get("seconds",30))
                key=(rule["id"],t.track_id)
                if triggered and (key not in self.emitted or (now-self.emitted[key]).total_seconds()>=cfg.get("cooldown_seconds",30)):
                    self.emitted[key]=now; out.append(EventCandidate(rule["id"],"intrusion" if rule["kind"]=="zone" else "loitering",cfg.get("severity","medium"),t.track_id,t.label,{"box":t.box,"confidence":t.confidence,"rule_name":rule["name"]}))
        return out

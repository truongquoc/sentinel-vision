from app.core.types import Detection
from app.pipeline.rules import RuleEngine
from app.pipeline.tracker import CentroidTracker

def test_zone_rule_creates_event_for_person():
    tracker=CentroidTracker()
    track=tracker.update([Detection('person', .94, (100,100,200,300))])[0]
    rules=[{'id':'front','name':'Front area','kind':'zone','enabled':True,'config':{'classes':['person'],'polygon':[[0,0],[1,0],[1,1],[0,1]]}}]
    events=RuleEngine().evaluate(rules,[track],640,480)
    assert len(events)==1 and events[0].event_type=='intrusion'

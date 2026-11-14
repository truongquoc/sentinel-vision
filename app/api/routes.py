import asyncio, uuid
from pathlib import Path
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

router=APIRouter(prefix="/api")
class RuleIn(BaseModel):
    camera_id:str="front-door"; name:str; kind:Literal["zone","loiter"]; config:dict=Field(default_factory=dict); enabled:bool=True

def deps(request:Request): return request.app.state.store,request.app.state.engine
@router.get('/health')
def health(request:Request):
    store,engine=deps(request); return {"status":"ok","detector":engine.detector_status(),"cameras":[engine.status(c) for c in store.cameras()]}
@router.get('/cameras')
def cameras(request:Request):
    store,engine=deps(request); return [engine.status(c) for c in store.cameras()]
@router.post('/cameras/{camera_id}/start')
def start(camera_id:str,request:Request):
    store,engine=deps(request); camera=next((c for c in store.cameras() if c['id']==camera_id),None)
    if not camera: raise HTTPException(404,'Camera not found')
    engine.start_camera(camera); return engine.status(camera)
@router.get('/cameras/{camera_id}/mjpeg')
async def stream(camera_id:str,request:Request):
    _,engine=deps(request); reader=engine.readers.get(camera_id)
    if not reader: raise HTTPException(409,'Camera has not been started')
    return StreamingResponse(reader.mjpeg(),media_type='multipart/x-mixed-replace; boundary=frame')
@router.get('/rules')
def rules(request:Request,camera_id: Optional[str] = None): return deps(request)[0].rules(camera_id)
@router.post('/rules',status_code=201)
def add_rule(payload:RuleIn,request:Request):
    store,_=deps(request); rule=payload.model_dump()|{'id':str(uuid.uuid4())}; store.add_rule(rule); return rule
@router.get('/events')
def events(request:Request,camera_id: Optional[str] = None, limit: int = 100): return deps(request)[0].events(camera_id,min(max(limit,1),500))
@router.post('/events/{event_id}/ack')
def ack(event_id:str,request:Request):
    if not deps(request)[0].acknowledge(event_id): raise HTTPException(404,'Event not found')
    return {'ok':True}
@router.get('/events/{event_id}/snapshot')
def snapshot(event_id:str,request:Request):
    store,_=deps(request); event=next((e for e in store.events(limit=500) if e['id']==event_id),None)
    if not event or not event['snapshot_path']: raise HTTPException(404,'Snapshot not found')
    path=Path(request.app.state.settings.data_dir)/'media'/event['snapshot_path']
    if not path.exists(): raise HTTPException(404,'Snapshot expired')
    return FileResponse(path)

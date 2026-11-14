import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.services.store import Store
from app.services.notifier import WebhookNotifier
from app.services.engine import SurveillanceEngine
from app.api.routes import router

@asynccontextmanager
async def lifespan(app):
    settings.data_dir.mkdir(parents=True,exist_ok=True)
    store=Store(settings.data_dir/'sentinel.db'); store.seed_camera('front-door',settings.camera_name,settings.camera_rtsp_url)
    store.seed_rule({'id':'front-door-person-zone','camera_id':'front-door','name':'Person in monitored area','kind':'zone','config':{'classes':['person'],'polygon':[[0,0],[1,0],[1,1],[0,1]],'severity':'medium','cooldown_seconds':30},'enabled':True})
    engine=SurveillanceEngine(store,WebhookNotifier(settings.webhook_url,settings.webhook_secret),settings)
    app.state.store,app.state.engine,app.state.settings=store,engine,settings
    if settings.camera_rtsp_url: engine.start_camera(store.cameras()[0])
    engine.task=asyncio.create_task(engine.run())
    yield
    engine.task.cancel()
    for reader in engine.readers.values(): reader.stop()

app=FastAPI(title='Sentinel Vision',version='0.1.0',lifespan=lifespan)
app.include_router(router)
app.mount('/',StaticFiles(directory='app/static',html=True),name='dashboard')

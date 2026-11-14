import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import Settings, settings
from app.core.logging import configure_logging, get_logger
from app.services.engine import SurveillanceEngine, TrackerFactory
from app.services.notifier import WebhookNotifier
from app.services.store import Store

log = get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
RETENTION_INTERVAL_SECONDS = 3600


async def retention_loop(engine: SurveillanceEngine) -> None:
    while True:
        removed = await asyncio.to_thread(engine.purge_old_media)
        if removed:
            log.info(f"Retention removed {removed} media files", retention_days=engine.settings.event_retention_days)
        await asyncio.sleep(RETENTION_INTERVAL_SECONDS)


def create_app(config: Settings = settings, tracker_factory: TrackerFactory | None = None, autostart: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(config.log_format)
        store = Store(config.data_dir / "sentinel.db")
        store.upsert_camera(config.camera_id, config.camera_name, config.camera_rtsp_url)
        notifier = WebhookNotifier(config.webhook_url, config.webhook_secret)
        notifier.start()
        engine = SurveillanceEngine(store, notifier, config, tracker_factory)
        app.state.store, app.state.engine, app.state.settings = store, engine, config
        if autostart and config.camera_rtsp_url:
            engine.start_camera(config.camera_id)
        retention = asyncio.create_task(retention_loop(engine))
        yield
        retention.cancel()
        engine.stop()
        notifier.stop()
        store.close()

    app = FastAPI(title="Sentinel Vision", version="0.2.0", lifespan=lifespan)
    app.include_router(router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
    return app


app = create_app()

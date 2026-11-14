"""Dependency-free configuration loader for local edge deployments."""
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CLASSES = "person,bicycle,car,motorcycle,bus,truck,cat,dog"


def load_dotenv(path: Path | None = None) -> None:
    """Load simple KEY=value entries without overriding explicit environment variables."""
    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    camera_id: str = field(default_factory=lambda: _env("CAMERA_ID", "front-door"))
    camera_rtsp_url: str = field(default_factory=lambda: _env("CAMERA_RTSP_URL", ""))
    camera_name: str = field(default_factory=lambda: _env("CAMERA_NAME", "Front entrance"))
    detector: str = field(default_factory=lambda: _env("DETECTOR", "ultralytics"))
    yolo_model: str = field(default_factory=lambda: _env("YOLO_MODEL", "yolo11s.pt"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    detect_classes: tuple[str, ...] = field(default_factory=lambda: _csv(_env("DETECT_CLASSES", DEFAULT_CLASSES)))
    min_confidence: float = field(default_factory=lambda: float(_env("MIN_CONFIDENCE", "0.35")))
    inference_fps: float = field(default_factory=lambda: float(_env("INFERENCE_FPS", "8")))
    stream_timeout_seconds: float = field(default_factory=lambda: float(_env("STREAM_TIMEOUT_SECONDS", "8")))
    webhook_url: str = field(default_factory=lambda: _env("WEBHOOK_URL", ""))
    webhook_secret: str = field(default_factory=lambda: _env("WEBHOOK_SECRET", ""))
    event_retention_days: int = field(default_factory=lambda: int(_env("EVENT_RETENTION_DAYS", "14")))
    log_format: str = field(default_factory=lambda: _env("LOG_FORMAT", "console"))


settings = Settings()

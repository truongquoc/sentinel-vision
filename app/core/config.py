"""Dependency-free configuration loader for local edge deployments."""
import os
from dataclasses import dataclass, field
from pathlib import Path

def load_dotenv(path: Path = None):
    """Load simple KEY=value entries without overriding explicit environment variables."""
    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.exists(): return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_dotenv()

@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    camera_rtsp_url: str = field(default_factory=lambda: os.getenv("CAMERA_RTSP_URL", ""))
    camera_name: str = field(default_factory=lambda: os.getenv("CAMERA_NAME", "Front entrance"))
    detector: str = field(default_factory=lambda: os.getenv("DETECTOR", "null"))
    yolo_model: str = field(default_factory=lambda: os.getenv("YOLO_MODEL", "yolo11n.pt"))
    inference_fps: float = field(default_factory=lambda: float(os.getenv("INFERENCE_FPS", "4")))
    webhook_url: str = field(default_factory=lambda: os.getenv("WEBHOOK_URL", ""))
    webhook_secret: str = field(default_factory=lambda: os.getenv("WEBHOOK_SECRET", ""))
    event_cooldown_seconds: int = field(default_factory=lambda: int(os.getenv("EVENT_COOLDOWN_SECONDS", "30")))

settings = Settings()

# Sentinel Vision

Self-hosted AI detection for RTSP cameras, built first for the Imou Bullet 2C. Everything runs on your own
machine: no cloud, no subscription.

**Phase 1 (this release): the detector.** Detects people, vehicles and animals, follows each one with a
stable ID, and records an event when an object enters a zone you draw, stays too long, or leaves.

## Quick start (macOS / Apple Silicon)

```bash
cp .env.example .env            # then set CAMERA_RTSP_URL
uv venv --python 3.12 .venv     # or python3.12 -m venv .venv
uv pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, wait for **ONLINE**, then click **+ Draw zone** and click the corners of the area
you care about on the live image.

Run directly on macOS rather than in Docker: Docker on a Mac cannot use the Apple GPU, so inference would fall
back to the CPU. The Dockerfile is for Linux hosts.

### Connecting the Bullet 2C

1. In the Imou app, make sure the camera is on your local network and note its IP (reserve it in your router).
2. The RTSP user is `admin`, the password is the **safety code** printed on the camera label.
3. Use the sub-stream for detection:
   `rtsp://admin:<SAFETY_CODE>@<IP>:554/cam/realmonitor?channel=1&subtype=1`
4. Check it plays first: `ffplay -rtsp_transport tcp "<url>"` or VLC → Open Network.

Credentials live only in `.env` (gitignored). They are redacted from logs and never returned by the API.

## How it works

```
CameraReader thread      newest frame only, open/read timeouts, exponential reconnect
      │
CameraPipeline thread    YOLO11 + ByteTrack (Apple GPU via mps)  →  RuleEngine  →  SQLite + snapshots
      │                                                                         └→ webhook thread (retries)
FastAPI                  REST API, MJPEG live view, dashboard
```

- **Tracking:** ByteTrack keeps IDs stable through crossings and short occlusions; one tracker per camera.
- **Zones:** polygons in normalised 0..1 coordinates, so they survive resolution changes. An object is in
  a zone when its *feet* (bottom-centre of its box) are inside, not its torso.
- **Events:** a per-object state machine per zone:
  - `zone_enter` after `min_frames` consecutive frames inside (default 2), which filters edge jitter.
  - `loitering` once per visit after `loiter_seconds`.
  - `zone_exit` after the object is gone for `exit_grace_seconds` (default 2s). Brief occlusions don't
    create exit/enter pairs.
- **Media:** each event stores an annotated full-frame snapshot plus a tight crop of the object. The crop is
  what a vision-language model will describe in phase 2.
- **Retention:** events and media older than `EVENT_RETENTION_DAYS` are purged hourly.

Measured on an M4 with `yolo11s.pt` at 640×480: about 38 ms inference, steady 8 fps (the configured cap).

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | camera, detector, pipeline fps/latency, zone occupancy, webhook stats |
| POST | `/api/cameras/{id}/start` | start reader + pipeline |
| GET | `/api/cameras/{id}/mjpeg` · `/snapshot` | annotated live view |
| GET/POST | `/api/rules` | list / create zone rules |
| PUT/DELETE | `/api/rules/{id}` | update / delete |
| GET | `/api/events?event_type=&label=&unacknowledged=&limit=` | event ledger |
| GET | `/api/events/{id}` · `/snapshot` · `/thumbnail` | event detail & media |
| POST | `/api/events/{id}/ack` | acknowledge |

Zone rule body:

```json
{"camera_id": "front-door", "name": "Front gate",
 "config": {"polygon": [[0.1,0.5],[0.9,0.5],[0.9,1],[0.1,1]], "classes": ["person"],
            "events": ["enter", "loiter"], "loiter_seconds": 30, "severity": "medium"}}
```

Webhooks are POSTed as JSON with `X-Sentinel-Signature: sha256=<hmac>` when `WEBHOOK_SECRET` is set.
They are retried 4 times with backoff, and failures are logged and counted in `/api/health`.

## Tests

```bash
.venv/bin/python -m pytest
```

To try it without a camera, set `CAMERA_RTSP_URL` to a local video file. It loops at its native frame rate.

## Not yet (next phases)

- AI descriptions of what a person in a zone is doing (local VLM via Ollama)
- Phone push notifications
- go2rtc for full-resolution, low-latency live view (H.265 main stream)
- Continuous recording / clips
- Authentication. Keep this on your LAN, or reach it remotely through Tailscale. Do not port-forward it.

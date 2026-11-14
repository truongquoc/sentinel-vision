# Sentinel Vision

Production-oriented, edge-first surveillance platform for RTSP cameras (including Imou Bullet 2C once RTSP is enabled). It is intentionally modular: camera ingestion, detection, tracking, rules, persistence, delivery, and UI are separated behind small interfaces.

## Start

```bash
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000`. Set `CAMERA_RTSP_URL` to the RTSP URL exposed by your camera; never commit credentials. The service remains usable without a camera: it reports an offline source and accepts a synthetic health check.

For the Bullet 2C, enable RTSP/ONVIF in the Imou app or device settings, reserve the camera IP, and use a least-privilege local account. RTSP paths vary by firmware, so obtain the exact path from the device/ONVIF discovery rather than guessing it.

## Architecture

`RTSP reader -> bounded latest-frame buffer -> detector -> tracker -> rule engine -> SQLite + snapshot -> notification adapters -> REST/MJPEG dashboard`

The default detector is `null` (safe for API/UI deployment). Set `DETECTOR=ultralytics` and install `ultralytics` to use a YOLO model. Rules can be created through the API and use normalized polygons (`0..1`) so they work across resolutions.

## Operating notes

- Each reader has exponential reconnect backoff and a one-frame buffer: surveillance should favor current frames over latency.
- SQLite is ideal for a single edge node. Move the repository implementation to PostgreSQL and media to S3/MinIO before multi-node deployment.
- Webhook delivery is retryable and signed with an optional HMAC secret. Put API auth/TLS behind a reverse proxy before exposing this outside a trusted LAN.

## API highlights

- `GET /api/health`, `GET /api/cameras`, `POST /api/cameras/{id}/start`
- `GET/POST /api/rules`, `GET /api/events`, `POST /api/events/{id}/ack`
- `GET /api/cameras/{id}/mjpeg`


"""SQLite persistence. One shared connection guarded by a lock: camera threads and API handlers both write."""
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (id TEXT PRIMARY KEY, name TEXT NOT NULL, rtsp_url TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rules (id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL, config TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, rule_id TEXT, event_type TEXT NOT NULL, severity TEXT NOT NULL, track_id INTEGER, label TEXT, details TEXT NOT NULL, snapshot_path TEXT, occurred_at TEXT NOT NULL, acknowledged_at TEXT);
CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events(camera_id, occurred_at DESC);
"""

MIGRATIONS = {
    "events": {"thumbnail_path": "TEXT", "description_status": "TEXT", "description": "TEXT"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate(self) -> None:
        for table, columns in MIGRATIONS.items():
            existing = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            for column, ddl in columns.items():
                if column not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        self._conn.commit()

    def _write(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cursor = self._conn.execute(sql, args)
            self._conn.commit()
            return cursor.rowcount

    def _read(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    # cameras -----------------------------------------------------------------
    def upsert_camera(self, camera_id: str, name: str, url: str) -> None:
        """The .env is the source of truth for the camera URL, so restarts pick up edits."""
        self._write(
            "INSERT INTO cameras VALUES (?, ?, ?, 1, ?) ON CONFLICT(id) DO UPDATE SET name=excluded.name, rtsp_url=excluded.rtsp_url",
            (camera_id, name, url, utc_now()),
        )

    def cameras(self) -> list[dict[str, Any]]:
        return [dict(r) | {"enabled": bool(r["enabled"])} for r in self._read("SELECT * FROM cameras ORDER BY created_at")]

    def camera(self, camera_id: str) -> dict[str, Any] | None:
        return next((c for c in self.cameras() if c["id"] == camera_id), None)

    # rules -------------------------------------------------------------------
    def rules(self, camera_id: str | None = None) -> list[dict[str, Any]]:
        if camera_id is None:
            rows = self._read("SELECT * FROM rules ORDER BY created_at")
        else:
            rows = self._read("SELECT * FROM rules WHERE camera_id=? ORDER BY created_at", (camera_id,))
        return [self._rule(r) for r in rows]

    def rule(self, rule_id: str) -> dict[str, Any] | None:
        rows = self._read("SELECT * FROM rules WHERE id=?", (rule_id,))
        if not rows:
            return None
        return self._rule(rows[0])

    def add_rule(self, rule: dict[str, Any]) -> None:
        self._write(
            "INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rule["id"], rule["camera_id"], rule["name"], rule["kind"], json.dumps(rule["config"]), int(rule.get("enabled", True)), utc_now()),
        )

    def update_rule(self, rule_id: str, name: str, config: dict[str, Any], enabled: bool) -> bool:
        return self._write("UPDATE rules SET name=?, config=?, enabled=? WHERE id=?", (name, json.dumps(config), int(enabled), rule_id)) > 0

    def delete_rule(self, rule_id: str) -> bool:
        return self._write("DELETE FROM rules WHERE id=?", (rule_id,)) > 0

    @staticmethod
    def _rule(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row) | {"config": json.loads(row["config"]), "enabled": bool(row["enabled"])}

    # events ------------------------------------------------------------------
    def add_event(self, e: dict[str, Any]) -> None:
        self._write(
            "INSERT INTO events (id, camera_id, rule_id, event_type, severity, track_id, label, details, snapshot_path, "
            "thumbnail_path, occurred_at, description_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (e["id"], e["camera_id"], e["rule_id"], e["event_type"], e["severity"], e["track_id"], e["label"],
             json.dumps(e["details"]), e.get("snapshot_path"), e.get("thumbnail_path"), e["occurred_at"], e.get("description_status")),
        )

    def set_description(self, event_id: str, status: str, payload: dict[str, Any] | None) -> bool:
        """status: pending | done | failed | skipped. payload is the description, or {"error": ...}."""
        body = json.dumps(payload) if payload is not None else None
        return self._write("UPDATE events SET description_status=?, description=? WHERE id=?", (status, body, event_id)) > 0

    def events(
        self,
        camera_id: str | None = None,
        event_type: str | None = None,
        label: str | None = None,
        unacknowledged: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        # An activity update exists only to carry a description; without one it is noise.
        clauses, args = ["NOT (event_type = 'activity' AND COALESCE(description_status, '') IN ('skipped', 'failed'))"], []
        for column, value in (("camera_id", camera_id), ("event_type", event_type), ("label", label)):
            if value:
                clauses.append(f"{column}=?")
                args.append(value)
        if unacknowledged:
            clauses.append("acknowledged_at IS NULL")
        where = f" WHERE {' AND '.join(clauses)}"
        rows = self._read(f"SELECT * FROM events{where} ORDER BY occurred_at DESC LIMIT ?", (*args, limit))
        return [self._event(r) for r in rows]

    def event(self, event_id: str) -> dict[str, Any] | None:
        rows = self._read("SELECT * FROM events WHERE id=?", (event_id,))
        if not rows:
            return None
        return self._event(rows[0])

    def count_events_since(self, since_iso: str) -> int:
        return self._read("SELECT COUNT(*) AS n FROM events WHERE occurred_at >= ?", (since_iso,))[0]["n"]

    def acknowledge(self, event_id: str) -> bool:
        return self._write("UPDATE events SET acknowledged_at=? WHERE id=? AND acknowledged_at IS NULL", (utc_now(), event_id)) > 0

    def purge_events_older_than(self, days: int) -> list[str]:
        """Delete old events; returns media filenames the caller should remove from disk."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._read("SELECT snapshot_path, thumbnail_path FROM events WHERE occurred_at < ?", (cutoff,))
        self._write("DELETE FROM events WHERE occurred_at < ?", (cutoff,))
        return [path for row in rows for path in (row["snapshot_path"], row["thumbnail_path"]) if path]

    @staticmethod
    def _event(row: sqlite3.Row) -> dict[str, Any]:
        description = json.loads(row["description"]) if row["description"] else None
        return dict(row) | {"details": json.loads(row["details"]), "description": description}

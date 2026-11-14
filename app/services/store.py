import json, sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); self.path = path
        self.initialize()
    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path); c.row_factory = sqlite3.Row
        try: yield c; c.commit()
        finally: c.close()
    def initialize(self):
        with self.conn() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS cameras (id TEXT PRIMARY KEY, name TEXT NOT NULL, rtsp_url TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS rules (id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL, config TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, rule_id TEXT, event_type TEXT NOT NULL, severity TEXT NOT NULL, track_id INTEGER, label TEXT, details TEXT NOT NULL, snapshot_path TEXT, occurred_at TEXT NOT NULL, acknowledged_at TEXT);
            CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events(camera_id, occurred_at DESC);
            ''')
    def seed_camera(self, camera_id, name, url):
        with self.conn() as c: c.execute("INSERT OR IGNORE INTO cameras VALUES (?, ?, ?, 1, ?)", (camera_id,name,url,self.now()))
    def seed_rule(self, rule):
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO rules VALUES (?, ?, ?, ?, ?, ?, ?)", (rule["id"],rule["camera_id"],rule["name"],rule["kind"],json.dumps(rule["config"]),rule.get("enabled",True),self.now()))
    def cameras(self):
        with self.conn() as c: return [dict(r) | {"enabled": bool(r["enabled"])} for r in c.execute("SELECT * FROM cameras")]
    def rules(self, camera_id=None):
        q="SELECT * FROM rules"; args=()
        if camera_id: q += " WHERE camera_id=?"; args=(camera_id,)
        with self.conn() as c:
            return [dict(r) | {"config":json.loads(r["config"]),"enabled":bool(r["enabled"])} for r in c.execute(q,args)]
    def add_rule(self, rule):
        with self.conn() as c: c.execute("INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?)",(rule["id"],rule["camera_id"],rule["name"],rule["kind"],json.dumps(rule["config"]),rule.get("enabled",True),self.now()))
    def add_event(self, e):
        with self.conn() as c: c.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",(e["id"],e["camera_id"],e["rule_id"],e["event_type"],e["severity"],e["track_id"],e["label"],json.dumps(e["details"]),e.get("snapshot_path"),e["occurred_at"]))
    def events(self, camera_id=None, limit=100):
        q="SELECT * FROM events"; args=[]
        if camera_id: q += " WHERE camera_id=?"; args.append(camera_id)
        q += " ORDER BY occurred_at DESC LIMIT ?"; args.append(limit)
        with self.conn() as c: return [dict(r) | {"details":json.loads(r["details"])} for r in c.execute(q,args)]
    def acknowledge(self, event_id):
        with self.conn() as c: return c.execute("UPDATE events SET acknowledged_at=? WHERE id=?",(self.now(),event_id)).rowcount > 0
    @staticmethod
    def now(): return datetime.now(timezone.utc).isoformat()

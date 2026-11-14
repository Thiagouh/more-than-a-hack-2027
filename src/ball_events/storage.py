"""Disk-backed run storage and interval-indexed hints; no external services."""
from collections import OrderedDict
import json
import sqlite3

from .events import delivery_key, validate_event, validate_parent
from .contract import InputError


class TrackRegistry:
    def __init__(self, connection):
        self.db = connection
        self.db.execute('CREATE TABLE IF NOT EXISTS tracks (id TEXT PRIMARY KEY, team TEXT NOT NULL)')
        self.recent = OrderedDict()

    def setdefault(self, key, value):
        if key in self.recent:
            self.recent.move_to_end(key)
            return self.recent[key]
        row = self.db.execute('SELECT team FROM tracks WHERE id=?', (key,)).fetchone()
        actual = row[0] if row else value
        if row is None:
            self.db.execute('INSERT INTO tracks VALUES (?,?)', (key, value))
        self.recent[key] = actual
        if len(self.recent) > 2048:
            self.recent.popitem(last=False)
        return actual

    def __len__(self):
        return self.db.execute('SELECT count(*) FROM tracks').fetchone()[0]


class HintIndex:
    def __init__(self, hints):
        # Balanced interval tree augmented with max end; arbitrary query order is supported.
        self.hints = sorted(hints, key=lambda h: (h['start_frame'], h['hint_id']))
        def build(lo, hi):
            if lo >= hi:
                return None
            mid = (lo+hi)//2
            left, right = build(lo, mid), build(mid+1, hi)
            h = self.hints[mid]
            return (h, max(h['end_frame'], left[1] if left else 0, right[1] if right else 0), left, right)
        self.root = build(0, len(self.hints))

    def at(self, frame):
        result = []
        def visit(node):
            if node is None or node[1] <= frame:
                return
            h, _, left, right = node
            visit(left)
            if h['start_frame'] <= frame:
                if frame < h['end_frame']:
                    result.append(h)
                visit(right)
        visit(self.root)
        return sorted(result, key=lambda h: h['hint_id'])


class RunStore:
    def __init__(self, path, meta):
        self.db = sqlite3.connect(path)
        self.meta = meta
        self.db.execute('CREATE TABLE events (id TEXT PRIMARY KEY, available INTEGER, confirmed INTEGER, frame INTEGER, child INTEGER, body TEXT)')
        self.db.execute('CREATE INDEX delivery ON events(available,confirmed,frame,child,id)')
        self.db.execute('CREATE TABLE sources (start INTEGER PRIMARY KEY, end INTEGER, source TEXT)')
        self.last_source = None
        self.registry = TrackRegistry(self.db)

    def add_event(self, e):
        validate_event(e, self.meta)
        if e['parent_event_id']:
            row = self.db.execute('SELECT body FROM events WHERE id=?', (e['parent_event_id'],)).fetchone()
            validate_parent(e, json.loads(row[0]) if row else None)
        try:
            self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?)',
                            (e['event_id'], e['available_at_frame'], e['confirmed_at_frame'], e['event_frame'],
                             bool(e['parent_event_id']), json.dumps(e, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))))
        except sqlite3.IntegrityError:
            raise InputError('Duplicate event ID') from None

    def source(self, frame, source):
        if self.last_source and self.last_source[2] == source:
            self.last_source[1] = frame+1
        else:
            self.flush_source()
            self.last_source = [frame, frame+1, source]

    def flush_source(self):
        if self.last_source:
            self.db.execute('INSERT INTO sources VALUES (?,?,?)', self.last_source)
            self.last_source = None

    def events(self):
        for row in self.db.execute('SELECT body FROM events ORDER BY available,confirmed,frame,child,id'):
            yield json.loads(row[0])

    def runs(self):
        self.flush_source()
        for start, end, source in self.db.execute('SELECT start,end,source FROM sources ORDER BY start'):
            yield source, start, end

    def close(self):
        self.db.close()

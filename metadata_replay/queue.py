"""Portable entries and a thread-safe queue, independent of Gradio and Forge."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import random
import threading
import time
from dataclasses import asdict, dataclass, field

SCHEMA = "forge-metadata-replay"
MAX_ENTRIES = 2000
MAX_TEXT = 1_000_000


@dataclass
class Entry:
    id: str
    name: str
    infotext: str
    parameters: dict
    source_preview: str = ""
    generated_preview: str = ""
    status: str = "pending"
    batches: int = 0
    error: str = ""


def identity(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def preview_ok(value):
    return isinstance(value, str) and len(value) <= 100_000 and (
        not value or value.startswith("data:image/jpeg;base64,"))


def decode_document(data: bytes) -> list[Entry]:
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("JSON must be smaller than 32 MiB.")
    def reject_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")
    doc = json.loads(data, parse_constant=reject_constant)
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("version") != 1:
        raise ValueError("Choose a Metadata Replay version 1 JSON export.")
    rows = doc.get("entries")
    if not isinstance(rows, list) or len(rows) > MAX_ENTRIES:
        raise ValueError(f"JSON must contain at most {MAX_ENTRIES} entries.")
    entries = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid entry.")
        key = row.get("id", "")
        params = row.get("parameters")
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("An entry has an invalid file fingerprint.")
        if not isinstance(params, dict) or "Prompt" not in params:
            raise ValueError("An entry has no prompt metadata.")
        if len(json.dumps(params)) > MAX_TEXT or any(
            not isinstance(k, str) or not isinstance(v, (str, int, float, bool, list, type(None)))
            for k, v in params.items()
        ):
            raise ValueError("Invalid or oversized parameters.")
        if not all(isinstance(params.get(k, ""), str) for k in ("Prompt", "Negative prompt")):
            raise ValueError("Prompts must be text.")
        name, infotext = row.get("name", "Imported image"), row.get("infotext", "")
        thumb = row.get("generated_preview", "")
        if not isinstance(name, str) or len(name) > 1024 or not isinstance(infotext, str) or len(infotext) > MAX_TEXT:
            raise ValueError("Invalid entry name or infotext.")
        if not preview_ok(thumb):
            raise ValueError("Invalid preview.")
        entries.append(Entry(key, name, infotext, copy.deepcopy(params), generated_preview=thumb))
    return entries


class ReplayQueue:
    def __init__(self):
        self.condition = threading.Condition(threading.RLock())
        self.entries: list[Entry] = []
        self.revision = 0
        self.active = False
        self.paused = False
        self.waiting = False
        self.current = None
        self.cursor = None
        self.pending = None
        self.batch = 0
        self.batch_count = 1
        self.message = "Add images or import a queue JSON, then use txt2img Generate."
        self.touched = time.monotonic()

    def changed(self):
        self.revision += 1
        self.touched = time.monotonic()
        self.condition.notify_all()

    def find(self, key):
        return next((e for e in self.entries if e.id == key), None)

    def adjacent(self, key, offset):
        index = next((i for i, e in enumerate(self.entries) if e.id == key), -1)
        target = index + offset
        return self.entries[target].id if 0 <= target < len(self.entries) else None

    def merge(self, entries, replace=False):
        with self.condition:
            if replace and self.active:
                raise ValueError("Stop generation before replacing the queue. Merge remains available.")
            incoming = list(entries)
            known = {} if replace else {e.id: e for e in self.entries}
            additions = []
            duplicates = 0
            for entry in incoming:
                if entry.id in known:
                    # Restore a source preview when JSON was imported before its image.
                    if entry.source_preview:
                        known[entry.id].source_preview = entry.source_preview
                    duplicates += 1
                else:
                    known[entry.id] = entry
                    additions.append(entry)
            if len(known) > MAX_ENTRIES:
                raise ValueError(f"Queue limit: {MAX_ENTRIES} images.")
            if replace:
                self.entries = additions
                self.cursor = self.current = None
                self.batch = 0
            else:
                self.entries.extend(additions)
            self.message = f"Added {len(additions)} entries; deduplicated {duplicates} identical files."
            self.changed()

    def snapshot(self):
        with self.condition:
            return dict(revision=self.revision, active=self.active, paused=self.paused,
                        waiting=self.waiting, current=self.current, next=self.cursor,
                        batch=self.batch, batch_count=self.batch_count, message=self.message,
                        entries=[asdict(e) for e in self.entries])

    def document(self):
        with self.condition:
            rows = []
            for entry in self.entries:
                row = asdict(entry)
                for key in ("source_preview", "status", "batches", "error"):
                    row.pop(key)
                rows.append(row)
            return dict(schema=SCHEMA, version=1, entries=rows)

    def command(self, action, key=None, **kwargs):
        with self.condition:
            if action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
            elif action in ("previous", "skip", "jump"):
                base = self.current if self.active else self.cursor
                if action == "jump":
                    if not self.find(key):
                        raise ValueError("Entry no longer exists.")
                    target = key
                elif action == "previous":
                    target = self.adjacent(base, -1) or base
                else:
                    target = self.adjacent(base, 1)
                if self.active:
                    self.pending = target or "end"
                else:
                    self.cursor, self.batch = target, 0
            elif action == "move":
                entry = self.find(key)
                if not entry:
                    raise ValueError("Entry no longer exists.")
                index = max(0, min(len(self.entries) - 1, int(kwargs["index"])))
                self.entries.remove(entry)
                self.entries.insert(index, entry)
            elif action == "order":
                kind = kwargs.get("kind")
                if kind in ("filename", "filename_reverse"):
                    def filename_key(entry):
                        return tuple(int(part) if part.isdigit() else part.casefold()
                                     for part in re.split(r"(\d+)", entry.name))
                    self.entries.sort(key=filename_key, reverse=kind == "filename_reverse")
                elif kind == "reverse":
                    self.entries.reverse()
                elif kind == "shuffle":
                    random.SystemRandom().shuffle(self.entries)
                else:
                    raise ValueError("Unknown ordering mode.")
                # Keep an active batch attached to its entry. Idle sorting starts
                # at the beginning of the newly arranged list.
                if not self.active:
                    self.cursor = None
                    self.batch = 0
                self.message = "Queue order updated. Drag rows or use ↑/↓ to adjust it further."
            elif action in ("remove", "edit", "clear"):
                if self.active:
                    raise ValueError("Stop generation before editing or removing entries. Reordering and merging remain available.")
                if action == "clear":
                    self.entries.clear()
                    self.cursor = self.current = None
                else:
                    entry = self.find(key)
                    if not entry:
                        raise ValueError("Entry no longer exists.")
                    if action == "remove":
                        if self.cursor == key:
                            self.cursor = self.adjacent(key, 1)
                            self.batch = 0
                        self.entries.remove(entry)
                    else:
                        for field_name in ("Prompt", "Negative prompt"):
                            value = kwargs.get(field_name, entry.parameters.get(field_name, ""))
                            if not isinstance(value, str) or len(value) > MAX_TEXT:
                                raise ValueError("Invalid prompt.")
                            entry.parameters[field_name] = value
            else:
                raise ValueError("Unknown queue action.")
            self.changed()

    def begin(self, batch_count):
        with self.condition:
            if self.active:
                raise ValueError("This queue is already generating.")
            if not self.entries:
                raise ValueError("Add images with generation metadata or import a queue JSON first.")
            self.active = True
            self.pending = None
            self.batch_count = batch_count
            if not self.find(self.cursor):
                self.cursor, self.batch = self.entries[0].id, 0
            # Generate again resumes an interrupted entry from its first batch.
            self.batch = 0
            self.message = "Generating. Navigation takes effect after the current batch."
            self.changed()

    def take(self, interrupted):
        with self.condition:
            while self.paused and not interrupted():
                if not self.waiting:
                    self.waiting = True
                    self.changed()
                self.condition.wait(0.25)
            self.waiting = False
            if interrupted():
                return None
            if self.pending is not None:
                previous = self.find(self.current)
                if previous and previous.status == "generating":
                    previous.status = "partial" if previous.batches else "skipped"
                self.cursor = None if self.pending == "end" else self.pending
                self.pending, self.batch = None, 0
            entry = self.find(self.cursor)
            if entry is None:
                return None
            self.current = entry.id
            entry.status, entry.error = "generating", ""
            if self.batch == 0:
                entry.batches = 0
            self.changed()
            return copy.deepcopy(entry), self.batch

    def complete_batch(self, key, preview=""):
        with self.condition:
            entry = self.find(key)
            self.batch += 1
            entry.batches = self.batch
            if preview and not entry.generated_preview:
                entry.generated_preview = preview
            if self.batch >= self.batch_count:
                entry.status = "done"
                self.cursor, self.batch = self.adjacent(key, 1), 0
            self.changed()

    def fail(self, key, error):
        with self.condition:
            entry = self.find(key)
            entry.status, entry.error = "error", str(error)
            self.cursor, self.batch = self.adjacent(key, 1), 0
            self.changed()

    def end(self, interrupted=False):
        with self.condition:
            self.active = self.waiting = False
            entry = self.find(self.current)
            if entry and entry.status == "generating":
                entry.status = "partial"
            self.message = "Stopped. Generate resumes the selected entry from its first batch." if interrupted else "Queue finished. Generate runs it again."
            self.changed()


_queues = {}
_lock = threading.Lock()


def get_queue(token: str) -> ReplayQueue:
    if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("Metadata Replay is still initializing. Reload the page and try again.")
    with _lock:
        # Expire abandoned browser sessions; active/paused generations are retained.
        expired = [k for k, q in _queues.items() if not q.active and time.monotonic() - q.touched > 86400]
        for key in expired:
            del _queues[key]
        if token not in _queues:
            if len(_queues) >= 64:
                raise ValueError("Too many open Metadata Replay sessions. Close unused sessions or restart Forge.")
            _queues[token] = ReplayQueue()
        queue = _queues[token]
        queue.touched = time.monotonic()
        return queue

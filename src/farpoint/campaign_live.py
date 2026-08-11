"""Live campaign publication and read-only Dashboard projections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable

from farpoint.campaign import CampaignEventLog, atomic_status_write


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _iso_timestamp(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat()


@dataclass
class LiveCampaignPublisher:
    """Publish one campaign's status, events, heartbeat, and active preview."""

    root: Path
    campaign_id: str
    segment_id: str
    heartbeat_interval_seconds: float = 5.0
    preview_interval_seconds: float = 1.0
    clock: Any = time.time
    _status: dict[str, Any] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _last_preview_unix: float = field(default=-math.inf, init=False)
    _event_log: CampaignEventLog = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.campaign_id or not self.segment_id:
            raise ValueError("campaign_id and segment_id must be non-empty")
        if self.heartbeat_interval_seconds <= 0 or self.preview_interval_seconds <= 0:
            raise ValueError("live publication intervals must be positive")
        self._event_log = CampaignEventLog(
            self.root / "events.jsonl", self.campaign_id
        )

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def preview_path(self) -> Path:
        return self.root / "active-preview.jpg"

    @property
    def event_log(self) -> CampaignEventLog:
        return self._event_log

    def start(self, *, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("live campaign publisher is already started")
            now = float(self.clock())
            self._status = {
                "campaign_id": self.campaign_id,
                "segment_id": self.segment_id,
                "execution_status": "RUNNING",
                "quality_status": "NOT_EVALUATED",
                "heartbeat_unix": now,
                "started_unix": now,
                "updated_unix": now,
                "completed_attempts": 0,
                "successful_episodes": 0,
                **deepcopy(payload or {}),
            }
            atomic_status_write(self.status_path, self._status)
            self.event_log.append(
                "segment_started", deepcopy(payload or {}),
                segment_id=self.segment_id, timestamp_unix=now,
            )
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"campaign-heartbeat-{self.campaign_id}",
                daemon=True,
            )
            self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            self.heartbeat()

    def _write_status(self, **updates: Any) -> None:
        now = float(self.clock())
        self._status.update(deepcopy(updates))
        self._status["heartbeat_unix"] = now
        self._status["updated_unix"] = now
        atomic_status_write(self.status_path, self._status)

    def heartbeat(self) -> None:
        with self._lock:
            if not self._status or self._status["execution_status"] != "RUNNING":
                return
            self._write_status()
            self.event_log.append(
                "heartbeat", {"alive": True},
                segment_id=self.segment_id,
                timestamp_unix=self._status["heartbeat_unix"],
            )

    def event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            event = self.event_log.append(
                event_type, payload,
                segment_id=self.segment_id,
                timestamp_unix=float(self.clock()),
            )
            self._write_status(last_event_sequence=event["sequence"])
            return event

    def attempt_started(self, attempt_id: str, variation_id: str) -> None:
        with self._lock:
            self._write_status(
                active_attempt_id=attempt_id,
                active_variation_id=variation_id,
            )
            event = self.event_log.append(
                "attempt_started",
                {"attempt_id": attempt_id, "variation_id": variation_id},
                segment_id=self.segment_id,
                timestamp_unix=float(self.clock()),
            )
            self._write_status(last_event_sequence=event["sequence"])

    def attempt_completed(
        self,
        *,
        attempt_id: str,
        variation_id: str,
        success: bool,
        dataset_valid: bool,
        episode_id: str | None,
        failure_reason: str | None,
    ) -> None:
        with self._lock:
            completed = int(self._status.get("completed_attempts", 0)) + 1
            successes = int(self._status.get("successful_episodes", 0))
            if success and dataset_valid:
                successes += 1
            self._write_status(
                completed_attempts=completed,
                successful_episodes=successes,
                active_attempt_id=None,
                active_variation_id=None,
            )
            payload = {
                "attempt_id": attempt_id,
                "variation_id": variation_id,
                "episode_id": episode_id,
                "success": bool(success),
                "dataset_valid": bool(dataset_valid),
                "failure_reason": failure_reason,
            }
            event = self.event_log.append(
                "attempt_completed", payload,
                segment_id=self.segment_id,
                timestamp_unix=float(self.clock()),
            )
            if success and dataset_valid and episode_id:
                event = self.event_log.append(
                    "episode_completed",
                    {"episode_id": episode_id, "variation_id": variation_id},
                    segment_id=self.segment_id,
                    timestamp_unix=float(self.clock()),
                )
            self._write_status(last_event_sequence=event["sequence"])

    def publish_preview(self, jpeg: bytes, *, force: bool = False) -> bool:
        if not jpeg.startswith(b"\xff\xd8"):
            raise ValueError("active preview must be JPEG bytes")
        with self._lock:
            now = float(self.clock())
            if not force and now - self._last_preview_unix < self.preview_interval_seconds:
                return False
            _atomic_bytes(self.preview_path, jpeg)
            self._last_preview_unix = now
            event = self.event_log.append(
                "preview_updated",
                {"path": self.preview_path.name, "size_bytes": len(jpeg)},
                segment_id=self.segment_id,
                timestamp_unix=now,
            )
            self._write_status(
                active_preview_updated_unix=now,
                last_event_sequence=event["sequence"],
            )
            return True

    def preview_due(self) -> bool:
        with self._lock:
            return float(self.clock()) - self._last_preview_unix >= self.preview_interval_seconds

    def finish(self, *, execution_status: str, quality_status: str) -> None:
        if execution_status not in {"FINISHED", "ABORTED", "PAUSED"}:
            raise ValueError("live campaign terminal execution_status is invalid")
        if quality_status not in {"NOT_EVALUATED", "PASS", "FAIL"}:
            raise ValueError("live campaign terminal quality_status is invalid")
        with self._lock:
            self._stop.set()
            self._write_status(
                execution_status=execution_status,
                quality_status=quality_status,
                finished_unix=float(self.clock()),
            )
            event_type = "segment_finished" if execution_status == "FINISHED" else "watchdog_paused"
            event = self.event_log.append(
                event_type,
                {"execution_status": execution_status, "quality_status": quality_status},
                segment_id=self.segment_id,
                timestamp_unix=float(self.clock()),
            )
            self._write_status(last_event_sequence=event["sequence"])
            thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(self.heartbeat_interval_seconds * 2, 1.0))


def _campaign_directories(roots: Iterable[Path]) -> tuple[Path, ...]:
    found: dict[Path, None] = {}
    for root in roots:
        root = Path(root).resolve()
        if (root / "campaign.json").is_file():
            found[root] = None
        if root.is_dir():
            for campaign_path in root.rglob("campaign.json"):
                found[campaign_path.parent.resolve()] = None
    return tuple(sorted(found))


@dataclass(frozen=True)
class CampaignDashboardIndex:
    roots: tuple[Path, ...]
    stale_after_seconds: float = 60.0

    def __init__(self, roots: Iterable[str | Path], stale_after_seconds: float = 60.0):
        object.__setattr__(self, "roots", tuple(Path(root).resolve() for root in roots))
        object.__setattr__(self, "stale_after_seconds", float(stale_after_seconds))
        if self.stale_after_seconds <= 0:
            raise ValueError("campaign stale timeout must be positive")

    def _records(self, now_unix: float | None = None) -> list[dict[str, Any]]:
        now = float(time.time() if now_unix is None else now_unix)
        rows = []
        for root in _campaign_directories(self.roots):
            try:
                campaign = _read_json(root / "campaign.json")
                status = _read_json(root / "status.json") if (root / "status.json").is_file() else {}
            except (OSError, ValueError):
                continue
            heartbeat = status.get("heartbeat_unix")
            stale = (
                status.get("execution_status") == "RUNNING"
                and (not isinstance(heartbeat, (int, float)) or now - float(heartbeat) > self.stale_after_seconds)
            )
            target = campaign.get("target") or {}
            rows.append(
                {
                    "campaign_id": campaign.get("campaign_id") or root.name,
                    "campaign_version": campaign.get("campaign_version"),
                    "task_id": campaign.get("task_id"),
                    "execution_status": "STALE" if stale else status.get("execution_status", "NOT_STARTED"),
                    "quality_status": status.get("quality_status", "NOT_EVALUATED"),
                    "heartbeat_unix": heartbeat,
                    "started_unix": status.get("started_unix"),
                    "updated_unix": status.get("updated_unix"),
                    "started_at": _iso_timestamp(status.get("started_unix")),
                    "updated_at": _iso_timestamp(status.get("updated_unix")),
                    "completed_attempts": int(status.get("completed_attempts", 0)),
                    "successful_episodes": int(status.get("successful_episodes", 0)),
                    "target_successful_episodes": int(target.get("successful_episodes", 0)),
                    "active_attempt_id": status.get("active_attempt_id"),
                    "segment_id": status.get("segment_id"),
                    "stale": stale,
                    "preview_url": (
                        f"/api/campaigns/{campaign.get('campaign_id') or root.name}/active-preview"
                        if (root / "active-preview.jpg").is_file()
                        else None
                    ),
                    "root": str(root),
                }
            )
        return sorted(rows, key=lambda row: row.get("updated_unix") or 0, reverse=True)

    def live_runs(self, now_unix: float | None = None) -> list[dict[str, Any]]:
        return [
            row for row in self._records(now_unix)
            if row["execution_status"] in {"RUNNING", "STALE", "PAUSED"}
        ]

    def collections(self, now_unix: float | None = None) -> list[dict[str, Any]]:
        return self._records(now_unix)

    def benchmarks(self, now_unix: float | None = None) -> list[dict[str, Any]]:
        return [
            row for row in self._records(now_unix)
            if row["execution_status"] == "FINISHED" and row["quality_status"] == "PASS"
        ]

    def campaign_root(self, campaign_id: str) -> Path:
        matches = [
            root for root in _campaign_directories(self.roots)
            if (_read_json(root / "campaign.json").get("campaign_id") or root.name) == campaign_id
        ]
        if len(matches) != 1:
            raise FileNotFoundError(campaign_id)
        return matches[0]

    def campaign_detail(self, campaign_id: str) -> dict[str, Any]:
        root = self.campaign_root(campaign_id)
        segments = []
        for path in sorted((root / "segments").glob("*/segment.json")):
            try:
                segments.append(_read_json(path))
            except (OSError, ValueError):
                continue
        return {
            "campaign": _read_json(root / "campaign.json"),
            "status": _read_json(root / "status.json") if (root / "status.json").is_file() else None,
            "segments": segments,
        }

    def segment_detail(self, campaign_id: str, segment_id: str) -> dict[str, Any]:
        if not segment_id or "/" in segment_id or "\\" in segment_id:
            raise FileNotFoundError(segment_id)
        path = self.campaign_root(campaign_id) / "segments" / segment_id / "segment.json"
        if not path.is_file():
            raise FileNotFoundError(segment_id)
        return _read_json(path)

    def events(self, *, after_sequence: int = -1) -> list[dict[str, Any]]:
        events = []
        for root in _campaign_directories(self.roots):
            path = root / "events.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if int(event.get("sequence", -1)) > after_sequence:
                    events.append(event)
        return sorted(events, key=lambda event: (event["timestamp_unix"], event["campaign_id"], event["sequence"]))

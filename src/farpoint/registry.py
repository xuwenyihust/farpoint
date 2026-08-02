import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 2
TERMINAL_STATUSES = {"PASS", "FAIL"}
BENCHMARK_ALIASES = {
    "robotsim_v1_release_candidate": "farpoint_v1_release_candidate",
}


def canonical_benchmark_id(value):
    """Return the public benchmark ID while preserving raw-data lineage."""
    if value is None:
        return None
    value = str(value)
    return BENCHMARK_ALIASES.get(value, value)


def utc_now():
    return datetime.now(timezone.utc)


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def directory_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


@dataclass(frozen=True)
class DataLayout:
    outputs_root: Path

    @property
    def episodes(self):
        return self.outputs_root / "episodes"

    @property
    def benchmarks(self):
        return self.outputs_root / "benchmarks"

    @property
    def datasets(self):
        return self.outputs_root / "datasets"

    @property
    def reports(self):
        return self.outputs_root / "reports"

    @property
    def state(self):
        return self.outputs_root / ".data-platform"

    @property
    def database(self):
        return self.state / "registry.sqlite3"

    @property
    def runs(self):
        return self.state / "runs"

    @property
    def pins(self):
        return self.state / "pins"

    @property
    def quarantine(self):
        return self.outputs_root / "quarantine"

    @property
    def audit_log(self):
        return self.state / "audit.jsonl"

    def ensure(self):
        for path in (
            self.outputs_root,
            self.episodes,
            self.benchmarks,
            self.datasets,
            self.reports,
            self.state,
            self.runs,
            self.pins,
            self.quarantine,
        ):
            path.mkdir(parents=True, exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    artifact_path TEXT,
    task_name TEXT,
    task_type TEXT,
    status TEXT NOT NULL,
    health TEXT NOT NULL,
    seed INTEGER,
    benchmark_id TEXT,
    benchmark_repeat INTEGER,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL,
    elapsed_seconds REAL,
    failure_category TEXT,
    failure_reason TEXT,
    dataset_valid INTEGER,
    observation_count INTEGER,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    preview_path TEXT,
    report_path TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    protected_reason TEXT,
    run_id TEXT,
    diagnostic_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS episodes_filter_idx
ON episodes(status, task_name, benchmark_id, seed, started_at);
CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL DEFAULT 'BENCHMARK',
    task_name TEXT,
    task_type TEXT,
    status TEXT NOT NULL,
    created_at TEXT,
    finished_at TEXT,
    planned_trials INTEGER,
    completed_trials INTEGER,
    passed_trials INTEGER,
    success_rate REAL,
    accepted INTEGER,
    manifest_path TEXT NOT NULL,
    report_path TEXT,
    size_bytes INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS storage_snapshots (
    captured_at TEXT PRIMARY KEY,
    total_bytes INTEGER NOT NULL,
    available_bytes INTEGER NOT NULL,
    episode_bytes INTEGER NOT NULL,
    benchmark_bytes INTEGER NOT NULL,
    dataset_bytes INTEGER NOT NULL,
    report_bytes INTEGER NOT NULL
);
"""


class EpisodeRegistry:
    def __init__(self, outputs_root, incomplete_timeout_seconds=1800):
        self.layout = DataLayout(Path(outputs_root).resolve())
        self.incomplete_timeout_seconds = int(incomplete_timeout_seconds)
        self.layout.ensure()

    def connect(self):
        connection = sqlite3.connect(self.layout.database)
        connection.row_factory = sqlite3.Row
        try:
            self._initialize(connection)
            return connection
        except sqlite3.DatabaseError:
            connection.close()
            timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
            corrupt = self.layout.database.with_name(
                f"{self.layout.database.name}.corrupt-{timestamp}"
            )
            if self.layout.database.exists():
                self.layout.database.replace(corrupt)
            connection = sqlite3.connect(self.layout.database)
            connection.row_factory = sqlite3.Row
            self._initialize(connection)
            return connection

    @staticmethod
    def _initialize(connection):
        connection.executescript(SCHEMA)
        benchmark_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(benchmarks)")
        }
        if "record_type" not in benchmark_columns:
            connection.execute(
                "ALTER TABLE benchmarks ADD COLUMN record_type TEXT NOT NULL DEFAULT 'BENCHMARK'"
            )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.commit()

    def rebuild(self):
        backup = self.layout.database.with_suffix(".sqlite3.previous")
        if self.layout.database.exists():
            try:
                shutil.copy2(self.layout.database, backup)
            except OSError:
                pass
            self.layout.database.unlink()
        return self.scan()

    def scan(self):
        now = utc_now()
        with self.connect() as connection:
            cached = {
                row["episode_id"]: dict(row)
                for row in connection.execute(
                    "SELECT episode_id, status, size_bytes FROM episodes"
                )
            }
        episode_rows = [
            self._episode_row(path, now, cached.get(path.name))
            for path in self._episode_dirs()
        ]
        episode_rows.extend(self._active_run_rows(now, episode_rows))
        benchmark_paths = []
        for directory in sorted(self.layout.benchmarks.iterdir()):
            if not directory.is_dir():
                continue
            manifest = directory / "manifest.json"
            run_state = directory / "run-state.json"
            if manifest.is_file():
                benchmark_paths.append(manifest)
            elif run_state.is_file():
                benchmark_paths.append(run_state)
        benchmark_rows = [
            row
            for path in benchmark_paths
            if (row := self._benchmark_row(path)) is not None
        ]
        with self.connect() as connection:
            connection.execute("DELETE FROM episodes")
            connection.execute("DELETE FROM benchmarks")
            for row in episode_rows:
                self._insert_mapping(connection, "episodes", row)
            for row in benchmark_rows:
                self._insert_mapping(connection, "benchmarks", row)
            self._capture_storage(connection, now)
            connection.commit()
        return {
            "episodes": len(episode_rows),
            "benchmarks": len(benchmark_rows),
            "statuses": self.status_counts(),
        }

    def _episode_dirs(self):
        return sorted(
            path
            for path in self.layout.episodes.glob("episode_*")
            if path.is_dir()
        )

    def _episode_row(self, path, now, cached=None):
        metadata_path = path / "metadata.json"
        metrics_path = path / "metrics.json"
        metadata = {}
        metrics = {}
        problems = []
        metadata_valid = False
        metrics_valid = False
        try:
            metadata = read_json(metadata_path)
            metadata_valid = isinstance(metadata, dict)
        except (OSError, ValueError) as error:
            problems.append(f"metadata:{type(error).__name__}")
        try:
            metrics = read_json(metrics_path)
            metrics_valid = isinstance(metrics, dict)
        except (OSError, ValueError) as error:
            problems.append(f"metrics:{type(error).__name__}")

        updated = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        age_seconds = max(0.0, (now - updated).total_seconds())
        episode_id = metadata.get("episode_id") if metadata_valid else None
        health = "OK"
        if not metadata_valid:
            health = "ORPHANED" if not metadata_path.exists() else "CORRUPT"
        elif episode_id != path.name:
            health = "ORPHANED"
            problems.append("episode_id_path_mismatch")
        elif not metrics_valid and metrics_path.exists():
            health = "CORRUPT"

        if metadata_valid and metrics_valid:
            status = "PASS" if metrics.get("success") is True else "FAIL"
        elif age_seconds <= self.incomplete_timeout_seconds:
            status = "RUNNING"
        else:
            status = "INCOMPLETE"

        benchmark_id = canonical_benchmark_id(
            metadata.get("benchmark_id") or metrics.get("benchmark_id")
        )
        pinned = (self.layout.pins / f"{path.name}.json").exists()
        protected_reason = None
        if pinned:
            protected_reason = "pinned"
        elif benchmark_id:
            protected_reason = "benchmark"
        preview = next(iter(sorted((path / "preview").glob("*.png"))), None)
        report = self.layout.reports / path.name / "index.html"
        size_bytes = (
            cached["size_bytes"]
            if cached
            and cached.get("status") in TERMINAL_STATUSES
            and status in TERMINAL_STATUSES
            else directory_size(path)
        )
        return {
            "episode_id": path.name,
            "artifact_path": str(path),
            "task_name": metadata.get("task_name") or metrics.get("task_name"),
            "task_type": metadata.get("task_type") or metrics.get("task_type"),
            "status": status,
            "health": health,
            "seed": self._integer(metadata.get("episode_seed", metrics.get("episode_seed"))),
            "benchmark_id": benchmark_id,
            "benchmark_repeat": self._integer(
                metadata.get("benchmark_repeat", metrics.get("benchmark_repeat"))
            ),
            "started_at": metadata.get("started_at"),
            "finished_at": metadata.get("finished_at"),
            "updated_at": updated.isoformat(),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
            "failure_category": metrics.get("failure_category"),
            "failure_reason": metrics.get("failure_reason"),
            "dataset_valid": self._boolean(metrics.get("dataset_valid")),
            "observation_count": self._integer(metrics.get("dataset_observation_count")),
            "size_bytes": size_bytes,
            "preview_path": str(preview) if preview else None,
            "report_path": str(report) if report.exists() else None,
            "pinned": int(pinned),
            "protected_reason": protected_reason,
            "run_id": metadata.get("run_id"),
            "diagnostic_json": json.dumps(
                {"problems": problems, "age_seconds": round(age_seconds, 3)},
                sort_keys=True,
            ),
        }

    def _active_run_rows(self, now, episode_rows):
        rows = []
        known_run_ids = {row.get("run_id") for row in episode_rows if row.get("run_id")}
        for path in sorted(self.layout.runs.glob("*.json")):
            try:
                run = read_json(path)
            except (OSError, ValueError):
                continue
            run_id = str(run.get("run_id") or path.stem)
            if run_id in known_run_ids:
                continue
            started = parse_time(run.get("started_at"))
            age = (now - started).total_seconds() if started else float("inf")
            raw_status = str(run.get("status", "RUNNING")).upper()
            if raw_status == "RUNNING" and age > self.incomplete_timeout_seconds:
                status = "INCOMPLETE"
                reason = "run heartbeat exceeded timeout"
            elif raw_status in {"PASS", "FAIL", "INCOMPLETE"}:
                status = raw_status
                reason = run.get("failure_reason")
            else:
                status = "RUNNING"
                reason = None
            rows.append(
                {
                    "episode_id": f"run:{run_id}",
                    "artifact_path": None,
                    "task_name": run.get("task_name"),
                    "task_type": run.get("task_type"),
                    "status": status,
                    "health": "OK" if status == "RUNNING" else "ORPHANED",
                    "seed": self._integer(run.get("seed")),
                    "benchmark_id": canonical_benchmark_id(run.get("benchmark_id")),
                    "benchmark_repeat": self._integer(run.get("benchmark_repeat")),
                    "started_at": run.get("started_at"),
                    "finished_at": run.get("finished_at"),
                    "updated_at": run.get("updated_at") or run.get("started_at") or now.isoformat(),
                    "elapsed_seconds": None,
                    "failure_category": "runner" if status != "RUNNING" else None,
                    "failure_reason": reason,
                    "dataset_valid": None,
                    "observation_count": None,
                    "size_bytes": 0,
                    "preview_path": None,
                    "report_path": None,
                    "pinned": 0,
                    "protected_reason": "benchmark" if run.get("benchmark_id") else None,
                    "run_id": run_id,
                    "diagnostic_json": json.dumps({"run_state_path": str(path)}),
                }
            )
        return rows

    def _benchmark_row(self, manifest_path):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError):
            return None
        runtime = {}
        runtime_path = manifest_path.parent / "run-state.json"
        if manifest_path.name == "manifest.json" and runtime_path.is_file():
            try:
                runtime = read_json(runtime_path)
            except (OSError, ValueError):
                runtime = {}
        acceptance = manifest.get("acceptance") or {}
        collection = manifest.get("schema_version") in {
            "farpoint.collection.v1",
            "farpoint.collection-run.v1",
        }
        completed = int(
            manifest.get("completed_trials")
            or runtime.get("completed_trials")
            or manifest.get("task_attempts")
            or runtime.get("task_attempts")
            or acceptance.get("observed_task_attempts")
            or len(manifest.get("trials", []))
            or len(manifest.get("attempts", []))
        )
        planned = int(
            manifest.get("planned_trials")
            or runtime.get("planned_trials")
            or acceptance.get("maximum_task_attempts")
            or (runtime.get("acceptance") or {}).get("maximum_task_attempts")
            or len(manifest.get("trials", []))
            or len(manifest.get("attempts", []))
        )
        accepted = manifest.get("accepted")
        if accepted is None:
            accepted = acceptance.get("accepted")
        if accepted is True:
            status = "PASS"
        elif manifest.get("execution_status") == "PILOT_COMPLETE":
            status = "PILOT"
        elif manifest.get("execution_status") == "ABORTED":
            status = "FAIL"
        elif manifest.get("execution_status") == "RUNNING":
            status = "RUNNING"
        elif manifest.get("finished_at") or runtime.get("finished_at"):
            status = "FAIL"
        elif completed:
            status = "RUNNING"
        else:
            status = "INCOMPLETE"
        benchmark_id = str(
            manifest.get("collection_id")
            or manifest.get("benchmark_id")
            or manifest_path.parent.name
        )
        report = self.layout.reports / "benchmarks" / benchmark_id / "index.html"
        return {
            "benchmark_id": benchmark_id,
            "record_type": "COLLECTION" if collection else "BENCHMARK",
            "task_name": manifest.get("task_name") or manifest.get("task_id"),
            "task_type": manifest.get("task_type")
            or runtime.get("task_type")
            or ("cube_position_collection" if collection else None),
            "status": status,
            "created_at": manifest.get("created_at") or runtime.get("created_at"),
            "finished_at": manifest.get("finished_at") or runtime.get("finished_at"),
            "planned_trials": planned,
            "completed_trials": completed,
            "passed_trials": int(
                manifest.get("passed_trials")
                or runtime.get("passed_trials")
                or manifest.get("task_successes")
                or runtime.get("task_successes")
                or acceptance.get("observed_task_successes")
                or acceptance.get("observed_successes")
                or 0
            ),
            "success_rate": (
                manifest.get("task_yield")
                if manifest.get("task_yield") is not None
                else manifest.get("success_rate")
                if manifest.get("success_rate") is not None
                else runtime.get(
                    "task_yield",
                    runtime.get(
                        "success_rate",
                        acceptance.get(
                            "observed_task_yield", acceptance.get("observed_success_rate")
                        ),
                    ),
                )
            ),
            "accepted": self._boolean(accepted),
            "manifest_path": str(manifest_path),
            "report_path": str(report) if report.exists() else None,
            "size_bytes": directory_size(manifest_path.parent),
        }

    def _capture_storage(self, connection, now):
        usage = shutil.disk_usage(self.layout.outputs_root)
        latest = connection.execute(
            "SELECT captured_at FROM storage_snapshots ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
        latest_time = parse_time(latest["captured_at"]) if latest else None
        if latest_time and (now - latest_time).total_seconds() < 60:
            return
        values = {
            "captured_at": now.isoformat(),
            "total_bytes": usage.total,
            "available_bytes": usage.free,
            "episode_bytes": directory_size(self.layout.episodes),
            "benchmark_bytes": directory_size(self.layout.benchmarks),
            "dataset_bytes": directory_size(self.layout.datasets),
            "report_bytes": directory_size(self.layout.reports),
        }
        self._insert_mapping(connection, "storage_snapshots", values)
        connection.execute(
            """
            DELETE FROM storage_snapshots
            WHERE captured_at NOT IN (
                SELECT captured_at FROM storage_snapshots
                ORDER BY captured_at DESC LIMIT 10000
            )
            """
        )

    @staticmethod
    def _insert_mapping(connection, table, mapping):
        columns = ", ".join(mapping)
        placeholders = ", ".join("?" for _ in mapping)
        connection.execute(
            f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(mapping.values()),
        )

    @staticmethod
    def _integer(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _boolean(value):
        return None if value is None else int(bool(value))

    def status_counts(self):
        with self.connect() as connection:
            return {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM episodes GROUP BY status"
                )
            }

    def list_episodes(self, filters=None, limit=500, offset=0):
        filters = filters or {}
        clauses = []
        values = []
        supported = {
            "task": "task_name",
            "status": "status",
            "benchmark": "benchmark_id",
            "seed": "seed",
            "health": "health",
        }
        for key, column in supported.items():
            value = filters.get(key)
            if value not in (None, ""):
                clauses.append(f"{column} = ?")
                values.append(value.upper() if key in {"status", "health"} else value)
        if filters.get("start"):
            clauses.append("started_at >= ?")
            values.append(filters["start"])
        if filters.get("end"):
            clauses.append("started_at <= ?")
            values.append(filters["end"])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM episodes {where}
                ORDER BY COALESCE(started_at, updated_at) DESC
                LIMIT ? OFFSET ?
                """,
                (*values, int(limit), int(offset)),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_benchmarks(self):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM benchmarks ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def storage_summary(self):
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM storage_snapshots
                ORDER BY captured_at DESC LIMIT 168
                """
            ).fetchall()
        snapshots = [dict(row) for row in reversed(rows)]
        latest = snapshots[-1] if snapshots else {}
        growth_per_day = 0.0
        if len(snapshots) >= 2:
            first = snapshots[0]
            first_time = parse_time(first["captured_at"])
            last_time = parse_time(latest["captured_at"])
            seconds = (last_time - first_time).total_seconds()
            if seconds > 0:
                growth_per_day = max(
                    0.0,
                    (latest["episode_bytes"] - first["episode_bytes"])
                    * 86400
                    / seconds,
                )
        remaining_days = (
            latest.get("available_bytes", 0) / growth_per_day
            if growth_per_day > 0
            else None
        )
        return {
            "latest": latest,
            "growth_bytes_per_day": growth_per_day,
            "estimated_remaining_days": remaining_days,
            "history": snapshots,
        }

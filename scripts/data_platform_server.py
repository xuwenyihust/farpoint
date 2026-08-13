#!/usr/bin/env python3
import argparse
import base64
import hmac
import ipaddress
import json
import mimetypes
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data_platform_cli import build_reports  # noqa: E402
from farpoint.campaign_live import CampaignDashboardIndex  # noqa: E402
from farpoint.policy_rollout_dashboard import PolicyRolloutDashboardIndex  # noqa: E402
from farpoint.registry import EpisodeRegistry  # noqa: E402
from farpoint.retention import RetentionManager  # noqa: E402


def valid_basic_auth(header, expected_token):
    if not expected_token or not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
        username, password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError):
        return False
    return username == "farpoint" and hmac.compare_digest(
        password,
        expected_token,
    )


def metadata_episode_id(metadata):
    identity = metadata.get("identity") if isinstance(metadata, dict) else None
    return metadata.get("episode_id") or (
        identity.get("episode_id") if isinstance(identity, dict) else None
    )


def read_registered_episode_record(episode_dir, episode_id):
    episode_dir = Path(episode_dir).resolve()
    if episode_dir.name != episode_id:
        raise ValueError("episode asset root identity mismatch")
    metadata_path = episode_dir / "metadata.json"
    if metadata_path.exists():
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("episode asset root is not valid") from error
        if metadata_episode_id(record) == episode_id:
            return record
        raise ValueError("episode asset root identity mismatch")
    try:
        record = json.loads((episode_dir / "run-state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("episode asset root is not valid") from error
    if metadata_episode_id(record) == episode_id:
        return record
    raise ValueError("episode asset root is not valid")


def resolve_registered_episode_asset(episode_dir, episode_id, relative):
    episode_dir = Path(episode_dir).resolve()
    relative_path = Path(unquote(str(relative)))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("invalid episode asset path")
    read_registered_episode_record(episode_dir, episode_id)
    target = (episode_dir / relative_path).resolve()
    if target != episode_dir and episode_dir not in target.parents:
        raise ValueError("episode asset path escapes its episode root")
    return target


def build_preview_manifest(episodes_root, episode_id, episode_dir=None):
    episode_id = unquote(str(episode_id))
    if not episode_id or "/" in episode_id or "\\" in episode_id:
        raise ValueError("invalid episode id")
    if episode_dir is None:
        try:
            episode_dir = resolve_episode_asset(episodes_root, f"{episode_id}/metadata.json").parent
        except ValueError as error:
            raise FileNotFoundError(episode_id) from error
    else:
        try:
            episode_dir = resolve_registered_episode_asset(
                episode_dir, episode_id, "metadata.json"
            ).parent
        except ValueError as error:
            raise FileNotFoundError(episode_id) from error
    if not episode_dir.is_dir():
        raise FileNotFoundError(episode_id)
    frames = sorted((episode_dir / "preview").glob("*.png"))
    if not frames:
        frames = sorted((episode_dir / "rgb").glob("front_*.png"))
    encoded_episode = quote(episode_id, safe="")
    return {
        "episode_id": episode_id,
        "frame_count": len(frames),
        "frames": [
            f"/files/episodes/{encoded_episode}/"
            f"{quote(frame.relative_to(episode_dir).as_posix(), safe='/')}"
            for frame in frames
        ],
    }


def build_episode_detail(episode_dir, episode_id, registry_row=None):
    """Return queryable v3 scene metadata without exposing arbitrary files."""
    metadata = read_registered_episode_record(episode_dir, episode_id)
    identity = metadata.get("identity") or {}
    task = metadata.get("task") or {}
    scene = metadata.get("scene") or {}
    variation = metadata.get("variation") or {}
    requested = variation.get("requested") or {}
    resolved = variation.get("resolved") or {}
    row = registry_row or {}
    return {
        "episode_id": episode_id,
        "schema_version": metadata.get("schema_version"),
        "task": {
            "task_id": task.get("task_id") or identity.get("task_id") or metadata.get("task_name"),
            "instruction": task.get("instruction"),
            "manipulated_entity_id": task.get("manipulated_entity_id"),
            "target_entity_id": task.get("target_entity_id"),
            "acceptance_region_id": task.get("acceptance_region_id"),
        },
        "variation": {
            "variation_id": variation.get("variation_id"),
            "split": identity.get("split") or variation.get("split"),
            "varied_axes": variation.get("varied_axes") or [],
            "frozen_axes": variation.get("frozen_axes") or [],
        },
        "scene_entities": scene.get("entities") or [],
        "requested_entities": requested.get("entities") or {},
        "resolved_entities": resolved.get("entities") or {},
        "source": {
            "collection_id": row.get("collection_id"),
            "source_root": row.get("source_root"),
            "managed": bool(row.get("managed", True)),
        },
    }


def resolve_episode_asset(episodes_root, relative):
    """Resolve a local or symlinked episode asset without allowing path escape."""
    episodes_root = Path(episodes_root).resolve()
    relative_path = Path(unquote(str(relative)))
    if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
        raise ValueError("invalid episode asset path")
    episode_id = relative_path.parts[0]
    episode_root = (episodes_root / episode_id).resolve()
    metadata_path = episode_root / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("episode asset root is not valid") from error
    if metadata_episode_id(metadata) != episode_id:
        raise ValueError("episode asset root identity mismatch")
    target = (episodes_root / relative_path).resolve()
    if target != episode_root and episode_root not in target.parents:
        raise ValueError("episode asset path escapes its episode root")
    return target


class PlatformState:
    def __init__(
        self,
        outputs_root,
        scan_interval,
        incomplete_timeout,
        episode_roots=None,
        campaign_roots=None,
        policy_rollout_roots=None,
    ):
        self.registry = EpisodeRegistry(
            outputs_root,
            incomplete_timeout,
            episode_roots=episode_roots,
        )
        self.retention = RetentionManager(self.registry)
        self.campaigns = CampaignDashboardIndex(campaign_roots or [], stale_after_seconds=60)
        self.policy_rollouts = PolicyRolloutDashboardIndex(policy_rollout_roots or [])
        self.scan_interval = scan_interval
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_scan = None
        self.last_error = None

    def refresh(self, reports=True):
        with self.lock:
            result = self.registry.scan()
            if reports:
                result["reports"] = build_reports(self.registry)
            self.last_scan = time.time()
            self.last_error = None
            return result

    def watch(self):
        while not self.stop_event.is_set():
            try:
                self.refresh(reports=True)
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
            self.stop_event.wait(self.scan_interval)


class PlatformHandler(BaseHTTPRequestHandler):
    server_version = "FarpointDataPlatform/1"

    @property
    def state(self):
        return self.server.platform_state

    def do_GET(self):
        if not self._client_allowed():
            self.send_error(HTTPStatus.FORBIDDEN, "local network access only")
            return
        if not self._authenticated():
            self._request_authentication()
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(PROJECT_ROOT / "dashboard" / "data-platform" / "index.html")
        elif parsed.path == "/api/health":
            self._json(
                {
                    "ok": self.state.last_error is None,
                    "last_scan_unix": self.state.last_scan,
                    "last_error": self.state.last_error,
                }
            )
        elif parsed.path == "/api/episodes":
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            rows = self.state.registry.list_episodes(
                filters=query,
                limit=min(int(query.get("limit", 500)), 2000),
                offset=max(int(query.get("offset", 0)), 0),
            )
            for row in rows:
                try:
                    row["cameras"] = json.loads(row.get("cameras_json") or "[]")
                except (TypeError, ValueError):
                    row["cameras"] = []
                row["preview_url"] = self._artifact_url(row.get("preview_path"), row)
                row["report_url"] = (
                    f"/reports/{row['episode_id']}/index.html" if row.get("report_path") else None
                )
            self._json({"episodes": rows, "count": len(rows)})
        elif parsed.path == "/api/live-runs":
            self._json({"live_runs": self.state.campaigns.live_runs()})
        elif parsed.path == "/api/collections":
            self._json({"collections": self.state.campaigns.collections()})
        elif parsed.path == "/api/policy-rollouts":
            self._json({"policy_rollouts": self.state.policy_rollouts.list_rollouts()})
        elif parsed.path.startswith("/api/policy-rollouts/"):
            rollout_id = unquote(parsed.path.removeprefix("/api/policy-rollouts/"))
            try:
                self._json(self.state.policy_rollouts.detail(rollout_id))
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
        elif parsed.path == "/api/events":
            self._serve_campaign_events(parsed)
        elif parsed.path.startswith("/api/campaigns/") and parsed.path.endswith("/active-preview"):
            campaign_id = unquote(
                parsed.path.removeprefix("/api/campaigns/").removesuffix("/active-preview")
            )
            try:
                self._serve_file(
                    self.state.campaigns.campaign_root(campaign_id) / "active-preview.jpg"
                )
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
        elif parsed.path.startswith("/api/campaigns/") and "/segments/" in parsed.path:
            relative = parsed.path.removeprefix("/api/campaigns/")
            campaign_id, segment_id = map(unquote, relative.split("/segments/", 1))
            try:
                self._json(self.state.campaigns.segment_detail(campaign_id, segment_id))
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
        elif parsed.path.startswith("/api/campaigns/"):
            campaign_id = unquote(parsed.path.removeprefix("/api/campaigns/"))
            try:
                self._json(self.state.campaigns.campaign_detail(campaign_id))
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
        elif parsed.path.startswith("/api/episodes/") and parsed.path.endswith("/preview"):
            episode_id = parsed.path.removeprefix("/api/episodes/").removesuffix("/preview")
            try:
                row = self.state.registry.get_episode(episode_id)
                if not row or not row.get("artifact_path"):
                    raise FileNotFoundError(episode_id)
                self._json(
                    build_preview_manifest(
                        self.state.registry.layout.episodes,
                        episode_id,
                        episode_dir=row["artifact_path"],
                    )
                )
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
            except ValueError as error:
                self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        elif parsed.path.startswith("/api/episodes/"):
            episode_id = unquote(parsed.path.removeprefix("/api/episodes/"))
            try:
                if not episode_id or "/" in episode_id or "\\" in episode_id:
                    raise ValueError("invalid episode id")
                row = self.state.registry.get_episode(episode_id)
                if not row or not row.get("artifact_path"):
                    raise FileNotFoundError(episode_id)
                self._json(build_episode_detail(row["artifact_path"], episode_id, registry_row=row))
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
            except ValueError as error:
                self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        elif parsed.path == "/api/benchmarks":
            rows = self.state.registry.list_benchmarks()
            for row in rows:
                row["report_url"] = (
                    f"/reports/benchmarks/{row['benchmark_id']}/index.html"
                    if row.get("report_path")
                    else None
                )
            known = {row["benchmark_id"] for row in rows}
            for campaign in self.state.campaigns.benchmarks():
                if campaign["campaign_id"] in known:
                    continue
                target = campaign["target_successful_episodes"]
                successes = campaign["successful_episodes"]
                rows.append(
                    {
                        "benchmark_id": campaign["campaign_id"],
                        "display_name": campaign["campaign_id"],
                        "record_type": "CAMPAIGN",
                        "task_name": campaign["task_id"],
                        "status": "PASS",
                        "created_at": campaign["started_at"],
                        "planned_trials": target,
                        "completed_trials": campaign["completed_attempts"],
                        "passed_trials": successes,
                        "success_rate": successes / target if target else None,
                        "accepted": 1,
                        "report_url": None,
                    }
                )
            self._json({"benchmarks": rows})
        elif parsed.path == "/api/storage":
            self._json(self.state.registry.storage_summary())
        elif parsed.path == "/api/quarantine":
            self._json({"items": self.state.retention.list_quarantine()})
        elif parsed.path == "/api/retention/preview":
            self._json(self.state.retention.preview())
        elif parsed.path == "/api/audit":
            self._json({"events": self._audit_events()})
        elif parsed.path.startswith("/reports/episodes/"):
            # Benchmark reports resolve ../../episodes/... to /reports/episodes/...
            self._serve_episode_asset(parsed.path.removeprefix("/reports/episodes/"))
        elif parsed.path.startswith("/reports/"):
            self._serve_under(
                self.state.registry.layout.reports,
                parsed.path.removeprefix("/reports/"),
            )
        elif parsed.path.startswith("/episodes/"):
            # Existing reports use relative ../../episodes/... asset links.
            self._serve_episode_asset(parsed.path.removeprefix("/episodes/"))
        elif parsed.path.startswith("/files/episodes/"):
            self._serve_episode_asset(parsed.path.removeprefix("/files/episodes/"))
        elif parsed.path.startswith("/policy-rollouts/"):
            relative = parsed.path.removeprefix("/policy-rollouts/")
            try:
                rollout_id, marker, scene_id, filename = map(unquote, relative.split("/", 3))
                if marker != "episodes" or not filename.endswith(".mp4"):
                    raise FileNotFoundError(relative)
                camera_id = filename.removesuffix(".mp4")
                self._serve_file(
                    self.state.policy_rollouts.video_path(rollout_id, scene_id, camera_id)
                )
            except (FileNotFoundError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if not self._client_allowed():
            self.send_error(HTTPStatus.FORBIDDEN, "local network access only")
            return
        if not self._authenticated():
            self._request_authentication()
            return
        parsed = urlparse(self.path)
        try:
            payload = self._request_json()
            if parsed.path == "/api/refresh":
                result = self.state.refresh(reports=True)
            elif parsed.path == "/api/rebuild":
                result = self.state.registry.rebuild()
            elif parsed.path == "/api/retention/preview":
                policy = self.state.retention.load_policy()
                policy.update(payload.get("policy", {}))
                result = self.state.retention.preview(policy)
            elif parsed.path == "/api/quarantine":
                result = self.state.retention.quarantine(
                    payload.get("episode_ids", []),
                    actor=self.client_address[0],
                    reason=payload.get("reason", "dashboard"),
                )
            elif parsed.path == "/api/restore":
                result = self.state.retention.restore(
                    payload["quarantine_id"],
                    actor=self.client_address[0],
                )
            elif parsed.path == "/api/pin":
                result = self.state.retention.pin(
                    payload["episode_id"],
                    payload.get("reason", "dashboard"),
                    actor=self.client_address[0],
                )
            elif parsed.path == "/api/unpin":
                result = self.state.retention.unpin(
                    payload["episode_id"],
                    actor=self.client_address[0],
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(result)
        except (KeyError, TypeError, ValueError, OSError) as error:
            self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def _client_allowed(self):
        try:
            address = ipaddress.ip_address(self.client_address[0])
        except ValueError:
            return False
        return address.is_private or address.is_loopback

    def _authenticated(self):
        expected = os.environ.get("FARPOINT_DASHBOARD_TOKEN")
        if not expected:
            return self.client_address[0] in {"127.0.0.1", "::1"}
        header = self.headers.get("Authorization", "")
        return valid_basic_auth(header, expected)

    def _request_authentication(self):
        body = b"Authentication required."
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self._security_headers()
        self.send_header("WWW-Authenticate", 'Basic realm="Farpoint Data Platform"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _request_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("request body exceeds 1 MiB")
        raw = self.rfile.read(length)
        return json.loads(raw or b"{}")

    def _serve_campaign_events(self, parsed):
        del parsed
        events = self.state.campaigns.events()
        last_id = self.headers.get("Last-Event-ID")
        if last_id:
            try:
                timestamp, campaign_id, sequence = last_id.split("|", 2)
                last_key = (float(timestamp), campaign_id, int(sequence))
                events = [
                    event
                    for event in events
                    if (
                        float(event["timestamp_unix"]),
                        str(event["campaign_id"]),
                        int(event["sequence"]),
                    )
                    > last_key
                ]
            except (TypeError, ValueError):
                events = events[-100:]
        else:
            events = events[-100:]
        body = "".join(
            f"id: {event['timestamp_unix']}|{event['campaign_id']}|{event['sequence']}\n"
            f"data: {json.dumps(event, sort_keys=True)}\n\n"
            for event in events
        ).encode()
        if not body:
            body = b": keepalive\n\n"
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status=HTTPStatus.OK):
        encoded = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _serve_under(self, root, relative):
        root = Path(root).resolve()
        target = (root / unquote(relative)).resolve()
        if target != root and root not in target.parents:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self._serve_file(target)

    def _serve_episode_asset(self, relative):
        try:
            episode_id, asset_relative = unquote(relative).split("/", 1)
            row = self.state.registry.get_episode(episode_id)
            if not row or not row.get("artifact_path"):
                raise ValueError("episode is not registered")
            target = resolve_registered_episode_asset(
                row["artifact_path"], episode_id, asset_relative
            )
        except (ValueError, OSError):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self._serve_file(target)

    def _serve_file(self, path):
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        partial = False
        if range_header and range_header.startswith("bytes="):
            try:
                requested = range_header.removeprefix("bytes=")
                if "," in requested:
                    raise ValueError("multiple byte ranges are unsupported")
                requested_start, requested_end = requested.split("-", 1)
                if not requested_start:
                    suffix_length = int(requested_end)
                    if suffix_length <= 0:
                        raise ValueError("invalid suffix byte range")
                    start = max(0, size - suffix_length)
                    end = size - 1
                else:
                    start = int(requested_start)
                    end = int(requested_end) if requested_end else size - 1
                if start < 0 or end < start or start >= size:
                    raise ValueError("invalid byte range")
                end = min(end, size - 1)
                partial = True
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self._security_headers()
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header(
            "Cache-Control",
            "public, max-age=86400, immutable"
            if any(part in path.parts for part in ("observations", "preview", "rgb"))
            else "no-cache",
        )
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining and (chunk := handle.read(min(1024 * 1024, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "media-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'",
        )

    def _artifact_url(self, path, row):
        if not path or not row.get("artifact_path"):
            return None
        try:
            relative = Path(path).resolve().relative_to(Path(row["artifact_path"]).resolve())
        except ValueError:
            return None
        return (
            f"/files/episodes/{quote(row['episode_id'], safe='')}/"
            f"{quote(relative.as_posix(), safe='/')}"
        )

    def _audit_events(self):
        path = self.state.registry.layout.audit_log
        if not path.exists():
            return []
        events = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        return events[-200:][::-1]

    def log_message(self, format_string, *args):
        sys.stderr.write(
            f"{self.address_string()} [{self.log_date_time_string()}] {format_string % args}\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Serve the Farpoint remote data platform.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get(
            "FARPOINT_DASHBOARD_HOST",
            "0.0.0.0",
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(
            os.environ.get(
                "FARPOINT_DASHBOARD_PORT",
                "8765",
            )
        ),
    )
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=int(
            os.environ.get(
                "FARPOINT_REGISTRY_SCAN_SECONDS",
                "30",
            )
        ),
    )
    parser.add_argument(
        "--incomplete-timeout",
        type=int,
        default=int(
            os.environ.get(
                "FARPOINT_INCOMPLETE_TIMEOUT_SECONDS",
                "1800",
            )
        ),
    )
    parser.add_argument(
        "--episode-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Read-only root recursively containing Farpoint episodes. "
            "May be specified more than once."
        ),
    )
    parser.add_argument(
        "--campaign-root",
        action="append",
        default=[],
        type=Path,
        help="Read-only root containing collection-campaign.v1 directories.",
    )
    parser.add_argument(
        "--policy-rollout-root",
        action="append",
        default=[],
        type=Path,
        help="Read-only root containing immutable policy rollout evidence.",
    )
    args = parser.parse_args()
    environment_roots = [
        Path(value)
        for value in os.environ.get("FARPOINT_EPISODE_ROOTS", "").split(os.pathsep)
        if value
    ]
    environment_campaign_roots = [
        Path(value)
        for value in os.environ.get("FARPOINT_CAMPAIGN_ROOTS", "").split(os.pathsep)
        if value
    ]
    environment_policy_rollout_roots = [
        Path(value)
        for value in os.environ.get("FARPOINT_POLICY_ROLLOUT_ROOTS", "").split(os.pathsep)
        if value
    ]
    state = PlatformState(
        args.outputs_root,
        max(args.scan_interval, 5),
        max(args.incomplete_timeout, 60),
        episode_roots=[*environment_roots, *args.episode_root],
        campaign_roots=[*environment_campaign_roots, *args.campaign_root],
        policy_rollout_roots=[
            *environment_policy_rollout_roots,
            *args.policy_rollout_root,
        ],
    )
    state.refresh(reports=True)
    watcher = threading.Thread(target=state.watch, daemon=True)
    watcher.start()
    server = ThreadingHTTPServer((args.host, args.port), PlatformHandler)
    server.platform_state = state
    print(f"Farpoint Data Platform listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        state.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()

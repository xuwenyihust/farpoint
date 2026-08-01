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


def build_preview_manifest(episodes_root, episode_id):
    episodes_root = Path(episodes_root).resolve()
    episode_id = unquote(str(episode_id))
    if not episode_id or "/" in episode_id or "\\" in episode_id:
        raise ValueError("invalid episode id")
    episode_dir = (episodes_root / episode_id).resolve()
    if episode_dir.parent != episodes_root or not episode_dir.is_dir():
        raise FileNotFoundError(episode_id)
    frames = sorted((episode_dir / "preview").glob("*.png"))
    encoded_episode = quote(episode_id, safe="")
    return {
        "episode_id": episode_id,
        "frame_count": len(frames),
        "frames": [
            f"/files/episodes/{encoded_episode}/preview/{quote(frame.name, safe='')}"
            for frame in frames
        ],
    }


class PlatformState:
    def __init__(self, outputs_root, scan_interval, incomplete_timeout):
        self.registry = EpisodeRegistry(outputs_root, incomplete_timeout)
        self.retention = RetentionManager(self.registry)
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
                row["preview_url"] = self._artifact_url(row.get("preview_path"), row)
                row["report_url"] = (
                    f"/reports/{row['episode_id']}/index.html"
                    if row.get("report_path")
                    else None
                )
            self._json({"episodes": rows, "count": len(rows)})
        elif (
            parsed.path.startswith("/api/episodes/")
            and parsed.path.endswith("/preview")
        ):
            episode_id = parsed.path.removeprefix("/api/episodes/").removesuffix(
                "/preview"
            )
            try:
                self._json(
                    build_preview_manifest(
                        self.state.registry.layout.episodes,
                        episode_id,
                    )
                )
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
            self._serve_under(
                self.state.registry.layout.episodes,
                parsed.path.removeprefix("/reports/episodes/"),
            )
        elif parsed.path.startswith("/reports/"):
            self._serve_under(
                self.state.registry.layout.reports,
                parsed.path.removeprefix("/reports/"),
            )
        elif parsed.path.startswith("/episodes/"):
            # Existing reports use relative ../../episodes/... asset links.
            self._serve_under(
                self.state.registry.layout.episodes,
                parsed.path.removeprefix("/episodes/"),
            )
        elif parsed.path.startswith("/files/episodes/"):
            self._serve_under(
                self.state.registry.layout.episodes,
                parsed.path.removeprefix("/files/episodes/"),
            )
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

    def _serve_file(self, path):
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header(
            "Cache-Control",
            "public, max-age=86400, immutable"
            if "/observations/" in str(path) or "/preview/" in str(path)
            else "no-cache",
        )
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

    def _security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
        )

    def _artifact_url(self, path, row):
        if not path or not row.get("artifact_path"):
            return None
        try:
            relative = Path(path).resolve().relative_to(
                self.state.registry.layout.episodes.resolve()
            )
        except ValueError:
            return None
        return f"/files/episodes/{relative.as_posix()}"

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
            f"{self.address_string()} [{self.log_date_time_string()}] "
            f"{format_string % args}\n"
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
    args = parser.parse_args()
    state = PlatformState(
        args.outputs_root,
        max(args.scan_interval, 5),
        max(args.incomplete_timeout, 60),
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

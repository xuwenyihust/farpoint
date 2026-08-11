import base64
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import closing
from http.client import HTTPConnection
from pathlib import Path
from urllib.request import urlopen

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_BROWSER_QA = os.environ.get("FARPOINT_RUN_BROWSER_QA") == "1"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAASABIAAD/4QBMRXhpZgAATU0AKgAAAAgAAYdpAAQAAAABAAAAGgAA"
    "AAAAA6ABAAMAAAAB//8AAKACAAQAAAABAAAAAaADAAQAAAABAAAAAQAAAAD/7QA4UGhvdG9z"
    "aG9wIDMuMAA4QklNBAQAAAAAAAA4QklNBCUAAAAAABDUHYzZjwCyBOmACZjs+EJ+/8AACwgA"
    "AQABAQERAP/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMF"
    "BQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYn"
    "KCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SV"
    "lpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz"
    "9PX29/j5+v/bAEMAAgICAgICAwICAwUDAwMFBgUFBQUGCAYGBgYGCAoICAgICAgKCgoKCgoK"
    "CgwMDAwMDA4ODg4ODw8PDw8PDw8PD//dAAQAAf/aAAgBAQAAPwD8A6//2Q=="
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Dashboard server exited before becoming ready")
        try:
            connection = HTTPConnection("127.0.0.1", port, timeout=1)
            connection.request("GET", "/api/health")
            if connection.getresponse().status == 200:
                connection.close()
                return
            connection.close()
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("Dashboard server did not become ready")


@pytest.fixture
def dashboard(tmp_path):
    episode_id = "episode_qa_0001"
    benchmark_id = "dashboard_qa_benchmark"
    display_name = "Dashboard QA Collection"
    episode = tmp_path / "episodes" / episode_id
    write_json(
        episode / "metadata.json",
        {
            "episode_id": episode_id,
            "task_name": "dashboard_qa_pickup",
            "task_type": "qa",
            "episode_seed": 7,
            "benchmark_id": benchmark_id,
            "started_at": "2026-08-01T00:00:00+00:00",
            "finished_at": "2026-08-01T00:00:01+00:00",
        },
    )
    write_json(
        episode / "metrics.json",
        {
            "success": True,
            "task_name": "dashboard_qa_pickup",
            "dataset_valid": True,
            "dataset_observation_count": 2,
        },
    )
    preview = episode / "preview"
    preview.mkdir()
    (preview / "rgb_0000.png").write_bytes(PNG)
    (preview / "rgb_0001.png").write_bytes(PNG)

    benchmark = tmp_path / "benchmarks" / benchmark_id / "manifest.json"
    write_json(
        benchmark,
        {
            "benchmark_id": benchmark_id,
            "task_name": "dashboard_qa_pickup",
            "task_type": "qa",
            "created_at": "2026-08-01T00:00:00+00:00",
            "finished_at": "2026-08-01T00:00:01+00:00",
            "planned_trials": 1,
            "completed_trials": 1,
            "passed_trials": 1,
            "success_rate": 1.0,
            "accepted": True,
            "trials": [{"episode_id": episode_id, "success": True}],
        },
    )

    episode_report = tmp_path / "reports" / episode_id / "index.html"
    benchmark_report = tmp_path / "reports" / "benchmarks" / benchmark_id / "index.html"
    episode_report.parent.mkdir(parents=True)
    benchmark_report.parent.mkdir(parents=True)
    episode_report.write_text("<!doctype html><title>Episode QA report</title>", encoding="utf-8")
    benchmark_report.write_text(
        "<!doctype html><title>Benchmark QA report</title>", encoding="utf-8"
    )
    selection_id = "so101_balanced50_candidate"
    selection_name = "SO-101 Balanced 50 Candidate"
    write_json(
        tmp_path / "benchmarks" / selection_id / "manifest.json",
        {
            "schema_version": "farpoint.collection-selection.v1",
            "collection_id": selection_id,
            "task_id": "so101_cube_pick_place",
            "execution_status": "FINISHED",
            "quality_status": "PASS",
            "required_successes": 50,
            "maximum_attempts": 50,
            "created_at": "2026-08-07T00:00:00+00:00",
            "updated_at": "2026-08-07T00:01:00+00:00",
            "attempts": [
                {"attempt_id": f"attempt_{index}", "success": True}
                for index in range(50)
            ],
        },
    )
    selection_report = tmp_path / "reports" / "benchmarks" / selection_id / "index.html"
    selection_report.parent.mkdir(parents=True)
    selection_report.write_text(
        "<!doctype html><title>SO-101 Balanced 50 report</title>", encoding="utf-8"
    )
    write_json(
        tmp_path / ".data-platform" / "display-names.json",
        {
            "schema_version": "farpoint.display-names.v1",
            "records": {
                benchmark_id: display_name,
                selection_id: selection_name,
            },
        },
    )
    future = time.time() + 60
    os.utime(episode_report, (future, future))
    os.utime(benchmark_report, (future, future))

    external_root = tmp_path / "external-so101"
    collection_id = "so101_dashboard_gate"
    collection = external_root / "gates" / collection_id
    write_json(
        collection / "manifest.json",
        {"schema_version": "farpoint.so101-gate-manifest.v1", "gate_id": collection_id},
    )

    def make_so101_episode(episode_id, success, include_metrics=True):
        external_episode = collection / "episodes" / episode_id
        write_json(
            external_episode / "metadata.json",
            {
                "schema_version": "farpoint.episode.v3",
                "identity": {
                    "episode_id": episode_id,
                    "trial_id": f"trial_{episode_id}",
                    "task_id": "so101_cube_pick_place",
                    "split": "validation",
                    "episode_seed": 99,
                },
                "provenance": {"created_at": "2026-08-06T00:00:00+00:00"},
                "task": {"task_id": "so101_cube_pick_place"},
                "scene": {
                    "entities": [
                        {
                            "entity_id": "pick_object",
                            "role": "manipulated_object",
                            "entity_type": "cube",
                            "asset_id": "procedural_cube_v1",
                            "pose": {"position_m": [0.15, -0.11, 0.052]},
                            "geometry": {"dimensions_m": [0.04, 0.04, 0.04]},
                            "physics": {"body_type": "dynamic", "mass_kg": 0.04},
                        },
                        {
                            "entity_id": "placement_target",
                            "role": "placement_target",
                            "entity_type": "pad",
                            "asset_id": "green_pad_v1",
                            "pose": {"position_m": [0.2, 0.1, 0.037]},
                            "geometry": {"dimensions_m": [0.16, 0.14, 0.01]},
                            "physics": {"body_type": "static"},
                            "regions": [{"relation": "on", "geometry": {"dimensions_m": [0.16, 0.14, 0.01]}}],
                        },
                    ]
                },
                "variation": {
                    "variation_id": "cube_30mm_position_01",
                    "split": "validation",
                    "varied_axes": ["entities.pick_object.pose.position_m"],
                    "requested": {"entities": {"placement_target": {"entity_type": "pad", "pose": {"position_m": [0.2, 0.1, 0.037]}}}},
                    "resolved": {"entities": {"placement_target": {"entity_type": "pad", "pose": {"position_m": [0.200000003, 0.100000001, 0.037]}}}},
                },
                "recording": {
                    "frame_count": 2,
                    "cameras": ["observation.images.front"],
                },
                "outcome": {
                    "success": success,
                    "dataset_valid": success,
                    "failure_category": None if success else "grasp",
                    "failure_reason": None if success else "contact not established",
                },
            },
        )
        if include_metrics:
            write_json(
                external_episode / "metrics.json",
                {
                    "success": success,
                    "dataset_valid": success,
                    "observation_count": 2,
                    "failure_category": None if success else "grasp",
                    "failure_reason": None if success else "contact not established",
                },
            )
        rgb = external_episode / "rgb"
        rgb.mkdir()
        (rgb / "front_000000.png").write_bytes(PNG)
        (rgb / "front_000001.png").write_bytes(PNG)
        return external_episode

    so101_success = make_so101_episode("episode_so101_success", True)
    so101_failure = make_so101_episode("episode_so101_failure", False)
    so101_incomplete = make_so101_episode(
        "episode_so101_incomplete", False, include_metrics=False
    )
    old = time.time() - 3600
    os.utime(so101_incomplete, (old, old))

    campaign_id = "so101_v010_dashboard_live"
    campaign = tmp_path / "campaigns" / campaign_id
    write_json(
        campaign / "campaign.json",
        {
            "campaign_id": campaign_id,
            "campaign_version": "0.1.0",
            "task_id": "so101_cube_pick_place",
            "target": {"successful_episodes": 200},
        },
    )
    write_json(
        campaign / "status.json",
        {
            "campaign_id": campaign_id,
            "segment_id": "segment-000",
            "execution_status": "RUNNING",
            "quality_status": "NOT_EVALUATED",
            "heartbeat_unix": time.time(),
            "updated_unix": time.time(),
            "completed_attempts": 3,
            "successful_episodes": 2,
        },
    )
    (campaign / "active-preview.jpg").write_bytes(JPEG)
    (campaign / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "farpoint.collection-event.v1",
                "campaign_id": campaign_id,
                "segment_id": "segment-000",
                "sequence": 0,
                "timestamp_unix": time.time(),
                "event_type": "heartbeat",
                "payload": {"alive": True},
            }
        )
        + "\n"
    )

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "data_platform_server.py"),
            "--outputs-root",
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--scan-interval",
            "3600",
            "--incomplete-timeout",
            "60",
            "--episode-root",
            str(external_root),
            "--campaign-root",
            str(tmp_path / "campaigns"),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_server(port, process)
        yield {
            "url": url,
            "episode_id": episode_id,
            "benchmark_id": benchmark_id,
            "display_name": display_name,
            "selection_id": selection_id,
            "selection_name": selection_name,
            "so101_success": so101_success.name,
            "so101_failure": so101_failure.name,
            "so101_incomplete": so101_incomplete.name,
            "collection_id": collection_id,
            "campaign_id": campaign_id,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(not RUN_BROWSER_QA, reason="set FARPOINT_RUN_BROWSER_QA=1")
def test_dashboard_navigation_preview_and_mobile_layout(dashboard):
    playwright = pytest.importorskip("playwright.sync_api")
    errors = []
    with playwright.sync_playwright() as engine:
        browser = engine.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on(
            "console",
            lambda message: errors.append(message.text) if message.type == "error" else None,
        )
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(dashboard["url"], wait_until="networkidle")
        page.get_by_role("button", name="Live Runs").click()
        page.locator("#liveRunRows").get_by_text(
            dashboard["campaign_id"], exact=True
        ).wait_for()
        playwright.expect(page.locator("#liveRunRows img")).to_have_js_property(
            "naturalWidth", 1
        )
        page.get_by_role("button", name="Collections").click()
        collection_row = page.locator(
            "#collectionRows tr", has_text=dashboard["campaign_id"]
        )
        assert collection_row.get_by_text("2 / 200", exact=True).count() == 1
        page.get_by_role("button", name="Episodes").click()
        page.get_by_placeholder("Search episode or task").fill("dashboard_qa")
        page.get_by_role("link", name=dashboard["episode_id"]).wait_for()

        page.get_by_role("button", name=f"Play preview for {dashboard['episode_id']}").click()
        page.get_by_text("2 preview frames").wait_for()
        playwright.expect(page.locator("#playerImage")).to_have_js_property(
            "naturalWidth", 1
        )
        page.get_by_role("button", name="Close playback").click()

        page.get_by_placeholder("Search episode or task").fill("episode_so101")
        page.get_by_text(dashboard["so101_success"], exact=True).wait_for()
        page.get_by_text(dashboard["so101_failure"], exact=True).wait_for()
        page.get_by_text(dashboard["so101_incomplete"], exact=True).wait_for()
        page.get_by_role(
            "button", name=f"Play preview for {dashboard['so101_success']}"
        ).click()
        page.get_by_text("2 preview frames").wait_for()
        player_source = page.locator("#playerImage").get_attribute("src")
        assert any(
            frame in player_source
            for frame in (
                "/rgb/front_000000.png",
                "/rgb/front_000001.png",
            )
        )
        page.get_by_role("button", name="Close playback").click()
        page.get_by_role(
            "button", name=f"View metadata for {dashboard['so101_success']}"
        ).click()
        page.get_by_text("Manipulated object", exact=True).wait_for()
        page.get_by_text("Placement target", exact=True).wait_for()
        page.get_by_text("Requested entities", exact=True).wait_for()
        page.get_by_text("Resolved entities", exact=True).wait_for()
        assert page.get_by_text("0.200000003", exact=False).count() >= 1
        page.get_by_role("button", name="Close metadata").click()

        page.locator("#collectionFilter").fill(dashboard["collection_id"])
        page.locator("#splitFilter").select_option("validation")
        assert page.locator("#episodeRows tr").count() == 3
        failure_row = page.locator(
            "#episodeRows tr", has_text=dashboard["so101_failure"]
        )
        incomplete_row = page.locator(
            "#episodeRows tr", has_text=dashboard["so101_incomplete"]
        )
        assert failure_row.get_by_text("FAIL", exact=True).count() == 1
        assert failure_row.get_by_text("grasp", exact=True).count() == 1
        assert incomplete_row.get_by_text("INCOMPLETE", exact=True).count() == 1

        page.get_by_role("button", name="Benchmarks").click()
        benchmark = page.get_by_role("link", name=dashboard["display_name"])
        benchmark.wait_for()
        assert benchmark.get_attribute("href") == (
            f"/reports/benchmarks/{dashboard['benchmark_id']}/index.html"
        )
        selection = page.get_by_role("link", name=dashboard["selection_name"])
        selection.wait_for()
        selection_row = page.locator("#benchmarkRows tr", has=selection)
        assert selection_row.get_by_text("COLLECTION", exact=True).count() == 1
        assert selection_row.get_by_text("PASS", exact=True).count() == 1
        assert selection_row.get_by_text("50 / 50", exact=True).count() == 1
        assert selection.get_attribute("href") == (
            f"/reports/benchmarks/{dashboard['selection_id']}/index.html"
        )
        page.go_back(wait_until="networkidle")
        assert page.get_by_placeholder("Search episode or task").input_value() == (
            "episode_so101"
        )
        assert page.locator("#collectionFilter").input_value() == dashboard["collection_id"]
        assert page.locator("#splitFilter").input_value() == "validation"

        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )
        browser.close()
    assert errors == []


def test_campaign_dashboard_apis_and_sse(dashboard):
    with urlopen(f"{dashboard['url']}/api/live-runs") as response:
        live = json.load(response)["live_runs"]
    assert [row["campaign_id"] for row in live] == [dashboard["campaign_id"]]
    assert live[0]["preview_url"].endswith("/active-preview")

    with urlopen(f"{dashboard['url']}/api/collections") as response:
        collections = json.load(response)["collections"]
    assert collections[0]["successful_episodes"] == 2

    with urlopen(f"{dashboard['url']}/api/benchmarks") as response:
        benchmarks = json.load(response)["benchmarks"]
    assert dashboard["campaign_id"] not in {
        row["benchmark_id"] for row in benchmarks
    }

    with urlopen(f"{dashboard['url']}/api/events") as response:
        assert response.headers["Content-Type"] == "text/event-stream"
        body = response.read().decode()
    assert "data:" in body
    assert dashboard["campaign_id"] in body

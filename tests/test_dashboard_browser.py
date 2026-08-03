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

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_BROWSER_QA = os.environ.get("FARPOINT_RUN_BROWSER_QA") == "1"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
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
    write_json(
        tmp_path / ".data-platform" / "display-names.json",
        {
            "schema_version": "farpoint.display-names.v1",
            "records": {benchmark_id: display_name},
        },
    )
    future = time.time() + 60
    os.utime(episode_report, (future, future))
    os.utime(benchmark_report, (future, future))

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
        page.get_by_role("button", name="Episodes").click()
        page.get_by_placeholder("Search episode or task").fill("dashboard_qa")
        page.get_by_role("link", name=dashboard["episode_id"]).wait_for()

        page.get_by_role("button", name=f"Play preview for {dashboard['episode_id']}").click()
        page.get_by_text("2 preview frames").wait_for()
        assert page.locator("#playerImage").evaluate("image => image.naturalWidth") == 1
        page.get_by_role("button", name="Close playback").click()

        page.get_by_role("button", name="Benchmarks").click()
        benchmark = page.get_by_role("link", name=dashboard["display_name"])
        benchmark.wait_for()
        assert benchmark.get_attribute("href") == (
            f"/reports/benchmarks/{dashboard['benchmark_id']}/index.html"
        )
        page.go_back(wait_until="networkidle")
        assert page.get_by_placeholder("Search episode or task").input_value() == "dashboard_qa"

        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )
        browser.close()
    assert errors == []

import json
import os
import re
import shutil
import socket
import time
from multiprocessing import Process
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import pytest


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_http(url: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2):
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Server did not become ready: {last_error}")


def _run_flask_app(env: dict[str, str], port: int) -> None:
    os.environ.update(env)
    import app as app_module

    app_module.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_e2e_ui_flow(tmp_path, monkeypatch):
    """Run the pipeline, launch the UI, and exercise a basic tagging flow."""
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = repo_root / "test_vids"
    if not input_dir.exists():
        pytest.skip("test_vids dataset not available")

    pytest.importorskip("ffmpeg")
    try:
        import face_recognition  # noqa: F401
    except Exception:
        pytest.skip("face_recognition unavailable for end-to-end scan")

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary not available on PATH")

    pytest.importorskip(
        "playwright.sync_api",
        reason="Playwright not installed; install with `uv add playwright` or `pip install playwright`",
    )
    from playwright.sync_api import sync_playwright

    work_dir = tmp_path / "e2e_work"
    video_dir = work_dir / "videos"
    shutil.copytree(input_dir, video_dir)

    monkeypatch.setenv("INDEXIUM_VIDEO_DIR", str(video_dir))
    monkeypatch.setenv("INDEXIUM_DB", str(work_dir / "faces.db"))
    monkeypatch.setenv("CPU_CORES", "1")
    monkeypatch.setenv("METADATA_PLAN_WORKERS", "1")
    monkeypatch.setenv("DBSCAN_MIN_SAMPLES", "1")

    import e2e_test

    e2e_test.run_pipeline(str(video_dir), str(work_dir))

    db_path = work_dir / "faces.db"
    assert db_path.exists()

    server_env = {
        "INDEXIUM_VIDEO_DIR": str(video_dir),
        "INDEXIUM_DB": str(db_path),
        "CPU_CORES": "1",
        "METADATA_PLAN_WORKERS": "1",
    }
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    server = Process(target=_run_flask_app, args=(server_env, port), daemon=True)
    server.start()
    try:
        _wait_for_http(f"{base_url}/")
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"Playwright browser unavailable: {exc}")

            page = browser.new_page()
            page.goto(f"{base_url}/", wait_until="domcontentloaded")

            if "/group/" in page.url:
                page.fill("#person_name", "UI Test Person")
                page.click("#save-name")
                page.wait_for_load_state("domcontentloaded")

            page.goto(f"{base_url}/people", wait_until="domcontentloaded")
            has_test_person = page.locator("text=UI Test Person").count() > 0
            has_seed_person = page.locator("text=Test Person").count() > 0
            assert has_test_person or has_seed_person

            page.goto(f"{base_url}/metadata_preview", wait_until="domcontentloaded")
            page.wait_for_selector("[data-select-item]", timeout=15000)
            checkbox = page.locator("input[data-select-item]:not([disabled])").first
            if checkbox.count() == 0:
                pytest.skip("No writable metadata items available")
            checkbox.check()
            page.click("#write-selected-btn")
            page.wait_for_url(re.compile(".*/metadata_progress.*"), timeout=15000)

            parsed = urlparse(page.url)
            operation_id = parse_qs(parsed.query).get("operation_id", [None])[0]
            assert operation_id is not None

            timeout = time.monotonic() + 60
            status = None
            while time.monotonic() < timeout:
                status = _fetch_json(f"{base_url}/api/metadata/operations/{operation_id}")
                if status.get("status") in {"completed", "cancelled"}:
                    break
                time.sleep(0.5)

            assert status is not None
            assert status.get("status") in {"completed", "in_progress", "cancelled"}
            browser.close()
    finally:
        server.terminate()
        server.join(timeout=5)

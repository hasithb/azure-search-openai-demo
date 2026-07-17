"""
Deployed agentic-retrieval probe.

Phase 1 — Opens a headed Chromium with a persistent profile.
         If no session exists, it waits for you to sign in manually in the browser.
Phase 2 — Once authenticated, fires a /chat/stream request with
         use_agentic_knowledgebase=true and captures the response.

Usage:
    python scripts/_tmp_playwright_deployed_agentic.py
"""
import json
import os
import re
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".azure" / "cpr-rag" / ".env"
PROFILE_DIR = ROOT / "scripts" / "_tmp_pw_profile"
OUTPUT_PATH = ROOT / "scripts" / "_tmp_playwright_deployed_agentic_result.json"
SCREENSHOT_PATH = ROOT / "scripts" / "_tmp_playwright_deployed_agentic.png"

for raw in ENV_PATH.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    os.environ[key] = value

APP_URL = os.environ["BACKEND_URI"].rstrip("/")
QUESTION = "What is the test for relief from sanctions under CPR 3.9?"

PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# Clean stale lock files from previous crashes
for lock_file in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
    lock_path = PROFILE_DIR / lock_file
    if lock_path.exists():
        lock_path.unlink()


def log(msg):
    print(f"[probe] {msg}", flush=True)


def is_logged_in(page):
    """Check if the 'Sign In' button is absent (meaning user is logged in)."""
    sign_in = page.get_by_role("button", name=re.compile("sign in", re.I))
    return sign_in.count() == 0 or not sign_in.first.is_visible()


with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        str(PROFILE_DIR),
        headless=False,
        viewport={"width": 1440, "height": 960},
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(APP_URL, wait_until="networkidle")
    log(f"Opened {APP_URL}")

    attempts = []

    # Fetch app config
    auth_setup = page.evaluate("async () => { const r = await fetch('/auth_setup'); return r.json(); }")
    config = page.evaluate("async () => { const r = await fetch('/config'); return r.json(); }")
    log(f"useLogin={auth_setup.get('useLogin')}, unauthAccess={auth_setup.get('enableUnauthenticatedAccess')}, agenticOption={config.get('showAgenticRetrievalOption')}")
    attempts.append({"config": {"useLogin": auth_setup.get("useLogin"), "showAgentic": config.get("showAgenticRetrievalOption")}})

    # Wait for React to render
    page.wait_for_timeout(5000)

    # ── Phase 1: ensure logged in ──
    if not is_logged_in(page):
        log("Not logged in. Please complete sign-in in the browser window.")
        log("The script will wait up to 5 minutes for you to finish.")
        deadline = time.time() + 300
        last_reload = time.time()
        while time.time() < deadline:
            page.wait_for_timeout(3000)
            if is_logged_in(page):
                break
            # Only reload occasionally (every 30s) after the initial wait,
            # to avoid killing the MSAL popup flow
            if time.time() - last_reload > 30:
                try:
                    page.reload(wait_until="networkidle")
                    page.wait_for_timeout(3000)
                    last_reload = time.time()
                except Exception:
                    pass
                if is_logged_in(page):
                    break
        if not is_logged_in(page):
            result = {"status": "login_timeout", "message": "Login was not completed within 5 minutes.", "attempts": attempts}
            OUTPUT_PATH.write_text(json.dumps(result, indent=2))
            page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
            log("Login timed out.")
            context.close()
            raise SystemExit(1)

    log("Logged in! Proceeding to agentic retrieval test.")
    attempts.append({"status": "logged_in"})

    # ── Phase 2: fire the agentic chat request ──
    textarea = page.locator("textarea").first
    textarea.wait_for(state="visible", timeout=30000)
    textarea.fill(QUESTION)
    log(f"Filled question: {QUESTION}")

    # Intercept the outgoing /chat request to capture it
    started = time.time()
    try:
        with page.expect_response(
            lambda r: "/chat" in r.url and r.request.method == "POST",
            timeout=180000,
        ) as response_info:
            textarea.press("Enter")

        response = response_info.value
        elapsed = round(time.time() - started, 3)
        response_text = response.text()
        request_body = response.request.post_data or ""

        parsed_request = None
        if request_body:
            try:
                parsed_request = json.loads(request_body)
            except json.JSONDecodeError:
                parsed_request = request_body[:4000]

        final_result = {
            "status": response.status,
            "elapsed_s": elapsed,
            "current_url": page.url,
            "request_url": response.url,
            "authorization_header_present": bool(response.request.headers.get("authorization")),
            "use_agentic_knowledgebase": (parsed_request or {}).get("context", {}).get("overrides", {}).get("use_agentic_knowledgebase"),
            "request_json": parsed_request,
            "response_text": response_text[:12000],
            "attempts": attempts,
        }
        log(f"Response: status={response.status}, elapsed={elapsed}s, agentic={final_result['use_agentic_knowledgebase']}")

        # Wait a moment to let the answer render, then screenshot
        page.wait_for_timeout(5000)

    except Exception as exc:
        final_result = {
            "status": "exception",
            "message": str(exc),
            "current_url": page.url,
            "attempts": attempts,
        }
        log(f"Exception: {exc}")

    OUTPUT_PATH.write_text(json.dumps(final_result, indent=2))
    page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
    log(f"Results written to {OUTPUT_PATH}")
    print(json.dumps(final_result, indent=2), flush=True)
    context.close()

"""
Save Browser Auth State
=======================
Opens a Chromium browser window pointing at the target app URL and waits for
you to log in (e.g. via Microsoft Easy Auth). Once you confirm you are logged
in, it saves the browser cookies and local-storage state to a JSON file that
can be passed to the computer use test harness with --storage-state.

Usage:
    source .venv/bin/activate
    python scripts/save_browser_auth.py

    # Custom output file or URL:
    python scripts/save_browser_auth.py --output scripts/computer_use_auth.json \
        --url https://capps-backend-gz2m4s637t5me.nicedune-921e8a19.uksouth.azurecontainerapps.io/

Then run computer use tests with the saved state:
    python scripts/computer_use_test.py \
        --storage-state scripts/computer_use_auth.json \
        --suite detailed-citations
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

DEFAULT_URL = os.getenv("COMPUTER_USE_TARGET_URL", "http://localhost:50505")
DEFAULT_OUTPUT = str(REPO_ROOT / "scripts" / "computer_use_auth.json")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Capture browser auth state for computer use tests")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"App URL to log into (default: {DEFAULT_URL})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output path for the auth JSON (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening browser at: {args.url}")
    print("Log in with your Microsoft account when the browser opens.")
    print("Once you are on the main app page, press Enter here to save the session.")
    print()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, args=["--window-size=1440,900"])
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        await page.goto(args.url, wait_until="domcontentloaded")

        # Wait for the user to complete login
        input(">>> Press Enter once you are logged in and can see the app chat interface...")

        # Save the authenticated session
        await context.storage_state(path=str(output_path))
        print(f"\nSession saved to: {output_path}")
        print(f"\nNow run:")
        print(f"  python scripts/computer_use_test.py --storage-state {output_path} --suite detailed-citations")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())

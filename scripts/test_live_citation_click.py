#!/usr/bin/env python3
"""Smoke test the built app's citation click flow against a running localhost instance.

This script exercises the real built frontend served by the backend, rather than the
snapshot-mocked Playwright tests. It is intended as a low-risk diagnostic/regression
check for the citation -> Supporting Content -> highlighted subsection flow.

Run with:
  /Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2/.venv-upgrade/bin/python scripts/test_live_citation_click.py

Requires:
  - App running locally at http://localhost:50505
  - Playwright browser installed
"""

from __future__ import annotations

import asyncio
import os
import sys

from playwright.async_api import async_playwright

APP_URL = os.getenv("APP_URL", "http://localhost:50505")
QUESTION = os.getenv("CITATION_SMOKE_QUESTION", "What is CPR Part 1 and the overriding objective?")
CITATION_SELECTOR = "a.citation, button.citation, a[class*='citation'], button[class*='citation']"


async def wait_for_answer(page) -> None:
    await page.wait_for_selector('[class*="answerContainer"]', timeout=120000)
    stop_streaming = page.get_by_label("Stop streaming")
    for _ in range(120):
        generating = await page.locator('text="Generating answer"').count()
        still_streaming = await stop_streaming.count()
        if generating == 0 and still_streaming == 0:
            return
        await page.wait_for_timeout(1000)


async def wait_for_citation_badges(page) -> int:
    for _ in range(10):
        count = await page.locator(CITATION_SELECTOR).count()
        if count > 0:
            return count
        await page.wait_for_timeout(1000)
    return await page.locator(CITATION_SELECTOR).count()


async def select_all_sources(page) -> None:
    dropdown = page.locator('button[role="combobox"]').first
    await dropdown.click()
    option = page.locator('[role="menuitemcheckbox"]:has-text("All Sources")').first
    await option.click()
    await page.keyboard.press("Escape")


async def main() -> int:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})

        try:
            await page.goto(APP_URL, wait_until="networkidle")
            await select_all_sources(page)

            field = page.locator('textarea, input[type="text"]').first
            await field.fill(QUESTION)
            await field.press("Enter")
            await wait_for_answer(page)

            citation = page.locator(CITATION_SELECTOR).first
            citation_count = await wait_for_citation_badges(page)
            if citation_count == 0:
                inline_count = await page.locator('.supContainer').count()
                answer_text = await page.locator('[class*="answerText"]').first.inner_text()
                print(f"inline_citation_count: {inline_count}")
                print(f"answer_text_prefix: {answer_text[:400]!r}")
                print("FAIL: no citation badges were rendered")
                return 1

            citation_text = await citation.inner_text()
            print(f"citation_text: {citation_text!r}")

            support_button = page.get_by_label("Show supporting content")
            support_button_visible = await support_button.count() > 0
            print(f"support_button_present: {support_button_visible}")

            await citation.click()
            await page.wait_for_timeout(2000)

            supporting_count = await page.locator('text="Supporting Content"').count()
            highlight_count = await page.locator('#highlighted-subsection').count()

            print(f"supporting_count: {supporting_count}")
            print(f"highlight_count: {highlight_count}")

            if supporting_count == 0:
                print("FAIL: citation click did not open the Supporting Content panel in the built app")
                return 1

            if highlight_count == 0:
                print("FAIL: Supporting Content opened but no highlighted subsection anchor was created")
                return 1

            print("PASS: citation click opened Supporting Content and created a subsection highlight")
            return 0
        finally:
            await browser.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
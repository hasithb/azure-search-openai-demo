#!/usr/bin/env python3
"""Playwright-based screenshot test for visual verification of prompt improvements.

Takes screenshots of the running app after asking various test questions,
saves them for manual review of:
- Citation rendering
- Source metadata display
- Search depth info in answers
- Disambiguation behavior
- Supporting content panel

Requires: playwright (playwright install chromium)
"""

import asyncio
import os
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright

TARGET_URL = os.getenv("COMPUTER_USE_TARGET_URL", "http://localhost:50505")
OUTPUT_DIR = Path(__file__).parent / "screenshots"


async def wait_for_answer(page, timeout_ms=120000):
    """Wait for the answer to appear after submitting a question."""
    try:
        # Wait for the answer container to appear
        await page.wait_for_selector('[class*="answerContainer"]', timeout=timeout_ms)
        # Wait for the "Generating answer" text to disappear (answer is complete)
        # Poll until either citations appear or "Generating" text is gone
        for _ in range(60):  # up to 60 seconds
            generating = await page.locator('text="Generating answer"').count()
            if generating == 0:
                break
            await asyncio.sleep(1)
        # Extra wait for rendering
        await asyncio.sleep(2)
    except Exception:
        # Fallback: just wait a fixed time
        await asyncio.sleep(15)


async def select_source(page, category: str = "All Sources"):
    """Select a source from the Fluent UI multiselect dropdown."""
    try:
        # The Source dropdown is a button[role=combobox] with text "Source"
        dropdown_trigger = page.locator('button[role="combobox"]').first
        if await dropdown_trigger.is_visible(timeout=3000):
            await dropdown_trigger.click()
            await asyncio.sleep(1)

            # Options are rendered as role="menuitemcheckbox" in Fluent UI
            option = page.locator(f'[role="menuitemcheckbox"]:has-text("{category}")').first
            if await option.is_visible(timeout=3000):
                await option.click()
                await asyncio.sleep(0.5)

                # Close the dropdown
                await page.keyboard.press("Escape")
                await asyncio.sleep(0.3)
                return True
            else:
                print(f"  ⚠ Option '{category}' not found in dropdown")
                await page.keyboard.press("Escape")
        else:
            print("  ⚠ Source dropdown button not found")
    except Exception as e:
        print(f"  ⚠ Source selection failed: {e}")
    return False


async def ask_question(page, question: str, category: str = ""):
    """Type a question and submit it."""
    # Find the input field and type the question
    input_field = page.locator('textarea, input[type="text"]').first
    await input_field.fill(question)
    await asyncio.sleep(0.3)

    # Submit with Enter
    await input_field.press("Enter")

    # Wait for the answer
    await wait_for_answer(page)


async def clear_chat(page):
    """Click the 'Clear chat' button."""
    try:
        clear_btn = page.get_by_text("Clear chat").first
        if await clear_btn.is_visible(timeout=2000):
            await clear_btn.click()
            await asyncio.sleep(1)
            return True
    except Exception:
        pass
    return False


async def screenshot(page, name: str, full_page: bool = False):
    """Take a screenshot and save it."""
    filepath = OUTPUT_DIR / f"{name}.png"
    await page.screenshot(path=str(filepath), full_page=full_page)
    print(f"  📸 Saved: {filepath.name}")
    return filepath


async def run_tests():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        print(f"Opening {TARGET_URL}...")
        await page.goto(TARGET_URL, wait_until="networkidle")
        await asyncio.sleep(2)

        # Screenshot 1: Initial page load
        await screenshot(page, "01_initial_load")
        print("✓ Initial page loaded")

        # Select "All Sources" from the dropdown before asking questions
        if await select_source(page, "All Sources"):
            print("✓ 'All Sources' selected")
            await asyncio.sleep(1)
        else:
            print("⚠ Could not select 'All Sources' — trying to proceed anyway")

        # ═══ TEST A: Basic CPR question with citations ═══
        print("\n═══ Test A: CPR Part 1 (citations + source attribution) ═══")
        await ask_question(page, "What is CPR Part 1 and the overriding objective?")
        await screenshot(page, "02_cpr_part1_answer")

        # Try to click the first citation
        try:
            citation_link = page.locator('[class*="citation"]').first
            if await citation_link.is_visible(timeout=3000):
                await citation_link.click()
                await asyncio.sleep(2)
                await screenshot(page, "03_cpr_part1_citation_panel")
                print("  ✓ Citation panel opened")
        except Exception as e:
            print(f"  ⚠ Could not click citation: {e}")

        # ═══ TEST B: Disclosure question (disambiguation) ═══
        print("\n═══ Test B: Disclosure (disambiguation behavior) ═══")
        await clear_chat(page)

        await ask_question(page, "Tell me about disclosure requirements")
        await screenshot(page, "04_disclosure_answer")

        # ═══ TEST C: Expert evidence (multi-court source attribution) ═══
        print("\n═══ Test C: Expert evidence (cross-court sources) ═══")
        await clear_chat(page)

        await ask_question(page, "What are the rules about expert evidence across different courts?")
        await screenshot(page, "05_expert_evidence_answer")

        # Try to view supporting content
        try:
            citation_link = page.locator('[class*="citation"]').first
            if await citation_link.is_visible(timeout=3000):
                await citation_link.click()
                await asyncio.sleep(2)
                await screenshot(page, "06_expert_evidence_citation")
        except Exception:
            pass

        # ═══ TEST D: Out-of-scope question ═══
        print("\n═══ Test D: Out-of-scope question ═══")
        await clear_chat(page)

        await ask_question(page, "What is the weather in London today?")
        await screenshot(page, "07_out_of_scope_answer")

        # ═══ TEST E: Pre-action disclosure (query rewrite with PAD) ═══
        print("\n═══ Test E: PAD query rewrite ═══")
        await clear_chat(page)

        await ask_question(page, "What is PAD?")
        await screenshot(page, "08_pad_query_rewrite_answer")

        # ═══ TEST F: Developer Settings panel (search depth visible) ═══
        print("\n═══ Test F: Developer Settings ═══")
        try:
            settings_btn = page.locator('[aria-label*="Developer"]').or_(page.get_by_text("Developer Settings")).first
            if await settings_btn.is_visible(timeout=3000):
                await settings_btn.click()
                await asyncio.sleep(1)
                await screenshot(page, "09_developer_settings")
                print("  ✓ Developer settings opened")
            else:
                print("  ⚠ Developer settings button not found")
        except Exception as e:
            print(f"  ⚠ Could not open developer settings: {e}")

        # ═══ TEST G: Thought process panel ═══
        print("\n═══ Test G: Thought process panel ═══")
        try:
            # Close settings first if open
            settings_btn = page.locator('[aria-label*="Developer"]').or_(page.get_by_text("Developer Settings")).first
            if await settings_btn.is_visible(timeout=1000):
                await settings_btn.click()
                await asyncio.sleep(0.5)
        except Exception:
            pass

        await clear_chat(page)

        await ask_question(page, "What are the costs rules in CPR Part 44?")

        # Try to open thought process
        try:
            thought_btn = page.locator('[aria-label*="thought"]').or_(page.get_by_text("Thought process")).first
            if await thought_btn.is_visible(timeout=3000):
                await thought_btn.click()
                await asyncio.sleep(2)
                await screenshot(page, "10_thought_process")
                print("  ✓ Thought process panel opened")

                # Scroll down in the thought process to see the prompt
                thought_panel = page.locator('[class*="thoughtProcess"], [class*="thought"]').first
                if await thought_panel.is_visible(timeout=2000):
                    await thought_panel.evaluate("el => el.scrollTop = el.scrollHeight")
                    await asyncio.sleep(1)
                    await screenshot(page, "11_thought_process_scrolled")
            else:
                print("  ⚠ Thought process button not found")
        except Exception as e:
            print(f"  ⚠ Could not open thought process: {e}")

        # ═══ TEST H: Supporting content panel ═══
        print("\n═══ Test H: Supporting content panel ═══")
        try:
            support_btn = page.get_by_text("Supporting content").or_(page.locator('[aria-label*="supporting"]')).first
            if await support_btn.is_visible(timeout=3000):
                await support_btn.click()
                await asyncio.sleep(2)
                await screenshot(page, "12_supporting_content")
                print("  ✓ Supporting content panel opened")
            else:
                print("  ⚠ Supporting content button not found")
        except Exception as e:
            print(f"  ⚠ Could not open supporting content: {e}")

        await browser.close()

    print(f"\n{'='*60}")
    print(f"Screenshots saved to: {OUTPUT_DIR}")
    screenshots = sorted(OUTPUT_DIR.glob("*.png"))
    print(f"Total screenshots: {len(screenshots)}")
    for s in screenshots:
        size_kb = s.stat().st_size / 1024
        print(f"  {s.name} ({size_kb:.0f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(run_tests())

"""Deterministic Phase 1 smoke test: drives the live UI with Playwright."""
import asyncio
import sys
from playwright.async_api import async_playwright

URL = "http://localhost:50505"
QUESTION = "What is the overriding objective in CPR Part 1?"


async def main() -> int:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await ctx.new_page()
        await page.goto(URL, wait_until="networkidle")

        # Open dev settings, disable agentic retrieval if visible
        try:
            await page.get_by_role("button", name="Developer settings").click(timeout=3000)
            for label in ["Use agentic retrieval", "Use knowledge agent", "Use Knowledge Base"]:
                try:
                    sw = page.get_by_label(label, exact=False)
                    if await sw.count():
                        if await sw.first.is_checked():
                            await sw.first.click()
                            print(f"Toggled OFF: {label}")
                        break
                except Exception:
                    continue
            await page.keyboard.press("Escape")
        except Exception as e:
            print(f"settings step skipped: {e}")

        # Pick "All Sources" from the Source dropdown (Fluent v9 Dropdown)
        try:
            await page.locator("button:has-text('Source')").first.click(timeout=5000)
            await page.locator("#source-filter-option-all-sources").click(timeout=5000)
            print("Selected: All Sources")
            # Close the dropdown
            await page.keyboard.press("Escape")
        except Exception as e:
            print(f"source select failed: {e}")

        # Find chat input
        textbox = page.get_by_role("textbox").last
        await textbox.fill(QUESTION)
        await textbox.press("Enter")

        # Wait for assistant response containing a numbered citation
        try:
            await page.wait_for_selector("sup, .citation, button:has-text('1')", timeout=120000)
        except Exception:
            pass

        # Wait extra for full stream completion
        await page.wait_for_timeout(8000)

        body_text = await page.locator("main, body").first.inner_text()
        has_citation = any(tok in body_text for tok in ["[1]", "[2]", "[3]"])
        print(f"\nCitations found inline: {has_citation}")
        # Find citation buttons
        cite_buttons = page.locator("button.supContainer, sup button, [class*='supContainer']")
        n = await cite_buttons.count()
        print(f"Citation buttons in DOM: {n}")
        clicked = False
        if n:
            await cite_buttons.first.click()
            await page.wait_for_timeout(2500)
            panel_text = await page.locator("body").inner_text()
            clicked = "supporting" in panel_text.lower() or "citation" in panel_text.lower()
            print(f"Panel opened (text contains supporting/citation): {clicked}")

        await page.screenshot(path="/tmp/phase1_smoke.png", full_page=True)
        print("Screenshot: /tmp/phase1_smoke.png")
        await browser.close()
        return 0 if has_citation else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

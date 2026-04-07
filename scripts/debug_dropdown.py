#!/usr/bin/env python3
"""Quick debug to see what appears when Source dropdown is clicked."""
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

OUTPUT = Path(__file__).parent / "screenshots"


async def main():
    OUTPUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto("http://localhost:50505", wait_until="networkidle")
        await asyncio.sleep(3)

        # Click the Source dropdown
        dropdown = page.locator('button[role="combobox"]').first
        await dropdown.click()
        await asyncio.sleep(2)

        # Take a screenshot with dropdown open
        await page.screenshot(path=str(OUTPUT / "debug_dropdown_open.png"))

        # Get ALL elements that could be dropdown options
        options = await page.evaluate("""() => {
            // Check for portal elements too
            const ALL = document.querySelectorAll('*');
            let optionLike = [];
            for (const el of ALL) {
                const role = el.getAttribute('role');
                if (role === 'option' || role === 'listbox' || role === 'menuitem' || role === 'menuitemcheckbox') {
                    optionLike.push({
                        tag: el.tagName,
                        role: role,
                        text: el.textContent.trim().substring(0, 80),
                        cls: el.className.substring(0, 80),
                        id: el.id
                    });
                }
            }
            return optionLike;
        }""")
        print("Dropdown options:")
        print(json.dumps(options, indent=2))

        # Also check for listbox
        listboxes = await page.evaluate("""() => {
            const lbs = document.querySelectorAll('[role=listbox]');
            return Array.from(lbs).map(lb => ({
                childCount: lb.children.length,
                firstChild: lb.children[0] ? lb.children[0].textContent.trim().substring(0, 50) : 'none',
                cls: lb.className.substring(0, 80)
            }));
        }""")
        print("\nListboxes:")
        print(json.dumps(listboxes, indent=2))

        await browser.close()


asyncio.run(main())

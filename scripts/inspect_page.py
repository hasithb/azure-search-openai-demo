#!/usr/bin/env python3
"""Quick diagnostic to inspect the page's HTML structure."""
import asyncio
import json
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto("http://localhost:50505", wait_until="networkidle")
        await asyncio.sleep(3)

        # Get all combobox/dropdown elements
        combos = await page.evaluate("""() => {
            const combos = document.querySelectorAll('[role=combobox], [class*=dropdown], [class*=Dropdown]');
            let info = [];
            combos.forEach(el => {
                info.push({
                    tag: el.tagName,
                    role: el.getAttribute('role'),
                    cls: el.className.substring(0, 100),
                    text: el.textContent.substring(0, 50),
                    id: el.id
                });
            });
            return info;
        }""")
        print("Combobox/dropdown elements:")
        print(json.dumps(combos, indent=2))

        # Get all buttons
        buttons = await page.evaluate("""() => {
            const btns = document.querySelectorAll('button');
            return Array.from(btns).map(b => ({
                text: b.textContent.trim().substring(0, 50),
                role: b.getAttribute('role'),
                ariaLabel: b.getAttribute('aria-label'),
                cls: b.className.substring(0, 80)
            }));
        }""")
        print("\nAll buttons:")
        print(json.dumps(buttons, indent=2))

        # Get all inputs
        inputs = await page.evaluate("""() => {
            const inp = document.querySelectorAll('input, textarea');
            return Array.from(inp).map(i => ({
                tag: i.tagName,
                type: i.type,
                placeholder: i.placeholder,
                cls: i.className.substring(0, 80)
            }));
        }""")
        print("\nAll inputs:")
        print(json.dumps(inputs, indent=2))

        await browser.close()


asyncio.run(main())

"""Debug white screen: load deployed app, capture console errors and DOM state."""
import asyncio
from playwright.async_api import async_playwright

URL = "https://capps-backend-gz2m4s637t5me.nicedune-921e8a19.uksouth.azurecontainerapps.io"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        console_msgs = []
        page_errors = []

        page.on("console", lambda msg: console_msgs.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        print(f"Navigating to {URL} ...")
        try:
            resp = await page.goto(URL, wait_until="networkidle", timeout=30000)
            print(f"Response: {resp.status if resp else 'None'}")
        except Exception as e:
            print(f"Navigation error: {e}")

        # Wait a bit more for any async JS
        await page.wait_for_timeout(5000)

        # Check DOM state
        root_html = await page.evaluate("document.getElementById('root')?.innerHTML?.substring(0, 2000) || 'ROOT EMPTY'")
        body_html = await page.evaluate("document.body.innerHTML.substring(0, 2000)")
        title = await page.title()

        print(f"\n--- Page title: {title}")
        print(f"\n--- Root innerHTML (first 2000 chars):")
        print(root_html)

        if console_msgs:
            print(f"\n--- Console messages ({len(console_msgs)}):")
            for msg in console_msgs:
                print(f"  {msg}")
        else:
            print("\n--- No console messages")

        if page_errors:
            print(f"\n--- Page errors ({len(page_errors)}):")
            for err in page_errors:
                print(f"  {err}")
        else:
            print("\n--- No page errors")

        # Check for network failures
        print("\n--- Checking failed requests...")

        # Take screenshot
        await page.screenshot(path="scripts/_tmp_white_screen.png")
        print("\nScreenshot saved to scripts/_tmp_white_screen.png")

        await browser.close()

asyncio.run(main())

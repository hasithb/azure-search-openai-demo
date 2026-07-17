"""Debug white screen: load deployed app, track all network requests + 401s."""
import asyncio
from playwright.async_api import async_playwright

URL = "https://capps-backend-gz2m4s637t5me.nicedune-921e8a19.uksouth.azurecontainerapps.io"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        failed_requests = []
        all_requests = []

        def on_response(response):
            status = response.status
            url = response.url
            # Only log non-asset, non-200 responses, or all API calls
            if status >= 400 or "/api/" in url or "/auth" in url or "/config" in url or "/chat" in url:
                all_requests.append(f"  {status} {response.request.method} {url}")

        page.on("response", on_response)
        page.on("requestfailed", lambda req: failed_requests.append(f"  FAILED {req.method} {req.url} {req.failure}"))

        console_msgs = []
        page.on("console", lambda msg: console_msgs.append(f"  [{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: console_msgs.append(f"  [PAGE ERROR] {err}"))

        print(f"Navigating to {URL} ...")
        resp = await page.goto(URL, wait_until="networkidle", timeout=30000)
        print(f"Response: {resp.status if resp else 'None'}")
        await page.wait_for_timeout(3000)

        print(f"\n--- Network responses (errors + API calls): {len(all_requests)}")
        for r in all_requests:
            print(r)

        if failed_requests:
            print(f"\n--- Failed requests: {len(failed_requests)}")
            for r in failed_requests:
                print(r)

        if console_msgs:
            print(f"\n--- Console ({len(console_msgs)}):")
            for msg in console_msgs:
                print(msg)

        # Check for localStorage MSAL state
        msal_keys = await page.evaluate("""() => {
            const keys = [];
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && (key.includes('msal') || key.includes('login'))) {
                    keys.push(key);
                }
            }
            return keys;
        }""")
        print(f"\n--- MSAL localStorage keys: {msal_keys}")

        # Check if Sign In button is visible
        sign_in = await page.query_selector("text=Sign in")
        print(f"--- 'Sign in' button visible: {sign_in is not None}")

        # Check body visibility / opacity issues
        body_style = await page.evaluate("""() => {
            const body = document.body;
            const root = document.getElementById('root');
            const cs = window.getComputedStyle(root);
            return {
                bodyDisplay: window.getComputedStyle(body).display,
                bodyVisibility: window.getComputedStyle(body).visibility,
                rootDisplay: cs.display,
                rootVisibility: cs.visibility,
                rootOpacity: cs.opacity,
                rootHeight: cs.height,
                rootChildCount: root?.children?.length || 0
            };
        }""")
        print(f"--- DOM visibility: {body_style}")

        await browser.close()

asyncio.run(main())

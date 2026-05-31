# CUSTOM: Same-origin reverse proxy for iframe-blocked legal sources.
#
# GET /api/proxy-source?url=<encoded live url>&q=<encoded passage phrase>
#
# Fetches a page from the allowlisted legal sources server-side, strips the
# X-Frame-Options / frame-ancestors CSP headers that block embedding, injects
# a <base> tag (so the page's own assets still load from the real host) and a
# small highlight script that finds and scrolls to the cited passage.
#
# Security constraints (MUST remain intact):
#   - Strict HTTPS allowlist; anything else → 400.
#   - SSRF: private/loopback IPs rejected after DNS resolution.
#   - Redirects re-validated against allowlist at each hop.
#   - Response size capped at 5 MB; timeout 10 s.
#   - No user cookies / auth headers forwarded upstream.

import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse

import aiohttp
from quart import Blueprint, Response, abort, request

logger = logging.getLogger(__name__)

proxy_source_bp = Blueprint("proxy_source", __name__)

# Allowlist: ONLY official UK legal sources permitted.
# Must mirror the blockedDomains list in externalSourceHandler.ts.
ALLOWED_HOSTNAMES: frozenset[str] = frozenset(
    [
        "www.justice.gov.uk",
        "justice.gov.uk",
        "www.legislation.gov.uk",
        "legislation.gov.uk",
        "www.judiciary.gov.uk",
        "judiciary.gov.uk",
        "www.gov.uk",
        "gov.uk",
        "www.bailii.org",
        "bailii.org",
    ]
)

MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB
FETCH_TIMEOUT_SECONDS = 10

# Matches common JavaScript frame-buster patterns.
_FRAME_BUSTER_RE = re.compile(
    r"(top\.location|self\s*!==?\s*top|top\s*!==?\s*self|framebuster|frameBreaker)",
    re.IGNORECASE,
)

# Highlight script injected into the proxied page.
# Reads window.__HIGHLIGHT_PHRASE__, builds a flexible regex that matches
# the phrase with any internal whitespace, then wraps the first match in
# a <mark> and scrolls it into view.
# Key correctness fix: use the regex match's own .index/.length to slice the
# ORIGINAL text node content, not the indices from the normalised string.
_HIGHLIGHT_SCRIPT = r"""
<script>
(function() {
  function run() {
    try {
      var phrase = window.__HIGHLIGHT_PHRASE__;
      if (!phrase || phrase.length < 3) return;
      // Build a flexible regex: collapse phrase whitespace, escape special chars,
      // then allow any whitespace between words so it matches across DOM whitespace.
      var escaped = phrase.trim()
        .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        .replace(/\s+/g, '\\s+');
      var re = new RegExp(escaped, 'i');
      var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
      var node;
      while ((node = walker.nextNode())) {
        var nodeText = node.textContent || '';
        var m = re.exec(nodeText);
        if (!m) continue;
        var parent = node.parentNode;
        if (!parent || parent.nodeName === 'SCRIPT' || parent.nodeName === 'STYLE') continue;
        var before = nodeText.slice(0, m.index);
        var matched = m[0];
        var after = nodeText.slice(m.index + matched.length);
        var mark = document.createElement('mark');
        mark.style.cssText = 'background:#fde68a;color:inherit;border-radius:2px;scroll-margin:120px;padding:0 2px;outline:2px solid #f59e0b';
        mark.textContent = matched;
        var frag = document.createDocumentFragment();
        if (before) frag.appendChild(document.createTextNode(before));
        frag.appendChild(mark);
        if (after) frag.appendChild(document.createTextNode(after));
        parent.replaceChild(frag, node);
        mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
    } catch(e) {}
  }
  // Run after DOM is ready, and once more after a short delay for pages that
  // render content via JavaScript (e.g. hydration, lazy loading).
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
  setTimeout(run, 800);
})();
</script>
"""


def _is_allowed(url: str) -> bool:
    """Return True only if url is https and its hostname is in ALLOWED_HOSTNAMES."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return host in ALLOWED_HOSTNAMES


def _is_private_ip(hostname: str) -> bool:
    """Return True if the hostname resolves to a private/loopback address (SSRF guard)."""
    try:
        addr = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(addr)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except Exception:
        # Can't resolve → treat as unsafe
        return True


def _rewrite_html(html: str, base_url: str, phrase: str) -> str:
    """
    Inject <base href>, phrase variable, and highlight script into the HTML.
    Also neutralize frame-buster scripts.
    """
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    # Build a base href that covers the page path so relative links resolve correctly.
    base_href = urljoin(base_url, "/")

    # Inject <base> as the very first child of <head> (must come before any relative URLs).
    base_tag = f'<base href="{base_href}">'
    if re.search(r"<head[^>]*>", html, re.IGNORECASE):
        html = re.sub(r"(<head[^>]*>)", r"\1" + base_tag, html, count=1, flags=re.IGNORECASE)
    else:
        html = base_tag + html

    # Inject phrase variable + highlight script just before </body>.
    phrase_escaped = phrase.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", "")
    inject = f'<script>window.__HIGHLIGHT_PHRASE__ = "{phrase_escaped}";</script>' + _HIGHLIGHT_SCRIPT
    if re.search(r"</body>", html, re.IGNORECASE):
        # Use lambda replacement so the script content (which contains \s etc.) is
        # never parsed as a regex replacement pattern (Python 3.14 raises PatternError).
        html = re.sub(r"(</body>)", lambda m: inject + m.group(1), html, count=1, flags=re.IGNORECASE)
    else:
        html = html + inject

    # Neutralize frame-busters: comment out any <script> blocks containing frame-buster patterns.
    def neutralize_script(m: re.Match) -> str:
        script_content = m.group(0)
        if _FRAME_BUSTER_RE.search(script_content):
            return "<!-- frame-buster removed by proxy -->"
        return script_content

    html = re.sub(r"<script[^>]*>.*?</script>", neutralize_script, html, flags=re.DOTALL | re.IGNORECASE)

    return html


@proxy_source_bp.route("/api/proxy-source")
async def proxy_source() -> Response:
    """
    Proxy a page from an allowlisted legal source, stripping framing headers and
    injecting a highlight script so the cited passage is visible in-panel.
    """
    url = request.args.get("url", "").strip()
    phrase = request.args.get("q", "").strip()

    if not url:
        abort(400)

    # ── Security: allowlist check ────────────────────────────────────────────
    if not _is_allowed(url):
        logger.warning("proxy_source: rejected non-allowlisted URL: %s", url)
        abort(400)

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    # ── Security: SSRF guard ─────────────────────────────────────────────────
    if _is_private_ip(hostname):
        logger.warning("proxy_source: rejected private IP for host: %s", hostname)
        abort(400)

    # ── Fetch the live page ──────────────────────────────────────────────────
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CivilProcedureCopilot/1.0; +https://github.com/Azure-Samples/azure-search-openai-demo)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-GB,en;q=0.9",
    }
    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, allow_redirects=False) as resp:
                # Follow redirects manually so we can re-validate each hop.
                final_resp = resp
                hops = 0
                while final_resp.status in (301, 302, 303, 307, 308) and hops < 5:
                    location = final_resp.headers.get("Location", "")
                    if not location:
                        break
                    # Resolve relative redirects against the current URL.
                    redirect_url = urljoin(url, location)
                    if not _is_allowed(redirect_url):
                        logger.warning("proxy_source: redirect left allowlist → %s", redirect_url)
                        abort(400)
                    url = redirect_url
                    async with session.get(url, headers=headers, allow_redirects=False) as r:
                        final_resp = r
                        hops += 1

                if final_resp.status != 200:
                    logger.warning("proxy_source: upstream returned %s for %s", final_resp.status, url)
                    abort(502)

                # Size cap: read in chunks.
                chunks: list[bytes] = []
                total = 0
                async for chunk in final_resp.content.iter_chunked(65536):
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        logger.warning("proxy_source: response exceeded size cap for %s", url)
                        abort(502)
                    chunks.append(chunk)

                raw = b"".join(chunks)

    except aiohttp.ClientError as exc:
        logger.warning("proxy_source: fetch error for %s: %s", url, exc)
        abort(502)

    # ── Determine encoding ───────────────────────────────────────────────────
    content_type = ""
    charset = "utf-8"
    if "Content-Type" in (final_resp.headers if hasattr(final_resp, "headers") else {}):
        content_type = final_resp.headers["Content-Type"]
        m = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
        if m:
            charset = m.group(1)

    try:
        html = raw.decode(charset, errors="replace")
    except LookupError:
        html = raw.decode("utf-8", errors="replace")

    # ── Rewrite ──────────────────────────────────────────────────────────────
    html = _rewrite_html(html, url, phrase)

    # ── Build response with safe headers ─────────────────────────────────────
    response = Response(html, content_type="text/html; charset=utf-8")
    # Allow this page to be framed only by our own origin.
    response.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
    # Deliberately do NOT set X-Frame-Options (its presence would re-block).
    # Strip any upstream HSTS / X-Content-Type-Options that might cause issues.
    return response

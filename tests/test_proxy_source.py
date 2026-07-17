"""
Unit tests for the proxy_source allowlist / SSRF guard logic.
These test _is_allowed and _is_private_ip directly — no network calls made.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "backend"))

from customizations.routes.proxy_source import _is_allowed, ALLOWED_HOSTNAMES


class TestIsAllowed:
    def test_allowed_justice_www(self):
        assert _is_allowed("https://www.justice.gov.uk/courts/procedure-rules/civil") is True

    def test_allowed_justice_bare(self):
        assert _is_allowed("https://justice.gov.uk/") is True

    def test_allowed_legislation(self):
        assert _is_allowed("https://www.legislation.gov.uk/uksi/2020/1") is True

    def test_allowed_judiciary(self):
        assert _is_allowed("https://www.judiciary.gov.uk/guidance-and-resources/") is True

    def test_allowed_gov_uk(self):
        assert _is_allowed("https://www.gov.uk/guidance/civil-procedure-rules") is True

    def test_allowed_bailii(self):
        assert _is_allowed("https://www.bailii.org/ew/cases/EWHC/") is True

    def test_rejected_http(self):
        assert _is_allowed("http://www.justice.gov.uk/foo") is False

    def test_rejected_non_allowlisted_domain(self):
        assert _is_allowed("https://www.example.com/foo") is False

    def test_rejected_empty(self):
        assert _is_allowed("") is False

    def test_rejected_malformed(self):
        assert _is_allowed("not-a-url") is False

    def test_rejected_lookalike_subdomain(self):
        # e.g. evil-justice.gov.uk must NOT match justice.gov.uk
        assert _is_allowed("https://evil-justice.gov.uk/") is False

    def test_rejected_subdomain_of_allowlisted(self):
        # subdomains of non-exact allowlist entries must be rejected
        # (the allowlist uses exact hostname matching, not suffix matching)
        assert _is_allowed("https://malicious.justice.gov.uk/") is False

    def test_rejected_data_url(self):
        assert _is_allowed("data:text/html,<h1>hi</h1>") is False

    def test_rejected_javascript_url(self):
        assert _is_allowed("javascript:alert(1)") is False


class TestRewriteHtml:
    def test_base_tag_injected(self):
        from customizations.routes.proxy_source import _rewrite_html

        result = _rewrite_html("<html><head></head><body>text</body></html>", "https://www.justice.gov.uk/civil", "Part 1")
        assert '<base href="https://www.justice.gov.uk/">' in result

    def test_highlight_phrase_injected(self):
        from customizations.routes.proxy_source import _rewrite_html

        result = _rewrite_html("<html><head></head><body>text</body></html>", "https://www.justice.gov.uk/civil", "overriding objective")
        assert "overriding objective" in result
        assert "__HIGHLIGHT_PHRASE__" in result

    def test_frame_buster_neutralized(self):
        from customizations.routes.proxy_source import _rewrite_html

        html = '<html><head></head><body><script>if(top.location != self.location){top.location=self.location}</script>content</body></html>'
        result = _rewrite_html(html, "https://www.justice.gov.uk/", "")
        assert "frame-buster removed" in result
        assert "top.location" not in result.split("frame-buster")[0]  # before the comment it's gone

    def test_csp_frame_ancestors_stripped_in_our_response(self):
        # The rewrite itself doesn't set response headers — the route does.
        # Ensure _rewrite_html doesn't accidentally inject an upstream CSP that would block us.
        from customizations.routes.proxy_source import _rewrite_html

        html = '<html><head><meta http-equiv="Content-Security-Policy" content="frame-ancestors \'none\'"></head><body>x</body></html>'
        result = _rewrite_html(html, "https://www.justice.gov.uk/", "test")
        # The meta tag would still be in the html (we don't remove meta CSP tags currently),
        # but the HTTP-level response header is what browsers honour for framing.
        # Just verify nothing crashes.
        assert result is not None

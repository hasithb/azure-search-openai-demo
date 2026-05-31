// Custom handler for external sources that cannot be embedded in an iframe
// due to X-Frame-Options or CSP headers (e.g. www.justice.gov.uk)

export const isIframeBlocked = (url: string): boolean => {
    if (!url) return false;

    try {
        const hostname = new URL(url).hostname;
        // Add domains known to block iframes
        const blockedDomains = ["www.justice.gov.uk", "justice.gov.uk", "www.legislation.gov.uk", "legislation.gov.uk"];

        return blockedDomains.some(domain => hostname === domain || hostname.endsWith("." + domain));
    } catch (e) {
        return false;
    }
};

/**
 * Pick a short, distinctive phrase from extracted passage content for use as a
 * highlight target. Strips leading breadcrumb annotations like [PART X > Y],
 * then returns the first ~8 words of the actual passage text.
 */
export function pickDistinctivePhrase(content: string | undefined): string {
    if (!content) return "";
    // Strip leading [BREADCRUMB] annotations the backend injects.
    const stripped = content.replace(/^(\[[^\]]*\]\s*)+/, "").trim();
    // Take first ~8 words.
    const words = stripped.split(/\s+/).slice(0, 8).join(" ");
    return words;
}

/**
 * Build a Scroll-To-Text-Fragment URL so a new browser tab lands directly on
 * the cited passage (Chromium/Edge). Degrades to the plain URL on other browsers.
 * See https://developer.mozilla.org/en-US/docs/Web/Text_fragments
 */
export function buildTextFragmentUrl(url: string, content: string | undefined): string {
    const phrase = pickDistinctivePhrase(content);
    if (!phrase || !url) return url;
    try {
        // Strip any existing fragment first.
        const base = url.split("#")[0];
        const encoded = encodeURIComponent(phrase);
        return `${base}#:~:text=${encoded}`;
    } catch {
        return url;
    }
}

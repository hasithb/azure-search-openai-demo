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
 * Remove all backend index noise from a passage string:
 *   - [BREADCRUMB > PATH] annotations (anywhere in the string)
 *   - Markdown headings (## 1.1, # Title, etc.)
 *   - Bare rule/section numbers at the start of a line (1.1, 3.2A, etc.)
 *   - Paragraph labels like (1), (a), (i) at the start of a segment
 * Returns plain, human-readable prose.
 */
export function cleanPassageContent(content: string): string {
    return content
        .replace(/\[[^\]]*\]/g, "") // strip ALL [breadcrumb] annotations
        .replace(/^#{1,6}\s*/gm, "") // strip ## markdown headings
        .replace(/^\s*\d+\.\d+[A-Z]?\s+/gm, "") // strip rule numbers like 1.1 or 3.2A
        .replace(/\s+/g, " ")
        .trim();
}

/**
 * Pick a short, distinctive phrase from extracted passage content for use as a
 * highlight target. Strips all index noise, then returns the first ~8 words of
 * the actual passage text.
 */
export function pickDistinctivePhrase(content: string | undefined): string {
    if (!content) return "";
    const clean = cleanPassageContent(content);
    // Take first ~8 words, skipping any leading (1)/(a) paragraph labels.
    const words = clean
        .replace(/^(\(\w+\)\s*)+/, "") // skip leading (1) (a) etc.
        .trim()
        .split(/\s+/)
        .slice(0, 8)
        .join(" ");
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

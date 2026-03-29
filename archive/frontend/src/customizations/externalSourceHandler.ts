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

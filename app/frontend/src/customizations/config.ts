// Custom Features Configuration
// ==============================
// Enable/disable custom features here without modifying upstream code.
// This file is the single source of truth for frontend feature flags.

export const CUSTOM_FEATURES = {
    // Source filtering feature - adds source dropdown in chat/ask pages
    categoryFilter: true,

    // Citation sanitization - fixes malformed citations like "1. 1" -> "[1]"
    citationSanitizer: true,

    // Legal domain customizations (prompts are handled backend-side)
    legalDomainPrompts: true,

    // Admin/Developer mode - shows developer settings, thought process, and supporting content buttons
    // Set to true for development, false for production end-users
    adminMode: false,

    // Show citations panel at bottom of answers (separate from inline citations which always show)
    showCitationsPanel: true,

    // Answer formatting - adds paragraph breaks to long single-block answers
    answerParagraphs: true,

    // Structured citation matching - use subsection_id and sourcefile metadata for
    // precise SupportingContent matching instead of text-based regex parsing
    structuredCitationMatching: true,

    // Citation metadata display - show subsection, sourcepage, sourcefile, category
    // badges in SupportingContent item headers
    citationMetadataDisplay: false,

    // Preserve subsection boundaries during chunk deduplication — keeps chunks from
    // different subsections of the same document as separate items
    preserveSubsectionBoundaries: true
};

export function isFeatureEnabled(featureName: keyof typeof CUSTOM_FEATURES): boolean {
    return CUSTOM_FEATURES[featureName] ?? false;
}

// Check URL parameter for admin mode override (?admin=true)
export function isAdminMode(): boolean {
    if (typeof window !== "undefined") {
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get("admin") === "true") {
            return true;
        }
    }
    return CUSTOM_FEATURES.adminMode;
}

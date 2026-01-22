// Frontend Customizations Module
// ===============================
// This module contains custom features that are isolated from the main codebase
// to prevent merge conflicts when updating from upstream.
//
// Structure:
// - config.ts: Feature flags
// - citationSanitizer.ts: Citation formatting fixes
// - useCategories.ts: Hook for fetching categories
// - useMobile.ts: Mobile detection and abbreviations
// - DataPrivacyNotice.tsx: Data privacy information panel for users
// - mobile.css: Mobile-responsive styles (imported in index.tsx)
// - __tests__/: Tests for customizations

// Feature configuration
export { CUSTOM_FEATURES, isFeatureEnabled, isAdminMode } from "./config";

// Citation sanitization
export { sanitizeCitations, fixMalformedCitations, collapseAdjacentCitations } from "./citationSanitizer";

// Answer paragraph formatting
export { formatAnswerParagraphs } from "./answerParagraphs";

// Category filtering
export { useCategories } from "./useCategories";
export type { Category } from "./useCategories";

// Mobile detection and abbreviations (source names and responsive hooks)
export { useIsMobile, getAbbreviatedCategory, getDepthLabel, DEPTH_OPTIONS } from "./useMobile";

// Legal Feedback
export { LegalFeedback } from "./LegalFeedback";

// External source handling
export { isIframeBlocked } from "./externalSourceHandler";

// Help & About Panel (replaces DataPrivacyNotice)
export { HelpAboutPanel } from "./HelpAboutPanel";

// Splash Screen (animated intro with morph-to-header effect)
export { SplashScreen } from "./SplashScreen";

// Citation Metadata Utilities
// ===========================
// Extracts structured metadata from data points and builds correct
// citation paths for Supporting Content matching and source-document viewing.
// Part of the merge-safe customizations architecture.

import type { SourceTextItem } from "../api/models";

/**
 * Structured metadata carried alongside a citation for precise
 * SupportingContent matching and UI display.
 */
export interface StructuredCitationMetadata {
    subsectionId: string;
    sourcepage: string;
    sourcefile: string;
    category: string;
    content: string;
    storageUrl: string;
}

/**
 * Extract structured metadata directly from a SourceTextItem data point,
 * avoiding lossy string encoding/re-parsing cycles.
 */
export function extractMetadataFromDataPoint(dp: SourceTextItem | undefined | null): StructuredCitationMetadata {
    if (!dp) {
        return { subsectionId: "", sourcepage: "", sourcefile: "", category: "", content: "", storageUrl: "" };
    }
    return {
        subsectionId: (dp.subsection_id ?? "").trim(),
        sourcepage: (dp.sourcepage ?? "").trim(),
        sourcefile: (dp.sourcefile ?? "").trim(),
        category: (dp.category ?? "").trim(),
        content: (dp.content ?? "").trim(),
        storageUrl: (dp.storageurl ?? "").trim()
    };
}

function normalizeLabelPart(value: string): string {
    return value.trim().replace(/\s+/g, " ").toLowerCase();
}

/**
 * Build a human-readable citation label from structured metadata.
 * This preserves subsection specificity for same-document citations while
 * avoiding repeated sourcepage/sourcefile fragments.
 */
export function buildCitationLabel(metadata: Partial<StructuredCitationMetadata> | undefined | null, fallback = ""): string {
    if (!metadata) {
        return fallback;
    }

    const subsectionId = (metadata.subsectionId ?? "").trim();
    const sourcepage = (metadata.sourcepage ?? "").trim();
    const sourcefile = (metadata.sourcefile ?? "").trim();
    const parts: string[] = [];

    if (subsectionId) {
        parts.push(subsectionId);
    }

    if (sourcepage && normalizeLabelPart(sourcepage) !== normalizeLabelPart(subsectionId)) {
        parts.push(sourcepage);
    }

    if (
        sourcefile &&
        normalizeLabelPart(sourcefile) !== normalizeLabelPart(sourcepage) &&
        normalizeLabelPart(sourcefile) !== normalizeLabelPart(subsectionId)
    ) {
        parts.push(sourcefile);
    }

    return parts.length > 0 ? parts.join(", ") : fallback;
}

/**
 * Build a citation path for Supporting Content interactions.
 * Prefer in-app /content references whenever we have a stable document identifier,
 * and only fall back to the external storageUrl when that is the only available path.
 */
export function buildCitationPath(dp: SourceTextItem | undefined | null): string {
    if (!dp) return "";

    const sourcepage = (dp.sourcepage ?? "").trim();
    if (sourcepage && sourcepage.includes("#page=")) {
        return `/content/${encodeURIComponent(sourcepage)}`;
    }

    const sourcefile = (dp.sourcefile ?? "").trim();
    if (sourcefile) {
        return `/content/${encodeURIComponent(sourcefile)}`;
    }

    if (sourcepage) {
        return `/content/${encodeURIComponent(sourcepage)}`;
    }

    const storageUrl = (dp.storageurl ?? "").trim();
    if (storageUrl) {
        return storageUrl;
    }

    return "";
}

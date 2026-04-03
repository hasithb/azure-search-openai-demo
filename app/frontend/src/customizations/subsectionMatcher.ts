// Subsection Matcher
// ==================
// Precision subsection matching that uses structured metadata fields
// instead of text-based regex parsing. Falls back to existing logic
// when metadata is unavailable.
// Part of the merge-safe customizations architecture.

import type { StructuredCitationMetadata } from "./citationMetadata";

interface MatchableItem {
    subsection_id?: string;
    sourcepage?: string;
    sourcefile?: string;
    content?: string;
    full_content?: string;
    citation?: string;
}

/**
 * Find the best matching item index using structured citation metadata.
 *
 * Scoring priority:
 *   1. Exact subsection_id match on the item's subsection_id field (100)
 *   2. Exact sourcepage match on the item's sourcepage field (90)
 *   3. subsection_id found in item content via word-boundary regex (80)
 *   4. sourcefile match + subsection in content (60)
 *   5. sourcefile-only match (20)
 *
 * Returns -1 when no match meets the minimum threshold.
 */
export function findBestMatch(metadata: StructuredCitationMetadata, items: MatchableItem[]): number {
    if (!items || items.length === 0) return -1;

    const { subsectionId, sourcepage: metaSourcepage, sourcefile: metaSourcefile } = metadata;
    const hasSubsection = subsectionId.length > 0;
    const hasSourcepage = metaSourcepage.length > 0;
    const hasSourcefile = metaSourcefile.length > 0;

    if (!hasSubsection && !hasSourcepage && !hasSourcefile) return -1;

    let bestIndex = -1;
    let bestScore = 0;
    const threshold = 15;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        let score = 0;

        const itemSubsectionId = (item.subsection_id ?? "").trim();
        const itemSourcepage = (item.sourcepage ?? "").trim();
        const itemSourcefile = (item.sourcefile ?? "").trim();
        const itemContent = item.full_content || item.content || "";

        // 1. Exact subsection_id field match
        if (hasSubsection && itemSubsectionId && itemSubsectionId === subsectionId) {
            score += 100;
        }

        // 2. Exact sourcepage match helps disambiguate multiple sections from the same guide.
        if (hasSourcepage && itemSourcepage) {
            const normalizedMetaSourcepage = metaSourcepage.toLowerCase();
            const normalizedItemSourcepage = itemSourcepage.toLowerCase();
            if (
                normalizedMetaSourcepage === normalizedItemSourcepage ||
                normalizedMetaSourcepage.includes(normalizedItemSourcepage) ||
                normalizedItemSourcepage.includes(normalizedMetaSourcepage)
            ) {
                score += 90;
            }
        }

        // 3. subsection_id found in content text
        if (hasSubsection && score < 100 && itemContent) {
            const escaped = subsectionId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
            const pattern = new RegExp(`(^|\\n|\\s)${escaped}(\\s|\\.|,|$)`, "i");
            if (pattern.test(itemContent)) {
                score += 80;
            }
        }

        // 4/5. sourcefile match
        if (hasSourcefile && itemSourcefile) {
            const normalizedMeta = metaSourcefile.toLowerCase();
            const normalizedItem = itemSourcefile.toLowerCase();
            if (normalizedMeta === normalizedItem || normalizedMeta.includes(normalizedItem) || normalizedItem.includes(normalizedMeta)) {
                // sourcefile matches — does subsection also appear in content?
                if (hasSubsection && itemContent) {
                    const escaped = subsectionId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                    if (new RegExp(`\\b${escaped}\\b`, "i").test(itemContent)) {
                        score += 60;
                    } else {
                        score += 20;
                    }
                } else {
                    score += 20;
                }
            }
        }

        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }

    return bestScore >= threshold ? bestIndex : -1;
}

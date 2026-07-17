// Chunk Deduplicator
// ==================
// Subsection-aware deduplication that preserves chunks from different
// subsections of the same document as separate items. Falls back to
// document-level dedup when subsection_id is absent.
// Part of the merge-safe customizations architecture.

import type { SourceTextItem } from "../api/models";

export interface DeduplicatedItem extends SourceTextItem {
    /** Merged full_content when multiple chunks share the same dedup key */
    full_content?: string;
}

/**
 * Build a dedup key that keeps subsection boundaries:
 *   normalize(sourcefile)::subsection_id   (when subsection_id present)
 *   normalize(sourcefile)                  (fallback — original behaviour)
 */
function dedupKey(item: SourceTextItem): string {
    const file = (item.sourcefile ?? "").trim().toLowerCase();
    const subsection = (item.subsection_id ?? "").trim();
    if (subsection) {
        return `${file}::${subsection}`;
    }
    return file || (item.sourcepage ?? "").trim().toLowerCase();
}

/**
 * Deduplicate supporting content items while preserving subsection boundaries.
 *
 * When two chunks share the same sourcefile AND subsection_id, they are merged.
 * When they share sourcefile but have different subsection_ids, they stay separate.
 * When subsection_id is empty for both, falls back to document-level merge (original behaviour).
 */
export function deduplicatePreservingSubsections(items: SourceTextItem[]): DeduplicatedItem[] {
    if (!items || items.length === 0) return [];

    const map = new Map<string, DeduplicatedItem>();
    const order: string[] = [];

    for (const item of items) {
        const key = dedupKey(item);
        const existing = map.get(key);

        if (!existing) {
            map.set(key, { ...item, full_content: item.content ?? "" });
            order.push(key);
        } else {
            // Merge content, avoiding duplicate text
            const newContent = (item.content ?? "").trim();
            if (newContent && !(existing.full_content ?? "").includes(newContent)) {
                existing.full_content = `${existing.full_content}\n\n${newContent}`;
            }
            // Prefer the item with more metadata populated
            if (!existing.sourcepage && item.sourcepage) existing.sourcepage = item.sourcepage;
            if (!existing.category && item.category) existing.category = item.category;
            if (!existing.storageurl && item.storageurl) existing.storageurl = item.storageurl;
        }
    }

    return order.map(key => map.get(key)!);
}

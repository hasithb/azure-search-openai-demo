/**
 * CUSTOM CHUNK DEDUPLICATOR TESTS
 * =================================
 * Tests for subsection-aware deduplication logic.
 * These tests are independent of upstream and will continue to work after merges.
 */

import { describe, expect, it } from "vitest";
import { deduplicatePreservingSubsections } from "../chunkDeduplicator";
import type { SourceTextItem } from "../../api/models";

describe("deduplicatePreservingSubsections", () => {
    it("returns empty array for empty input", () => {
        expect(deduplicatePreservingSubsections([])).toEqual([]);
    });

    it("returns empty array for null-ish input", () => {
        expect(deduplicatePreservingSubsections(null as unknown as SourceTextItem[])).toEqual([]);
    });

    it("returns single item unchanged", () => {
        const items: SourceTextItem[] = [{ sourcefile: "file.pdf", content: "Some content", subsection_id: "CPR-31.2" }];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(1);
        expect(result[0].content).toBe("Some content");
        expect(result[0].full_content).toBe("Some content");
    });

    it("merges chunks with same sourcefile and subsection_id", () => {
        const items: SourceTextItem[] = [
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "First chunk about disclosure" },
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "Second chunk about disclosure" }
        ];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(1);
        expect(result[0].full_content).toContain("First chunk about disclosure");
        expect(result[0].full_content).toContain("Second chunk about disclosure");
    });

    it("keeps chunks with same sourcefile but different subsection_ids separate", () => {
        const items: SourceTextItem[] = [
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "Disclosure section" },
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.6", content: "Standard disclosure" }
        ];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(2);
        expect(result[0].subsection_id).toBe("CPR-31.2");
        expect(result[1].subsection_id).toBe("CPR-31.6");
    });

    it("falls back to document-level merge when subsection_id is absent", () => {
        const items: SourceTextItem[] = [
            { sourcefile: "benefits.pdf", content: "First chunk" },
            { sourcefile: "benefits.pdf", content: "Second chunk" }
        ];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(1);
        expect(result[0].full_content).toContain("First chunk");
        expect(result[0].full_content).toContain("Second chunk");
    });

    it("preserves insertion order", () => {
        const items: SourceTextItem[] = [
            { sourcefile: "alpha.pdf", content: "Alpha content" },
            { sourcefile: "beta.pdf", content: "Beta content" },
            { sourcefile: "gamma.pdf", content: "Gamma content" }
        ];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(3);
        expect(result[0].sourcefile).toBe("alpha.pdf");
        expect(result[1].sourcefile).toBe("beta.pdf");
        expect(result[2].sourcefile).toBe("gamma.pdf");
    });

    it("does not duplicate content on merge when chunks have identical text", () => {
        const items: SourceTextItem[] = [
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "Same content" },
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "Same content" }
        ];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(1);
        // Should not duplicate the text
        expect(result[0].full_content).toBe("Same content");
    });

    it("preserves metadata from first item and fills gaps from subsequent items", () => {
        const items: SourceTextItem[] = [
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "First" },
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "Second", category: "Legal", storageurl: "https://example.com" }
        ];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(1);
        expect(result[0].category).toBe("Legal");
        expect(result[0].storageurl).toBe("https://example.com");
    });

    it("does not overwrite existing metadata on merge", () => {
        const items: SourceTextItem[] = [
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "First", category: "Original" },
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "Second", category: "Overwrite" }
        ];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(1);
        expect(result[0].category).toBe("Original");
    });

    it("handles mixed items with and without subsection_id", () => {
        const items: SourceTextItem[] = [
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "With subsection" },
            { sourcefile: "rules.pdf", content: "Without subsection" },
            { sourcefile: "rules.pdf", subsection_id: "CPR-31.2", content: "Another with same subsection" }
        ];
        const result = deduplicatePreservingSubsections(items);
        // Item 0 and Item 2 share key "rules.pdf::CPR-31.2" → merged
        // Item 1 has key "rules.pdf" → separate
        expect(result).toHaveLength(2);
    });

    it("normalizes sourcefile case for dedup keys", () => {
        const items: SourceTextItem[] = [
            { sourcefile: "Rules.PDF", content: "Upper case" },
            { sourcefile: "rules.pdf", content: "Lower case" }
        ];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(1);
        expect(result[0].full_content).toContain("Upper case");
        expect(result[0].full_content).toContain("Lower case");
    });

    it("falls back to sourcepage when sourcefile is missing", () => {
        const items: SourceTextItem[] = [
            { sourcepage: "page-ref-1", content: "Content A" },
            { sourcepage: "page-ref-1", content: "Content B" }
        ];
        const result = deduplicatePreservingSubsections(items);
        expect(result).toHaveLength(1);
        expect(result[0].full_content).toContain("Content A");
        expect(result[0].full_content).toContain("Content B");
    });
});

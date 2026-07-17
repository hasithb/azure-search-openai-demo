/**
 * CUSTOM SUBSECTION MATCHER TESTS
 * ================================
 * Tests for structured subsection matching using citation metadata.
 * These tests are independent of upstream and will continue to work after merges.
 */

import { describe, expect, it } from "vitest";
import { findBestMatch } from "../subsectionMatcher";
import type { StructuredCitationMetadata } from "../citationMetadata";

function makeMeta(overrides: Partial<StructuredCitationMetadata> = {}): StructuredCitationMetadata {
    return {
        subsectionId: "",
        sourcepage: "",
        sourcefile: "",
        category: "",
        content: "",
        storageUrl: "",
        ...overrides
    };
}

describe("findBestMatch", () => {
    it("returns -1 for empty items array", () => {
        const meta = makeMeta({ subsectionId: "CPR-31.2" });
        expect(findBestMatch(meta, [])).toBe(-1);
    });

    it("returns -1 when metadata has no subsection or sourcefile", () => {
        const meta = makeMeta();
        const items = [{ content: "Some content", sourcefile: "file.pdf" }];
        expect(findBestMatch(meta, items)).toBe(-1);
    });

    it("uses sourcepage to disambiguate sections from the same guide", () => {
        const meta = makeMeta({
            subsectionId: "9.28",
            sourcepage: "The Urgent and Short Applications List, Master's Appointments (p. 74)",
            sourcefile: "King's Bench Division Guide"
        });
        const items = [
            {
                sourcepage: "Annex 6 - Notice of Proposed Allocation to the Multi-Track (p. 232)",
                sourcefile: "King's Bench Division Guide",
                content: "rule 29.1 The notice of proposed allocation..."
            },
            {
                sourcepage: "The Urgent and Short Applications List, Master's Appointments (p. 74)",
                sourcefile: "King's Bench Division Guide",
                content: "9.28 Hearing dates for Masters' appointments..."
            }
        ];

        expect(findBestMatch(meta, items)).toBe(1);
    });

    it("scores exact subsection_id field match highest (100)", () => {
        const meta = makeMeta({ subsectionId: "CPR-31.2" });
        const items = [
            { subsection_id: "CPR-31.1", content: "Different section" },
            { subsection_id: "CPR-31.2", content: "The matching section" },
            { subsection_id: "CPR-31.3", content: "Another section" }
        ];
        expect(findBestMatch(meta, items)).toBe(1);
    });

    it("scores subsection_id in content (80) when no exact field match", () => {
        const meta = makeMeta({ subsectionId: "CPR-31.2" });
        const items = [
            { content: "Unrelated text about evidence rules" },
            { content: "Under CPR-31.2, the court may order disclosure" },
            { content: "Other procedural matters" }
        ];
        expect(findBestMatch(meta, items)).toBe(1);
    });

    it("scores sourcefile + subsection in content (60)", () => {
        const meta = makeMeta({ subsectionId: "Part 31", sourcefile: "civil-procedure-rules.pdf" });
        const items = [
            { sourcefile: "other-doc.pdf", content: "Part 31 unrelated" },
            { sourcefile: "civil-procedure-rules.pdf", content: "Under Part 31 disclosure obligations" },
            { sourcefile: "civil-procedure-rules.pdf", content: "Part 1 overriding objective" }
        ];
        // Item 1: sourcefile match (20) + subsection in content (60) = 80 total? No, it's not additive exactly.
        // Let's trace the logic: item 1 gets score 60 from sourcefile+subsection combo
        // But also gets score 0 from exact subsection_id (no field).
        // And score 0 from subsection in content check (preceded by score < 100 check — score is 0, so this runs).
        // Actually "Part 31" appearing in "Under Part 31 disclosure obligations" → regex needs word boundary.
        // Let's re-check: regex is (^|\n|\s)Part 31(\s|\.|,|$) — "Under Part 31 disclosure" matches
        // So item 1 gets score 80 from subsection-in-content check!
        // Then sourcefile match with subsection in content adds 60 — total 140.
        // Item 0 has wrong sourcefile so no sourcefile points, but "Part 31 unrelated" matches content → score 80.
        // Item 2 has right sourcefile but "Part 1" doesn't match "Part 31", so sourcefile only → 20.
        expect(findBestMatch(meta, items)).toBe(1);
    });

    it("scores sourcefile-only match (20)", () => {
        const meta = makeMeta({ sourcefile: "civil-procedure-rules.pdf" });
        const items = [
            { sourcefile: "other-doc.pdf", content: "Some content" },
            { sourcefile: "civil-procedure-rules.pdf", content: "Some content" }
        ];
        expect(findBestMatch(meta, items)).toBe(1);
    });

    it("returns -1 when no match meets threshold", () => {
        const meta = makeMeta({ subsectionId: "CPR-31.2" });
        const items = [{ content: "Unrelated content about tax law" }, { content: "Criminal procedure matters" }];
        expect(findBestMatch(meta, items)).toBe(-1);
    });

    it("prefers exact subsection_id match over content match", () => {
        const meta = makeMeta({ subsectionId: "CPR-31.2" });
        const items = [{ content: "CPR-31.2 appears directly in the content here" }, { subsection_id: "CPR-31.2", content: "Minimal content" }];
        // Item 0: subsection in content = 80
        // Item 1: exact subsection_id = 100
        expect(findBestMatch(meta, items)).toBe(1);
    });

    it("handles items with undefined fields gracefully", () => {
        const meta = makeMeta({ subsectionId: "CPR-31.2" });
        const items = [{ content: undefined, sourcefile: undefined, subsection_id: undefined }, { subsection_id: "CPR-31.2" }];
        expect(findBestMatch(meta, items)).toBe(1);
    });

    it("uses citation field as additional context", () => {
        const meta = makeMeta({ subsectionId: "CPR-31.2", sourcefile: "rules.pdf" });
        const items = [{ sourcefile: "rules.pdf", content: "CPR-31.2 disclosure", citation: "rules.pdf" }];
        // sourcefile match + subsection in content → 60, plus subsection in content → 80 = total 140
        expect(findBestMatch(meta, items)).toBe(0);
    });

    it("handles special regex characters in subsection_id", () => {
        const meta = makeMeta({ subsectionId: "Rule 31.2(a)" });
        const items = [{ subsection_id: "Rule 31.2(a)", content: "Some content" }];
        expect(findBestMatch(meta, items)).toBe(0);
    });

    it("handles case-insensitive sourcefile matching", () => {
        const meta = makeMeta({ sourcefile: "Civil-Procedure-Rules.PDF" });
        const items = [{ sourcefile: "civil-procedure-rules.pdf", content: "Content" }];
        expect(findBestMatch(meta, items)).toBe(0);
    });
});

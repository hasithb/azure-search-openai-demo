/**
 * CUSTOM CITATION SANITIZER TESTS
 * ================================
 * Tests for the citation sanitization logic.
 * These tests are independent of upstream and will continue to work after merges.
 */

import { describe, expect, it } from "vitest";
import { fixMalformedCitations, collapseAdjacentCitations, sanitizeCitations, findMatchingCitation } from "../citationSanitizer";

describe("fixMalformedCitations", () => {
    it("fixes duplicated number pattern like '1. 1' → '[1]'", () => {
        expect(fixMalformedCitations("The answer is here 1. 1")).toBe("The answer is here [1]");
    });

    it("fixes duplicated pattern at end of sentence", () => {
        expect(fixMalformedCitations("See the disclosure 1. 1 for details")).toBe("See the disclosure [1] for details");
    });

    it("fixes duplicated pattern WITHOUT space like '1.1' → '[1]'", () => {
        expect(fixMalformedCitations("The answer is here 1.1")).toBe("The answer is here [1]");
    });

    it("fixes duplicated no-space pattern at end of text", () => {
        expect(fixMalformedCitations("...at proportionate cost 1.1")).toBe("...at proportionate cost [1]");
    });

    it("fixes bracketed duplicates like '[1].[1]' → '[1]'", () => {
        expect(fixMalformedCitations("The result [1].[1]")).toBe("The result [1]");
    });

    it("fixes bracketed duplicates without period like '[1][1]' → '[1]'", () => {
        expect(fixMalformedCitations("The result [1][1]")).toBe("The result [1]");
    });

    it("fixes bracketed duplicates with space like '[1]. [1]' → '[1]'", () => {
        expect(fixMalformedCitations("The result [1]. [1]")).toBe("The result [1]");
    });

    it("fixes bracketed duplicates with just space like '[1] [1]' → '[1]'", () => {
        expect(fixMalformedCitations("The result [1] [1]")).toBe("The result [1]");
    });

    it("fixes multiple duplicated patterns in same text", () => {
        expect(fixMalformedCitations("First point 1. 1 and second point 2. 2")).toBe("First point [1] and second point [2]");
    });

    it("fixes multiple no-space duplicates in same text", () => {
        expect(fixMalformedCitations("First point 1.1 and second 2.2")).toBe("First point [1] and second [2]");
    });

    it("handles larger citation numbers", () => {
        expect(fixMalformedCitations("Reference 45. 45 in the guide")).toBe("Reference [45] in the guide");
    });

    it("does not modify valid decimal numbers like 3.14", () => {
        expect(fixMalformedCitations("The value is 3.14 and 2.5")).toBe("The value is 3.14 and 2.5");
    });

    it("does not modify non-duplicated patterns like 1.2", () => {
        expect(fixMalformedCitations("Section 1.2 covers this")).toBe("Section 1.2 covers this");
    });

    it("does not modify non-duplicated patterns with space", () => {
        expect(fixMalformedCitations("Section 1. 2 and 3. 4")).toBe("Section 1. 2 and 3. 4");
    });

    it("fixes unbracketed citation at end of text", () => {
        expect(fixMalformedCitations("...the proceedings 1.")).toBe("...the proceedings.[1]");
    });

    it("does not fix unbracketed number mid-sentence", () => {
        // Only fix at true end of text to avoid false positives
        expect(fixMalformedCitations("...the proceedings 2. The next")).toBe("...the proceedings 2. The next");
    });

    it("fixes unbracketed citation at paragraph end", () => {
        expect(fixMalformedCitations("First paragraph ends here 1.")).toBe("First paragraph ends here.[1]");
    });

    it("does not modify section numbers like 3.1 mid-sentence", () => {
        expect(fixMalformedCitations("See section 3.1 for details")).toBe("See section 3.1 for details");
    });

    it("fixes range citations with en-dash", () => {
        expect(fixMalformedCitations("See rules [69–81] for details")).toBe("See rules [69] for details");
    });

    it("fixes range citations with hyphen", () => {
        expect(fixMalformedCitations("See rules [69-81] for details")).toBe("See rules [69] for details");
    });

    it("fixes 'source1' pattern → '[1]'", () => {
        expect(fixMalformedCitations("according to source1")).toBe("according to [1]");
    });

    it("fixes 'source 1' pattern with space → '[1]'", () => {
        expect(fixMalformedCitations("according to source 1")).toBe("according to [1]");
    });

    it("fixes 'Source1' pattern (capitalized) → '[1]'", () => {
        expect(fixMalformedCitations("according to Source1")).toBe("according to [1]");
    });

    it("fixes 'Source 1' pattern (capitalized with space) → '[1]'", () => {
        expect(fixMalformedCitations("according to Source 1")).toBe("according to [1]");
    });

    it("fixes '(source 1)' pattern in parentheses → '[1]'", () => {
        expect(fixMalformedCitations("see details (source 1) here")).toBe("see details [1] here");
    });

    it("fixes multiple source patterns in same text", () => {
        expect(fixMalformedCitations("from source1 and Source 2")).toBe("from [1] and [2]");
    });

    it("fixes bare sentence-boundary citations", () => {
        expect(fixMalformedCitations("The duty applies. 1 The court may order otherwise.")).toBe("The duty applies.[1] The court may order otherwise.");
    });

    it("fixes chained bare citations after punctuation", () => {
        expect(fixMalformedCitations("The issues are listed. 4 1 Pre-action conduct applies.")).toBe(
            "The issues are listed.[4][1] Pre-action conduct applies."
        );
    });
});

describe("collapseAdjacentCitations", () => {
    it("preserves distinct adjacent citations", () => {
        expect(collapseAdjacentCitations("Text [1][2]")).toBe("Text [1][2]");
    });

    it("handles multiple distinct adjacent citations", () => {
        expect(collapseAdjacentCitations("Text [1][2][3]")).toBe("Text [1][2][3]");
    });

    it("handles repeated same citations", () => {
        expect(collapseAdjacentCitations("Text [1][1][1]")).toBe("Text [1]");
    });

    it("deduplicates only repeated runs inside mixed adjacent citations", () => {
        expect(collapseAdjacentCitations("Text [1][1][3][3][4]")).toBe("Text [1][3][4]");
    });

    it("preserves non-adjacent citations", () => {
        expect(collapseAdjacentCitations("Text [1] and more [2]")).toBe("Text [1] and more [2]");
    });
});

describe("sanitizeCitations", () => {
    it("applies both fixes in correct order", () => {
        // First fixes "1. 1" to "[1]", then if there were adjacent would collapse them
        expect(sanitizeCitations("Text 1. 1")).toBe("Text [1]");
    });

    it("handles complex mixed patterns", () => {
        const input = "First point 1. 1 and see [2][3] for more";
        const expected = "First point [1] and see [2][3] for more";
        expect(sanitizeCitations(input)).toBe(expected);
    });

    it("preserves multiple live citations returned by the backend", () => {
        const input = "The respondent must be named and served [1][3].";
        const expected = "The respondent must be named and served [1][3].";
        expect(sanitizeCitations(input)).toBe(expected);
    });

    it("removes trailing 'Citation:' block with bare numbers and fixes unbracketed citation", () => {
        const input = "The overriding objective is enabling the court to deal with cases justly and at proportionate cost 1.\n\nCitation:\n1. 1";
        const expected = "The overriding objective is enabling the court to deal with cases justly and at proportionate cost.[1]";
        expect(sanitizeCitations(input)).toBe(expected);
    });

    it("removes trailing 'Citations:' block with Source items", () => {
        const input = "Some answer text 1.\n\nCitations:\n1. Source 1\n2. Source 2";
        const expected = "Some answer text.[1]";
        expect(sanitizeCitations(input)).toBe(expected);
    });

    it("removes trailing 'References:' block", () => {
        const input = "Answer text here [1].\n\nReferences:\n1. Document.pdf";
        const expected = "Answer text here [1].";
        expect(sanitizeCitations(input)).toBe(expected);
    });

    it("removes decimal bracket references before parsing source citations", () => {
        const input = "The court may act under [7.3] and should refer to source 1.";
        const expected = "The court may act under and should refer to [1].";
        expect(sanitizeCitations(input)).toBe(expected);
    });
});

describe("findMatchingCitation", () => {
    const possibleCitations = [
        "Part 1, Practice_Direction_31B___Disclosure_Of_Electronic_Documents_chunk_000, Practice_Direction_31B.pdf",
        "1.1, CPR_Part_31___Disclosure_And_Inspection_chunk_002, CPR_Part_31.pdf",
        "Rule 31.6, CPR_Part_31.pdf",
        "Benefit_Options-2.pdf"
    ];

    it("matches exact endsWith (upstream behavior)", () => {
        expect(findMatchingCitation("Benefit_Options-2.pdf", possibleCitations)).toBe("Benefit_Options-2.pdf");
    });

    it("matches exact endsWith for partial citation", () => {
        // Both CPR_Part_31 citations end with "CPR_Part_31.pdf", find returns the first match
        const result = findMatchingCitation("CPR_Part_31.pdf", possibleCitations);
        expect(result).toBeDefined();
        expect(result!.endsWith("CPR_Part_31.pdf")).toBe(true);
    });

    it("matches LLM-humanized citation with spaces instead of underscores", () => {
        const result = findMatchingCitation("Part 1, Practice Direction 31B", possibleCitations);
        expect(result).toBe("Part 1, Practice_Direction_31B___Disclosure_Of_Electronic_Documents_chunk_000, Practice_Direction_31B.pdf");
    });

    it("matches LLM-simplified citation without chunk IDs or extensions", () => {
        const result = findMatchingCitation("1.1, CPR Part 31", possibleCitations);
        expect(result).toBe("1.1, CPR_Part_31___Disclosure_And_Inspection_chunk_002, CPR_Part_31.pdf");
    });

    it("matches subsection-only reference", () => {
        const result = findMatchingCitation("Rule 31.6, CPR Part 31", possibleCitations);
        expect(result).toBe("Rule 31.6, CPR_Part_31.pdf");
    });

    it("returns undefined for no match", () => {
        expect(findMatchingCitation("Completely Unknown Document", possibleCitations)).toBeUndefined();
    });

    it("returns undefined for very short parts", () => {
        expect(findMatchingCitation("ab", possibleCitations)).toBeUndefined();
    });

    it("prefers the most specific match", () => {
        const citations = ["Part 1, DocumentA_chunk_000, DocumentA.pdf", "Part 1, DocumentB_chunk_000, DocumentB.pdf"];
        // "Part 1, DocumentA" should match DocumentA (more specific overlap)
        const result = findMatchingCitation("Part 1, DocumentA", citations);
        expect(result).toBe("Part 1, DocumentA_chunk_000, DocumentA.pdf");
    });
});

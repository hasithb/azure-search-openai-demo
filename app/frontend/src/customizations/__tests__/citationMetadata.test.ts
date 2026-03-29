/**
 * CUSTOM CITATION METADATA TESTS
 * ===============================
 * Tests for structured citation metadata extraction and path building.
 * These tests are independent of upstream and will continue to work after merges.
 */

import { describe, expect, it } from "vitest";
import { extractMetadataFromDataPoint, buildCitationLabel, buildCitationPath, StructuredCitationMetadata } from "../citationMetadata";
import type { SourceTextItem } from "../../api/models";

describe("extractMetadataFromDataPoint", () => {
    it("returns empty metadata for null input", () => {
        const result = extractMetadataFromDataPoint(null);
        expect(result).toEqual({
            subsectionId: "",
            sourcepage: "",
            sourcefile: "",
            category: "",
            content: "",
            storageUrl: ""
        });
    });

    it("returns empty metadata for undefined input", () => {
        const result = extractMetadataFromDataPoint(undefined);
        expect(result).toEqual({
            subsectionId: "",
            sourcepage: "",
            sourcefile: "",
            category: "",
            content: "",
            storageUrl: ""
        });
    });

    it("extracts all fields from a fully populated data point", () => {
        const dp: SourceTextItem = {
            subsection_id: "CPR-31.2",
            sourcepage: "E. Disclosure, E.1 Generally (p. 60)",
            sourcefile: "civil-procedure-rules.pdf",
            category: "Civil Procedure Rules",
            content: "The court may make an order for disclosure",
            storageurl: "https://www.justice.gov.uk/courts/procedure-rules"
        };
        const result = extractMetadataFromDataPoint(dp);
        expect(result).toEqual({
            subsectionId: "CPR-31.2",
            sourcepage: "E. Disclosure, E.1 Generally (p. 60)",
            sourcefile: "civil-procedure-rules.pdf",
            category: "Civil Procedure Rules",
            content: "The court may make an order for disclosure",
            storageUrl: "https://www.justice.gov.uk/courts/procedure-rules"
        });
    });

    it("trims whitespace from all fields", () => {
        const dp: SourceTextItem = {
            subsection_id: "  CPR-31.2  ",
            sourcepage: "  page 5  ",
            sourcefile: "  file.pdf  ",
            category: "  Legal  ",
            content: "  some content  ",
            storageurl: "  https://example.com  "
        };
        const result = extractMetadataFromDataPoint(dp);
        expect(result.subsectionId).toBe("CPR-31.2");
        expect(result.sourcepage).toBe("page 5");
        expect(result.sourcefile).toBe("file.pdf");
        expect(result.category).toBe("Legal");
        expect(result.content).toBe("some content");
        expect(result.storageUrl).toBe("https://example.com");
    });

    it("handles partial data point with only sourcefile", () => {
        const dp: SourceTextItem = {
            sourcefile: "benefits.pdf"
        };
        const result = extractMetadataFromDataPoint(dp);
        expect(result.sourcefile).toBe("benefits.pdf");
        expect(result.subsectionId).toBe("");
        expect(result.storageUrl).toBe("");
    });
});

describe("buildCitationPath", () => {
    it("returns empty string for null input", () => {
        expect(buildCitationPath(null)).toBe("");
    });

    it("returns empty string for undefined input", () => {
        expect(buildCitationPath(undefined)).toBe("");
    });

    it("prefers /content/<sourcefile> when a legal web doc also has an internal sourcefile", () => {
        const dp: SourceTextItem = {
            storageurl: "https://www.justice.gov.uk/courts/procedure-rules/civil",
            sourcefile: "civil-procedure-rules.pdf",
            sourcepage: "Part 1 – Overriding Objective"
        };
        expect(buildCitationPath(dp)).toBe(`/content/${encodeURIComponent("civil-procedure-rules.pdf")}`);
    });

    it("returns /content/<sourcepage> when sourcepage has #page= (PDF ingested)", () => {
        const dp: SourceTextItem = {
            sourcefile: "benefits.pdf",
            sourcepage: "benefits.pdf#page=5"
        };
        expect(buildCitationPath(dp)).toBe(`/content/${encodeURIComponent("benefits.pdf#page=5")}`);
    });

    it("returns /content/<sourcefile> when sourcepage lacks #page=", () => {
        const dp: SourceTextItem = {
            sourcefile: "PerksPlus.pdf",
            sourcepage: "Part 2 – Benefits Overview"
        };
        expect(buildCitationPath(dp)).toBe(`/content/${encodeURIComponent("PerksPlus.pdf")}`);
    });

    it("returns /content/<sourcepage> as last fallback when sourcefile is empty", () => {
        const dp: SourceTextItem = {
            sourcepage: "some-page-reference"
        };
        expect(buildCitationPath(dp)).toBe(`/content/${encodeURIComponent("some-page-reference")}`);
    });

    it("returns empty string when all fields are empty", () => {
        const dp: SourceTextItem = {};
        expect(buildCitationPath(dp)).toBe("");
    });

    it("prefers /content/<sourcepage> over storageUrl when a PDF page reference is available", () => {
        const dp: SourceTextItem = {
            storageurl: "https://example.com/doc",
            sourcepage: "file.pdf#page=3"
        };
        expect(buildCitationPath(dp)).toBe(`/content/${encodeURIComponent("file.pdf#page=3")}`);
    });

    it("ignores whitespace-only storageUrl", () => {
        const dp: SourceTextItem = {
            storageurl: "   ",
            sourcefile: "doc.pdf"
        };
        expect(buildCitationPath(dp)).toBe(`/content/${encodeURIComponent("doc.pdf")}`);
    });

    it("returns storageUrl when it is the only citation path available", () => {
        const dp: SourceTextItem = {
            storageurl: "https://www.justice.gov.uk/courts/procedure-rules/civil"
        };
        expect(buildCitationPath(dp)).toBe("https://www.justice.gov.uk/courts/procedure-rules/civil");
    });
});

describe("buildCitationLabel", () => {
    it("includes subsection, sourcepage, and sourcefile when available", () => {
        const metadata: StructuredCitationMetadata = {
            subsectionId: "46.2",
            sourcepage: "Part 46 – Costs special cases",
            sourcefile: "Part 46",
            category: "Civil Procedure Rules and Practice Directions",
            content: "",
            storageUrl: "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-46-costs-special-cases"
        };

        expect(buildCitationLabel(metadata)).toBe("46.2, Part 46 – Costs special cases, Part 46");
    });

    it("avoids repeating equivalent sourcepage and sourcefile fragments", () => {
        const metadata: StructuredCitationMetadata = {
            subsectionId: "",
            sourcepage: "Commercial Court Guide",
            sourcefile: "Commercial Court Guide",
            category: "Commercial Court Guide",
            content: "",
            storageUrl: ""
        };

        expect(buildCitationLabel(metadata, "fallback")).toBe("Commercial Court Guide");
    });

    it("returns the fallback when metadata is empty", () => {
        expect(buildCitationLabel(undefined, "fallback label")).toBe("fallback label");
    });
});

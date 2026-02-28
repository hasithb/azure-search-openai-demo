import { describe, expect, it } from "vitest";
import { parseAnswerToHtml, fixMalformedCitations, collapseAdjacentCitations, sanitizeCitations } from "../AnswerParser";
import type { ChatAppResponse } from "../../../api";

type TestResponse = ChatAppResponse & {
    context: ChatAppResponse["context"] & {
        citation_map?: Record<string, string>;
        enhanced_citations?: string[];
    } & Record<string, any>;
};

const createResponse = (overrides: Partial<TestResponse>): TestResponse => {
    const base: TestResponse = {
        message: { content: "", role: "assistant" },
        delta: { content: "", role: "assistant" },
        context: {
            data_points: [],
            followup_questions: null,
            thoughts: [],
            citation_map: {},
            enhanced_citations: []
        },
        session_state: {}
    } as TestResponse;

    return {
        ...base,
        ...overrides,
        context: {
            ...base.context,
            ...(overrides.context || {})
        }
    };
};

describe("parseAnswerToHtml", () => {
    it("creates citation details for valid citations matching data_points.citations", () => {
        const response = createResponse({
            message: { content: "See this guidance [doc1.pdf]", role: "assistant" },
            context: {
                data_points: {
                    text: ["Some source text"],
                    citations: ["doc1.pdf"]
                }
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        expect(result.citations).toHaveLength(1);
        expect(result.citations[0].reference).toBe("doc1.pdf");
        expect(result.citations[0].index).toBe(1);
        expect(result.answerHtml).toContain("<sup");
    });

    it("returns empty citations when no matching citation exists in data_points", () => {
        const response = createResponse({
            message: { content: "Requirements are listed in [unknown.pdf]", role: "assistant" },
            context: {
                data_points: {
                    text: ["Some text"],
                    citations: ["other.pdf"]
                }
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        expect(result.citations).toHaveLength(0);
        // The unmatched citation is rendered as plain text
        expect(result.answerHtml).toContain("[unknown.pdf]");
    });
});

describe("fixMalformedCitations", () => {
    it("fixes duplicated number pattern like '1. 1' → '[1]'", () => {
        expect(fixMalformedCitations("The answer is here 1. 1")).toBe("The answer is here [1]");
    });

    it("fixes duplicated pattern at end of sentence", () => {
        expect(fixMalformedCitations("See the disclosure 1. 1 for details")).toBe("See the disclosure [1] for details");
    });

    it("fixes multiple duplicated patterns in same text", () => {
        expect(fixMalformedCitations("First point 1. 1 and second point 2. 2")).toBe("First point [1] and second point [2]");
    });

    it("handles larger citation numbers", () => {
        expect(fixMalformedCitations("Reference 45. 45 in the guide")).toBe("Reference [45] in the guide");
    });

    it("does not modify valid decimal numbers", () => {
        expect(fixMalformedCitations("The value is 3.14 and 2.5")).toBe("The value is 3.14 and 2.5");
    });

    it("does not modify non-duplicated patterns", () => {
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

    it("does not modify section numbers like 3.1", () => {
        expect(fixMalformedCitations("See section 3.1 for details")).toBe("See section 3.1 for details");
    });

    it("fixes range citations with en-dash", () => {
        expect(fixMalformedCitations("See rules [69–81] for details")).toBe("See rules [69] for details");
    });

    it("fixes range citations with hyphen", () => {
        expect(fixMalformedCitations("See rules [69-81] for details")).toBe("See rules [69] for details");
    });
});

describe("collapseAdjacentCitations", () => {
    it("collapses adjacent citations to keep last one", () => {
        expect(collapseAdjacentCitations("Text [1][2]")).toBe("Text [2]");
    });

    it("handles multiple adjacent citations", () => {
        expect(collapseAdjacentCitations("Text [1][2][3]")).toBe("Text [3]");
    });

    it("handles repeated same citations", () => {
        expect(collapseAdjacentCitations("Text [1][1][1]")).toBe("Text [1]");
    });
});

describe("sanitizeCitations", () => {
    it("applies both fixes in correct order", () => {
        // First fixes "1. 1" to "[1]", then if there were adjacent would collapse them
        expect(sanitizeCitations("Text 1. 1")).toBe("Text [1]");
    });
});

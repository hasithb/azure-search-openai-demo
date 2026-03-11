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

    it("resolves numbered citations from legacy citation_map when data_points text is unavailable", () => {
        const response = createResponse({
            message: { content: "See [1] for details.", role: "assistant" },
            context: {
                data_points: {
                    text: [],
                    citations: []
                },
                citation_map: {
                    "1": "35.1, Part 35, CPR_Part35.pdf"
                },
                enhanced_citations: ["35.1, Part 35, CPR_Part35.pdf"]
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        expect(result.citations).toHaveLength(1);
        expect(result.citations[0].reference).toBe("35.1, Part 35, CPR_Part35.pdf");
        expect(result.answerHtml).toContain("<sup");
    });

    it("preserves legacy enhanced citations when answer text has no inline citation markers", () => {
        const response = createResponse({
            message: { content: "Standard disclosure applies to documents relied upon.", role: "assistant" },
            context: {
                data_points: {
                    text: [],
                    citations: []
                },
                enhanced_citations: ["31.6, CPR Part 31, CPR_Part_31.pdf"]
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        expect(result.answerHtml).toContain("Standard disclosure applies");
        expect(result.citations).toHaveLength(1);
        expect(result.citations[0].reference).toBe("31.6, CPR Part 31, CPR_Part_31.pdf");
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
});

describe("sanitizeCitations", () => {
    it("applies both fixes in correct order", () => {
        // First fixes "1. 1" to "[1]", then if there were adjacent would collapse them
        expect(sanitizeCitations("Text 1. 1")).toBe("Text [1]");
    });
});

describe("parseAnswerToHtml - multi-citation pipeline", () => {
    it("resolves multiple numbered citations [1], [2], [3] to correct data_points", () => {
        const response = createResponse({
            message: {
                content: "Filing deadlines are set out in [1]. Strike out rules are in [2]. Bundle prep is in [3].",
                role: "assistant"
            },
            context: {
                data_points: {
                    text: [
                        { citation: "1.1, Part 1, CPR_Part1.pdf", content: "Filing deadlines." },
                        { citation: "3.1, Part 3, CPR_Part3.pdf", content: "Strike out." },
                        { citation: "A4.1, D5, Commercial_Court_Guide.pdf", content: "Bundle prep." }
                    ],
                    citations: ["1.1, Part 1, CPR_Part1.pdf", "3.1, Part 3, CPR_Part3.pdf", "A4.1, D5, Commercial_Court_Guide.pdf"]
                }
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        // Should produce 3 distinct citations
        expect(result.citations).toHaveLength(3);

        // Each citation should reference the enhanced citation string from data_points.text[n-1]
        expect(result.citations[0].reference).toBe("1.1, Part 1, CPR_Part1.pdf");
        expect(result.citations[0].index).toBe(1);

        expect(result.citations[1].reference).toBe("3.1, Part 3, CPR_Part3.pdf");
        expect(result.citations[1].index).toBe(2);

        expect(result.citations[2].reference).toBe("A4.1, D5, Commercial_Court_Guide.pdf");
        expect(result.citations[2].index).toBe(3);

        // HTML should contain superscript elements for all three
        expect(result.answerHtml).toContain("<sup");
        // Should contain the answer text
        expect(result.answerHtml).toContain("Filing deadlines are set out in");
        expect(result.answerHtml).toContain("Strike out rules are in");
        expect(result.answerHtml).toContain("Bundle prep is in");
    });

    it("resolves five numbered citations correctly", () => {
        const response = createResponse({
            message: {
                content: "See [1] and [2] and [3] and [4] and [5] for full guidance.",
                role: "assistant"
            },
            context: {
                data_points: {
                    text: [
                        { citation: "1.1, Part 1, CPR_Part1.pdf", content: "Content 1" },
                        { citation: "3.1, Part 3, CPR_Part3.pdf", content: "Content 2" },
                        { citation: "A4.1, D5, Commercial_Court_Guide.pdf", content: "Content 3" },
                        { citation: "35.1, Part 35, CPR_Part35.pdf", content: "Content 4" },
                        { citation: "3E.1, PD3E, PD3E.pdf", content: "Content 5" }
                    ],
                    citations: [
                        "1.1, Part 1, CPR_Part1.pdf",
                        "3.1, Part 3, CPR_Part3.pdf",
                        "A4.1, D5, Commercial_Court_Guide.pdf",
                        "35.1, Part 35, CPR_Part35.pdf",
                        "3E.1, PD3E, PD3E.pdf"
                    ]
                }
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        expect(result.citations).toHaveLength(5);
        expect(result.citations[0].reference).toBe("1.1, Part 1, CPR_Part1.pdf");
        expect(result.citations[4].reference).toBe("3E.1, PD3E, PD3E.pdf");
    });

    it("preserves adjacent distinct citations in rendered answer HTML", () => {
        const response = createResponse({
            message: {
                content: "The respondent must be named and served [1][3].",
                role: "assistant"
            },
            context: {
                data_points: {
                    text: [
                        {
                            citation: "1.3, Practice Direction 31C, Practice Direction 31C.pdf",
                            content: "1.3 The person who has control of the relevant evidence must be named as a respondent."
                        },
                        {
                            citation: "1.4, Practice Direction 31C, Practice Direction 31C.pdf",
                            content: "1.4 The application must include a narrow description of the evidence sought."
                        },
                        { citation: "CPR 31.3, Appendix 15, Commercial Court Guide.pdf", content: "CPR 31.3 Inspection rights and control of documents." }
                    ],
                    citations: [
                        "1.3, Practice Direction 31C, Practice Direction 31C.pdf",
                        "1.4, Practice Direction 31C, Practice Direction 31C.pdf",
                        "CPR 31.3, Appendix 15, Commercial Court Guide.pdf"
                    ]
                }
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        expect(result.citations).toHaveLength(2);
        expect(result.citations[0].reference).toBe("1.3, Practice Direction 31C, Practice Direction 31C.pdf");
        expect(result.citations[1].reference).toBe("CPR 31.3, Appendix 15, Commercial Court Guide.pdf");
        expect((result.answerHtml.match(/<sup/g) || []).length).toBe(2);
    });

    it("rebuilds section-based citations when data point citation strings are missing", () => {
        const response = createResponse({
            message: {
                content: "Disclosure starts at [1]. Proportionality is addressed at [2]. Competition authority safeguards are at [3].",
                role: "assistant"
            },
            context: {
                data_points: {
                    text: [
                        {
                            subsection_id: "1.1",
                            sourcepage: "Practice Direction 31C",
                            sourcefile: "Practice Direction 31C.pdf",
                            content: "1.1 A person seeking disclosure or inspection must apply in accordance with Part 23."
                        },
                        {
                            subsection_id: "2.1",
                            sourcepage: "Practice Direction 31C",
                            sourcefile: "Practice Direction 31C.pdf",
                            content: "2.1 The court may only permit disclosure or inspection that is proportionate."
                        },
                        {
                            subsection_id: "3.1",
                            sourcepage: "Practice Direction 31C",
                            sourcefile: "Practice Direction 31C.pdf",
                            content: "3.1 Competition authority materials require additional safeguards."
                        }
                    ],
                    citations: []
                },
                citation_map: {},
                enhanced_citations: []
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        expect(result.citations).toHaveLength(3);
        expect(result.citations[0].reference).toBe("1.1, Practice Direction 31C, Practice Direction 31C.pdf");
        expect(result.citations[1].reference).toBe("2.1, Practice Direction 31C, Practice Direction 31C.pdf");
        expect(result.citations[2].reference).toBe("3.1, Practice Direction 31C, Practice Direction 31C.pdf");
        expect(result.answerHtml).toContain('data-citation-text="1.1, Practice Direction 31C, Practice Direction 31C.pdf"');
        expect(result.answerHtml).toContain('data-citation-text="2.1, Practice Direction 31C, Practice Direction 31C.pdf"');
        expect(result.answerHtml).toContain('data-citation-text="3.1, Practice Direction 31C, Practice Direction 31C.pdf"');
    });

    it("derives nested subsection citations from a single composite data point", () => {
        const response = createResponse({
            message: {
                content:
                    "A defendant must file an acknowledgment of service within 14 days after service of the claim form. 1\n\n" +
                    "If the claim form is acknowledged as indicating an intention to defend, the claimant must serve particulars of claim within 28 days of the filing of that acknowledgment of service. 1\n\n" +
                    "If the defence is served, the claimant must file any reply within 21 days after service of the defence. 1\n\n" +
                    "An application under Part 11 for disputes about jurisdiction must be made within 28 days after filing an acknowledgment of service. 1\n\n" +
                    "Citation:\n1. PART 59, Part 59 – Circuit Commercial Court, Part 59",
                role: "assistant"
            },
            context: {
                data_points: {
                    text: [
                        {
                            subsection_id: "59.4",
                            sourcepage: "Part 59 – Circuit Commercial Court",
                            sourcefile: "Part 59",
                            content:
                                "59.4 (1) A defendant must file an acknowledgment of service within 14 days after service of the claim form. " +
                                "(2) If the claim form is acknowledged as indicating an intention to defend, the claimant must serve particulars of claim within 28 days of the filing of that acknowledgment of service. " +
                                "(3) If the defence is served, the claimant must file any reply within 21 days after service of the defence. " +
                                "(4) An application under Part 11 for disputes about jurisdiction must be made within 28 days after filing an acknowledgment of service."
                        }
                    ],
                    citations: []
                },
                citation_map: {},
                enhanced_citations: []
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        expect(result.citations).toHaveLength(4);
        expect(result.citations[0].reference).toBe("59.4(1), Part 59 – Circuit Commercial Court, Part 59");
        expect(result.citations[1].reference).toBe("59.4(2), Part 59 – Circuit Commercial Court, Part 59");
        expect(result.citations[2].reference).toBe("59.4(3), Part 59 – Circuit Commercial Court, Part 59");
        expect(result.citations[3].reference).toBe("59.4(4), Part 59 – Circuit Commercial Court, Part 59");
        expect((result.answerHtml.match(/<sup/g) || []).length).toBe(4);
    });

    it("handles repeated numbered citations without duplicating citation entries", () => {
        const response = createResponse({
            message: {
                content: "See [1] for details. As mentioned in [1], the rule applies.",
                role: "assistant"
            },
            context: {
                data_points: {
                    text: [{ citation: "1.1, Part 1, CPR_Part1.pdf", content: "Filing rules." }],
                    citations: ["1.1, Part 1, CPR_Part1.pdf"]
                }
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        // Same citation referenced twice, but citationList should deduplicate
        expect(result.citations).toHaveLength(1);
        expect(result.citations[0].reference).toBe("1.1, Part 1, CPR_Part1.pdf");
    });

    it("handles out-of-range citation number gracefully", () => {
        const response = createResponse({
            message: {
                content: "See [1] for info. Also see [99] for more.",
                role: "assistant"
            },
            context: {
                data_points: {
                    text: [{ citation: "1.1, Part 1, CPR_Part1.pdf", content: "Content" }],
                    citations: ["1.1, Part 1, CPR_Part1.pdf"]
                }
            } as any
        });

        const result = parseAnswerToHtml(response as any, false, () => {});

        // Only [1] should resolve; [99] is out of range and rendered as plain text
        expect(result.citations).toHaveLength(1);
        expect(result.answerHtml).toContain("[99]");
    });
});

/**
 * Comprehensive Citation Testing – Frontend Parser & Navigation Tests
 *
 * Tests the functions exported from SupportingContentParser.ts:
 *   - parseSupportingContentItem(): field extraction from objects and strings
 *   - extractSubsectionContent(): finding and bounding subsection text
 *   - parseSubsectionFromCitation(): validating subsection format tokens
 *
 * Tests simulated navigation matching (findMatchingContentIndex logic):
 *   - 3-part, 2-part, and 1-part citation matching
 *   - Subsection-in-content strict check
 *   - Metadata fallback path
 *   - 4-part and 5-part citations (commas in sourcepage)
 *
 * Run:
 *   cd app/frontend && npx vitest run src/components/SupportingContent/__tests__/SupportingContentParser.test.ts
 */

import { describe, it, expect } from "vitest";
import { parseSupportingContentItem, extractSubsectionContent, parseSubsectionFromCitation } from "../SupportingContentParser";

// ════════════════════════════════════════════════════════════════════════════════
// A. parseSupportingContentItem
// ════════════════════════════════════════════════════════════════════════════════

describe("parseSupportingContentItem", () => {
    describe("object format", () => {
        it("extracts all fields from a complete CPR object", () => {
            const item = {
                id: "doc-1",
                sourcepage: "Part 1 – Overriding Objective",
                sourcefile: "Part 1",
                category: "Civil Procedure Rules and Practice Directions",
                content: "1.1 These Rules are a procedural code.",
                storageUrl: "https://justice.gov.uk/rules/part01",
                updated: "2024-01-01"
            };
            const result = parseSupportingContentItem(item);
            expect(result.title).toBe("Part 1");
            expect(result.sourcepage).toBe("Part 1 – Overriding Objective");
            expect(result.sourcefile).toBe("Part 1");
            expect(result.category).toBe("Civil Procedure Rules and Practice Directions");
            expect(result.content).toContain("1.1");
            expect(result.url).toBe("https://justice.gov.uk/rules/part01");
        });

        it("extracts fields from a court guide object with storageurl variant", () => {
            const item = {
                sourcefile: "Commercial Court Guide",
                sourcepage: "E.  Disclosure, E.1 Generally (p. 60)",
                category: "Commercial Court",
                content: "E.1.1 Standard disclosure is not the norm.",
                storageurl: "https://example.com/ccg"
            };
            const result = parseSupportingContentItem(item);
            expect(result.sourcefile).toBe("Commercial Court Guide");
            expect(result.sourcepage).toBe("E.  Disclosure, E.1 Generally (p. 60)");
            expect(result.url).toBe("https://example.com/ccg");
        });

        it("handles storage_url snake_case variant", () => {
            const item = {
                sourcefile: "Part 35",
                sourcepage: "Practice Direction 35",
                content: "Expert evidence.",
                storage_url: "https://example.com/pd35"
            };
            const result = parseSupportingContentItem(item);
            expect(result.url).toBe("https://example.com/pd35");
        });

        it("falls back to url field when no storage variant exists", () => {
            const item = {
                sourcefile: "Part 3",
                content: "Scope.",
                url: "https://example.com/part3"
            };
            const result = parseSupportingContentItem(item);
            expect(result.url).toBe("https://example.com/part3");
        });

        it("prefers full_content over content", () => {
            const item = {
                sourcefile: "Part 1",
                content: "Short content.",
                full_content: "This is the full content with much more detail."
            };
            const result = parseSupportingContentItem(item);
            expect(result.content).toContain("full content");
        });

        it("handles missing sourcefile – falls back to sourcepage for title", () => {
            const item = {
                sourcepage: "Practice Direction 1A",
                content: "Vulnerable parties."
            };
            const result = parseSupportingContentItem(item);
            expect(result.title).toBe("Practice Direction 1A");
        });

        it("handles empty object – defaults gracefully", () => {
            const result = parseSupportingContentItem({});
            expect(result.title).toBe("Document Source");
            expect(result.content).toBe("");
        });

        it("extracts inline SOURCE metadata from content when top-level fields are missing", () => {
            const item = {
                content:
                    "Practice Direction 31B – Disclosure of Electronic Documents: SOURCE: Practice Direction 31B SOURCEPAGE: Practice Direction 31B – Disclosure of Electronic Documents CATEGORY: Civil Procedure Rules and Practice Directions SECTION: Practice Direction 31B – Disclosure of Electronic Documents\\n\\nPart 2 of 2\\n## Proposals for the method to be adopted for their searches"
            };

            const result = parseSupportingContentItem(item);

            expect(result.title).toBe("Practice Direction 31B");
            expect(result.sourcefile).toBe("Practice Direction 31B");
            expect(result.sourcepage).toContain("Practice Direction 31B");
            expect(result.category).toBe("Civil Procedure Rules and Practice Directions");
            expect(result.content).toContain("Part 2 of 2");
            expect(result.content).not.toContain("SOURCE:");
            expect(result.content).not.toContain("SOURCEPAGE:");
        });

        it("handles null input", () => {
            const result = parseSupportingContentItem(null);
            expect(result.title).toBe("Unknown source");
        });

        it("handles undefined input", () => {
            const result = parseSupportingContentItem(undefined);
            expect(result.title).toBe("Unknown source");
        });
    });

    describe("string format (legacy)", () => {
        it("parses citation pattern from string", () => {
            const item = "[1.1, Part 1, Part 1 – Overriding Objective]: Content text here.";
            const result = parseSupportingContentItem(item);
            expect(result.sourcepage).toBe("1.1");
            expect(result.sourcefile).toBe("Part 1 – Overriding Objective");
            expect(result.content).toContain("Content text here");
        });

        it("handles plain string without citation pattern", () => {
            const item = "Just some plain text content.";
            const result = parseSupportingContentItem(item);
            expect(result.content).toContain("plain text content");
        });
    });
});

// ════════════════════════════════════════════════════════════════════════════════
// B. extractSubsectionContent
// ════════════════════════════════════════════════════════════════════════════════

describe("extractSubsectionContent", () => {
    describe("numeric subsections", () => {
        it("extracts 1.1 and stops at 1.2", () => {
            const content = "1.1 These Rules are a procedural code.\n\n1.2 The overriding objective is.";
            const result = extractSubsectionContent(content, "1.1");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("1.1");
            expect(result!.content).toContain("procedural code");
            expect(result!.content).not.toContain("1.2");
        });

        it("extracts deep numeric 3.1.2 and stops at 3.1.3", () => {
            const content = "3.1.1 The court may.\n\n3.1.2 An order may also.\n\n3.1.3 Further provisions.";
            const result = extractSubsectionContent(content, "3.1.2");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("3.1.2");
            expect(result!.content).toContain("order may also");
            expect(result!.content).not.toContain("3.1.3");
        });

        it("extracts last subsection to end of content", () => {
            const content = "5.1 First.\n\n5.2 Second.\n\n5.3 This is the last subsection in content.";
            const result = extractSubsectionContent(content, "5.3");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("5.3");
            expect(result!.content).toContain("last subsection");
        });
    });

    describe("alpha-dot subsections", () => {
        it("extracts E.1.1 and stops at E.1.2", () => {
            const content = "E.1.1 Standard disclosure is not the norm.\n\nE.1.2 The parties should.";
            const result = extractSubsectionContent(content, "E.1.1");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("E.1.1");
            expect(result!.content).not.toContain("E.1.2");
        });

        it("extracts D.7.1 court guide subsection", () => {
            const content = "D.7.1 Split trials may be considered.\n\nD.7.2 Applications for split.";
            const result = extractSubsectionContent(content, "D.7.1");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("Split trials");
        });
    });

    describe("rule and para subsections", () => {
        it("extracts 'Rule 31.6' subsection", () => {
            const content = "Rule 31.6 Standard disclosure requires.\n\nRule 31.7 The court may.";
            const result = extractSubsectionContent(content, "Rule 31.6");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("Rule 31.6");
            expect(result!.content).not.toContain("Rule 31.7");
        });

        it("extracts 'Para 5' subsection", () => {
            const content = "Para 5 The parties should consider.\n\nPara 6 Where the dispute.";
            const result = extractSubsectionContent(content, "Para 5");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("Para 5");
            expect(result!.content).not.toContain("Para 6");
        });
    });

    describe("markdown formatted content", () => {
        it("finds subsection in markdown heading", () => {
            const content = "## 1.1 Introduction\n\nThis is the intro.\n\n## 1.2 Scope\n\nThe scope.";
            const result = extractSubsectionContent(content, "1.1");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("Introduction");
        });

        it("finds subsection after bold formatting", () => {
            const content = "**1.1** These rules are a procedural code.\n\n**1.2** The overriding objective.";
            const result = extractSubsectionContent(content, "1.1");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("procedural code");
        });
    });

    describe("edge cases", () => {
        it("returns null for empty content", () => {
            expect(extractSubsectionContent("", "1.1")).toBeNull();
        });

        it("returns null for empty target", () => {
            expect(extractSubsectionContent("Some content", "")).toBeNull();
        });

        it("returns null when subsection not found", () => {
            const content = "2.1 Only this section exists.";
            expect(extractSubsectionContent(content, "9.9")).toBeNull();
        });

        it("handles content with only the target subsection (no next boundary)", () => {
            const content = "A.1 This is the entire content without any other subsection.";
            const result = extractSubsectionContent(content, "A.1");
            expect(result).not.toBeNull();
            expect(result!.content).toContain("entire content");
            expect(result!.endIndex).toBe(content.length);
        });

        it("extracts 46.2 cleanly from the real Part 46 chunk format", () => {
            const content =
                "Document: Part 46 – Costs special cases\n" +
                "## 46.1\n\n" +
                "[PART 46 – COSTS-SPECIAL CASES > 46.1] (1) This paragraph applies where a person applies –\n\n" +
                "[PART 46 – COSTS-SPECIAL CASES > 46.1] (b) whether the parties to the application have complied with any relevant pre-action protocol. " +
                "## Costs orders in favour of or against non-parties\n\n" +
                "## 46.2\n\n" +
                "[PART 46 – COSTS-SPECIAL CASES > 46.2] (1) Where the court is considering whether to exercise its power under section 51 of the Senior Courts Act 1981...\n\n" +
                "[PART 46 – COSTS-SPECIAL CASES > 46.2] (3) Neither rule 19.4 nor rule 20.7 applies to the joinder of a person under paragraph (1). " +
                "## Limitations on court’s power to award costs in favour of trustee or personal representative\n\n" +
                "## 46.3\n\n" +
                "[PART 46 – COSTS-SPECIAL CASES > 46.3] (1) This rule applies where –";

            const result = extractSubsectionContent(content, "46.2");

            expect(result).not.toBeNull();
            expect(result!.content).toContain("46.2");
            expect(result!.content).toContain("section 51 of the Senior Courts Act 1981");
            expect(result!.content).not.toContain("46.3");
        });

        it("extracts only CPR 59.4 and stops before 59.5", () => {
            const content =
                "## Claim form and particulars of claim\n\n" +
                "## 59.4\n\n" +
                "[PART 59 – CIRCUIT COMMERCIAL COURTS > 59.4] (1) If particulars of claim are not contained in or served with the claim form –\n\n" +
                "[PART 59 – CIRCUIT COMMERCIAL COURTS > 59.4] (a) the claim form must state that, if an acknowledgment of service is filed which indicates an intention to defend the claim, particulars of claim will follow;\n\n" +
                "## Acknowledgment of service\n\n" +
                "## 59.5\n\n" +
                "[PART 59 – CIRCUIT COMMERCIAL COURTS > 59.5] (1) A defendant must file an acknowledgment of service in every case.";

            const result = extractSubsectionContent(content, "59.4");

            expect(result).not.toBeNull();
            expect(result!.content).not.toContain("Claim form and particulars of claim");
            expect(result!.content.startsWith("59.4")).toBe(true);
            expect(result!.content).not.toContain("Acknowledgment of service");
            expect(result!.content).not.toContain("## 59.5");
        });

        it.each([
            {
                heading: "Costs management orders and costs capping orders",
                subsection: "3.3",
                nextHeading: "Summary assessment",
                nextSubsection: "4.1"
            },
            {
                heading: "Application for disclosure",
                subsection: "39.2",
                nextHeading: "Application for security for costs",
                nextSubsection: "39.3"
            }
        ])("extracts only $subsection style content and excludes surrounding titles", ({ heading, subsection, nextHeading, nextSubsection }) => {
            const content =
                `## ${heading}\n\n` +
                `## ${subsection}\n\n` +
                `[GUIDE > ${subsection}] Example body text for ${subsection}.\n\n` +
                `## ${nextHeading}\n\n` +
                `## ${nextSubsection}\n\n` +
                `[GUIDE > ${nextSubsection}] Example body text for ${nextSubsection}.`;

            const result = extractSubsectionContent(content, subsection);

            expect(result).not.toBeNull();
            expect(result!.content).not.toContain(heading);
            expect(result!.content.startsWith(subsection)).toBe(true);
            expect(result!.content).not.toContain(nextHeading);
            expect(result!.content).not.toContain(`## ${nextSubsection}`);
        });
    });
});

// ════════════════════════════════════════════════════════════════════════════════
// C. parseSubsectionFromCitation
// ════════════════════════════════════════════════════════════════════════════════

describe("parseSubsectionFromCitation", () => {
    describe("numeric patterns", () => {
        it("parses '1.1, Part 1, Part 1 – Overriding Objective'", () => {
            expect(parseSubsectionFromCitation("1.1, Part 1, Part 1 – Overriding Objective")).toBe("1.1");
        });

        it("parses '35.4, Practice Direction 35, PD 35 – Experts'", () => {
            expect(parseSubsectionFromCitation("35.4, Practice Direction 35, PD 35 – Experts")).toBe("35.4");
        });

        it("parses deep numeric '3.1.2, Part 3, Part 3 – Scope'", () => {
            expect(parseSubsectionFromCitation("3.1.2, Part 3, Part 3 – Scope")).toBe("3.1.2");
        });
    });

    describe("alpha-dot patterns", () => {
        it("parses 'E.1.1, E. Disclosure, Commercial Court Guide'", () => {
            expect(parseSubsectionFromCitation("E.1.1, E. Disclosure, Commercial Court Guide")).toBe("E.1.1");
        });

        it("parses 'D.7.1, D. Case Management, Commercial Court Guide'", () => {
            expect(parseSubsectionFromCitation("D.7.1, D. Case Management, Commercial Court Guide")).toBe("D.7.1");
        });

        it("parses 'A.1, Introduction, Chancery Guide'", () => {
            expect(parseSubsectionFromCitation("A.1, Introduction, Chancery Guide")).toBe("A.1");
        });
    });

    describe("alpha-num patterns", () => {
        it("parses 'A1, Pre-Action Protocol, Pre'", () => {
            expect(parseSubsectionFromCitation("A1, Pre-Action Protocol, Pre")).toBe("A1");
        });

        it("parses 'D5, Some Section, Guide'", () => {
            expect(parseSubsectionFromCitation("D5, Some Section, Guide")).toBe("D5");
        });
    });

    describe("rule and para patterns", () => {
        it("parses 'Rule 31.6, Part 31, Part 31 – Disclosure'", () => {
            expect(parseSubsectionFromCitation("Rule 31.6, Part 31, Part 31 – Disclosure")).toBe("Rule 31.6");
        });

        it("parses 'Para 5, Appendix 1, Commercial Court Guide'", () => {
            expect(parseSubsectionFromCitation("Para 5, Appendix 1, Commercial Court Guide")).toBe("Para 5");
        });
    });

    describe("chapter, section, part patterns", () => {
        it("parses 'Chapter 9, Applications, Chancery Guide'", () => {
            expect(parseSubsectionFromCitation("Chapter 9, Applications, Chancery Guide")).toBe("Chapter 9");
        });

        it("parses 'Part 35, Expert Evidence, Part 35 – Experts'", () => {
            expect(parseSubsectionFromCitation("Part 35, Expert Evidence, Part 35 – Experts")).toBe("Part 35");
        });

        it("parses 'Appendix 1, Overview, Commercial Court Guide'", () => {
            expect(parseSubsectionFromCitation("Appendix 1, Overview, Commercial Court Guide")).toBe("Appendix 1");
        });
    });

    describe("4-part citations (commas in sourcepage)", () => {
        it("extracts subsection from 4-part citation", () => {
            // 4-part: "E.1.1, E.  Disclosure, E.1 Generally (p. 60), Commercial Court Guide"
            // The first part is the subsection
            const citation = "E.1.1, E.  Disclosure, E.1 Generally (p. 60), Commercial Court Guide";
            const result = parseSubsectionFromCitation(citation);
            expect(result).toBe("E.1.1");
        });

        it("extracts subsection from 4-part Chancery citation", () => {
            const citation = "9.1, Chapter 9 Applications, 9.1 General (p. 45), Chancery Guide";
            const result = parseSubsectionFromCitation(citation);
            expect(result).toBe("9.1");
        });
    });

    describe("dash-delimited display labels", () => {
        it("handles 'D.7.5 - D.  Case Management' format", () => {
            expect(parseSubsectionFromCitation("D.7.5 - D.  Case Management, Commercial Court Guide")).toBe("D.7.5");
        });
    });

    describe("edge cases", () => {
        it("returns null for empty string", () => {
            expect(parseSubsectionFromCitation("")).toBeNull();
        });

        it("returns null for null input", () => {
            expect(parseSubsectionFromCitation(null as any)).toBeNull();
        });

        it("handles 2-part citation like 'Practice Direction 1A, PD 1A'", () => {
            // "Practice Direction 1A" is not a recognized subsection pattern in the strict set
            // but may match multi-word or other patterns
            const result = parseSubsectionFromCitation("Practice Direction 1A, PD 1A");
            // This may or may not match — the key is it shouldn't crash
            expect(result === null || typeof result === "string").toBe(true);
        });

        it("handles leading index number: '1. 1.1, Part 1, Part 1'", () => {
            // The function normalizes leading "N. " prefix
            expect(parseSubsectionFromCitation("1. 1.1, Part 1, Part 1 – Overriding Objective")).toBe("1.1");
        });
    });
});

// ════════════════════════════════════════════════════════════════════════════════
// D. Citation Label Parsing (simulated parseCitationLabelParts)
// ════════════════════════════════════════════════════════════════════════════════

/** Replicates AnswerParser.parseCitationLabelParts() */
function parseCitationLabelParts(citation: string): {
    subsection: string;
    sourcePage: string;
    document: string;
    parts: string[];
} {
    const parts = (citation || "")
        .split(",")
        .map(p => p.trim())
        .filter(Boolean);
    const subsection = parts.length >= 3 ? parts[0] : "";
    const sourcePage = parts.length >= 3 ? parts.slice(1, -1).join(", ") : parts.length === 2 ? parts[0] : "";
    const document = parts.length >= 1 ? parts[parts.length - 1] : "";
    return { subsection, sourcePage, document, parts };
}

describe("parseCitationLabelParts (simulated)", () => {
    it("3-part: subsection, sourcepage, sourcefile", () => {
        const r = parseCitationLabelParts("1.1, Part 1 – Overriding Objective, Part 1");
        expect(r.subsection).toBe("1.1");
        expect(r.sourcePage).toBe("Part 1 – Overriding Objective");
        expect(r.document).toBe("Part 1");
    });

    it("4-part: joins middle parts as sourcePage", () => {
        const r = parseCitationLabelParts("E.1.1, E.  Disclosure, E.1 Generally (p. 60), Commercial Court Guide");
        expect(r.subsection).toBe("E.1.1");
        expect(r.sourcePage).toBe("E.  Disclosure, E.1 Generally (p. 60)");
        expect(r.document).toBe("Commercial Court Guide");
    });

    it("5-part: joins all middle parts as sourcePage", () => {
        const r = parseCitationLabelParts("D.7.1, D.  Case and Costs Management, D.7 Split trials, (p. 50), Commercial Court Guide");
        expect(r.subsection).toBe("D.7.1");
        expect(r.sourcePage).toBe("D.  Case and Costs Management, D.7 Split trials, (p. 50)");
        expect(r.document).toBe("Commercial Court Guide");
    });

    it("2-part: no subsection", () => {
        const r = parseCitationLabelParts("Practice Direction 1A, PD 1A – Participation");
        expect(r.subsection).toBe("");
        expect(r.sourcePage).toBe("Practice Direction 1A");
        expect(r.document).toBe("PD 1A – Participation");
    });

    it("1-part: just the document", () => {
        const r = parseCitationLabelParts("Part 1");
        expect(r.subsection).toBe("");
        expect(r.sourcePage).toBe("");
        expect(r.document).toBe("Part 1");
    });

    it("empty string", () => {
        const r = parseCitationLabelParts("");
        expect(r.parts).toHaveLength(0);
        expect(r.document).toBe("");
    });
});

// ════════════════════════════════════════════════════════════════════════════════
// E. Citation-to-Content Navigation Matching (simulated findMatchingContentIndex)
// ════════════════════════════════════════════════════════════════════════════════

/** Normalize for matching — mirrors SupportingContent.tsx normalizeMatchText */
function normalizeMatchText(text: string | undefined | null): string {
    return (text || "").trim().toLowerCase().replace(/\s+/g, " ");
}

interface MockDataPoint {
    sourcefile: string;
    sourcepage: string;
    content: string;
    subsection_id?: string;
    category?: string;
}

/**
 * Simulates findMatchingContentIndex matching logic.
 * Returns index into dataPoints array, or -1 if no match.
 */
function findMatchingContentIndex(citation: string, dataPoints: MockDataPoint[]): number {
    if (!citation) return -1;

    let bestIdx = -1;
    let bestScore = 0;

    for (let i = 0; i < dataPoints.length; i++) {
        const dp = dataPoints[i];
        let score = 0;

        const normalized = citation.replace(/^\s*\d+\.\s+/, "");
        const parts = normalized
            .split(",")
            .map((p: string) => p.trim())
            .filter(Boolean);

        // Using parseSubsectionFromCitation for the subsection token
        const parsedSub = parseSubsectionFromCitation(normalized) || "";

        if (parts.length >= 3) {
            const subsection = parts[0];
            const sourcePage = parts[1]; // Note: simplified, doesn't join middle parts
            const document = parts.slice(2).join(", ");

            const docMatch =
                normalizeMatchText(dp.sourcefile) === normalizeMatchText(document) ||
                normalizeMatchText(dp.sourcefile).includes(normalizeMatchText(document)) ||
                normalizeMatchText(document).includes(normalizeMatchText(dp.sourcefile));

            if (!docMatch) continue;
            score += 10;

            // Sourcepage match
            const spMatch =
                normalizeMatchText(dp.sourcepage) === normalizeMatchText(sourcePage) ||
                (dp.sourcepage && sourcePage && normalizeMatchText(dp.sourcepage).includes(normalizeMatchText(sourcePage))) ||
                (dp.sourcepage && sourcePage && normalizeMatchText(sourcePage).includes(normalizeMatchText(dp.sourcepage)));

            if (spMatch) {
                score += 50;
            } else if (!dp.sourcepage) {
                continue;
            }

            // Subsection in content (strict)
            const subToken = parsedSub || subsection;
            if (subToken && subToken.length > 1) {
                const escaped = subToken.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                const found = new RegExp(`(^|\\n)\\s*${escaped}\\b`, "i").test(dp.content) || new RegExp(`\\b${escaped}\\b`, "i").test(dp.content);
                if (found) {
                    score += 40;
                } else {
                    // Metadata fallback
                    if (dp.subsection_id && dp.subsection_id.toLowerCase() === subToken.toLowerCase()) {
                        score += 15;
                    } else {
                        continue;
                    }
                }
            }
        } else if (parts.length === 2) {
            const partA = parts[0];
            const partB = parts[1];
            const docMatch =
                normalizeMatchText(dp.sourcefile) === normalizeMatchText(partB) ||
                normalizeMatchText(dp.sourcefile).includes(normalizeMatchText(partB)) ||
                normalizeMatchText(partB).includes(normalizeMatchText(dp.sourcefile));

            if (docMatch) {
                score += 25;
            } else {
                const spMatch =
                    normalizeMatchText(dp.sourcepage) === normalizeMatchText(partA) ||
                    (dp.sourcepage && normalizeMatchText(dp.sourcepage).includes(normalizeMatchText(partA)));
                if (spMatch) {
                    score += 20;
                } else {
                    continue;
                }
            }
        } else {
            if (dp.sourcefile && citation.includes(dp.sourcefile)) {
                score += 10;
            } else {
                continue;
            }
        }

        if (score > bestScore && score >= 15) {
            bestScore = score;
            bestIdx = i;
        }
    }

    return bestIdx;
}

describe("Citation-to-Content navigation matching", () => {
    const mockDataPoints: MockDataPoint[] = [
        {
            sourcefile: "Part 1",
            sourcepage: "Part 1 – Overriding Objective",
            content: "1.1 These Rules are a procedural code with the overriding objective.",
            subsection_id: "1.1",
            category: "Civil Procedure Rules and Practice Directions"
        },
        {
            sourcefile: "Part 35",
            sourcepage: "Practice Direction 35",
            content: "2.1 Expert evidence should be the independent product of the expert.",
            subsection_id: "2.1",
            category: "Civil Procedure Rules and Practice Directions"
        },
        {
            sourcefile: "Commercial Court Guide",
            sourcepage: "E.  Disclosure, E.1 Generally (p. 60)",
            content: "E.1.1 Standard disclosure is not the norm in the Commercial Court.",
            subsection_id: "E.1.1",
            category: "Commercial Court"
        },
        {
            sourcefile: "Commercial Court Guide",
            sourcepage: "D.  Case and Costs Management, D.7 Split trials (p. 50)",
            content: "D.7.1 The judge will normally consider whether a split trial is appropriate.",
            subsection_id: "D.7.1",
            category: "Commercial Court"
        },
        {
            sourcefile: "Part 31 – Disclosure And Inspection",
            sourcepage: "Part 31",
            content: "Rule 31.6 Standard disclosure requires a party to disclose documents.",
            subsection_id: "Rule 31.6"
        },
        {
            sourcefile: "Commercial Court Guide",
            sourcepage: "Appendix 1: Overriding Objective (p. 139)",
            content: "Para 5 The parties should consider whether ADR is appropriate.",
            subsection_id: "Para 5",
            category: "Commercial Court"
        },
        {
            sourcefile: "Chancery Guide",
            sourcepage: "Chapter 9 Applications, 9.1 General (p. 45)",
            content: "9.1 Applications should be made by application notice.",
            subsection_id: "9.1",
            category: "Chancery Division"
        }
    ];

    describe("3-part citations", () => {
        it("matches CPR numeric citation to correct data_point", () => {
            const idx = findMatchingContentIndex("1.1, Part 1 – Overriding Objective, Part 1", mockDataPoints);
            expect(idx).toBe(0);
        });

        it("matches PD numeric citation", () => {
            const idx = findMatchingContentIndex("2.1, Practice Direction 35, Part 35", mockDataPoints);
            expect(idx).toBe(1);
        });

        it("matches Rule citation", () => {
            const idx = findMatchingContentIndex("Rule 31.6, Part 31, Part 31 – Disclosure And Inspection", mockDataPoints);
            expect(idx).toBe(4);
        });

        it("matches Para citation", () => {
            const idx = findMatchingContentIndex("Para 5, Appendix 1: Overriding Objective (p. 139), Commercial Court Guide", mockDataPoints);
            expect(idx).toBe(5);
        });

        it("matches Chancery guide citation", () => {
            const idx = findMatchingContentIndex("9.1, Chapter 9 Applications, Chancery Guide", mockDataPoints);
            expect(idx).toBe(6);
        });
    });

    describe("4-part citations (comma in sourcepage)", () => {
        it("matches alpha-dot citation with comma-containing sourcepage", () => {
            // This is parsed as: subsection="E.1.1", sourcePage="E.  Disclosure" (first middle), document="Commercial Court Guide"
            // The key question is whether the simplified matching (parts[1] as sourcePage) finds it
            const idx = findMatchingContentIndex("E.1.1, E.  Disclosure, E.1 Generally (p. 60), Commercial Court Guide", mockDataPoints);
            expect(idx).toBe(2);
        });

        it("matches D.7.1 citation with comma-containing sourcepage", () => {
            const idx = findMatchingContentIndex("D.7.1, D.  Case and Costs Management, D.7 Split trials (p. 50), Commercial Court Guide", mockDataPoints);
            expect(idx).toBe(3);
        });
    });

    describe("2-part citations", () => {
        it("matches by sourcefile", () => {
            const idx = findMatchingContentIndex("Practice Direction 35, Part 35", mockDataPoints);
            expect(idx).toBe(1);
        });
    });

    describe("no match", () => {
        it("returns -1 for non-existent document", () => {
            const idx = findMatchingContentIndex("1.1, Part 99 – Does Not Exist, Part 99", mockDataPoints);
            expect(idx).toBe(-1);
        });

        it("returns -1 for empty citation", () => {
            expect(findMatchingContentIndex("", mockDataPoints)).toBe(-1);
        });
    });

    describe("subsection strict check", () => {
        it("rejects match when subsection not in content and no metadata match", () => {
            const dataPoints: MockDataPoint[] = [
                {
                    sourcefile: "Part 1",
                    sourcepage: "Part 1 – Overriding Objective",
                    content: "Only section 1.2 is mentioned here with no other subsection.",
                    subsection_id: "1.2" // metadata is for 1.2, not 3.5
                }
            ];
            // Citation asks for 3.5 — not in content or metadata
            const idx = findMatchingContentIndex("3.5, Part 1 – Overriding Objective, Part 1", dataPoints);
            expect(idx).toBe(-1);
        });

        it("uses metadata fallback when subsection_id matches but content doesn't contain text", () => {
            const dataPoints: MockDataPoint[] = [
                {
                    sourcefile: "Part 1",
                    sourcepage: "Part 1 – Overriding Objective",
                    content: "The rules establish a procedural code.",
                    subsection_id: "1.1" // metadata matches
                }
            ];
            const idx = findMatchingContentIndex("1.1, Part 1 – Overriding Objective, Part 1", dataPoints);
            expect(idx).toBe(0); // Should match via metadata
        });
    });
});

// ════════════════════════════════════════════════════════════════════════════════
// F. Full Round-Trip: backend citation → frontend parse → navigation
// ════════════════════════════════════════════════════════════════════════════════

describe("Full citation round-trip", () => {
    // These test cases represent the exact citation formats generated by the backend
    // and verify the frontend can parse and navigate to them
    const roundTripCases = [
        {
            name: "CPR numeric",
            citation: "1.1, Part 1 – Overriding Objective, Part 1",
            dp: { sourcefile: "Part 1", sourcepage: "Part 1 – Overriding Objective", content: "1.1 Rules.", subsection_id: "1.1" }
        },
        {
            name: "Court guide alpha_dot",
            citation: "E.1.1, E.  Disclosure, E.1 Generally (p. 60), Commercial Court Guide",
            dp: {
                sourcefile: "Commercial Court Guide",
                sourcepage: "E.  Disclosure, E.1 Generally (p. 60)",
                content: "E.1.1 Standard.",
                subsection_id: "E.1.1"
            }
        },
        {
            name: "Rule subsection",
            citation: "Rule 31.6, Part 31, Part 31 – Disclosure",
            dp: { sourcefile: "Part 31 – Disclosure", sourcepage: "Part 31", content: "Rule 31.6 Standard disclosure.", subsection_id: "Rule 31.6" }
        },
        {
            name: "Para subsection",
            citation: "Para 5, Appendix 1 (p. 139), Commercial Court Guide",
            dp: { sourcefile: "Commercial Court Guide", sourcepage: "Appendix 1 (p. 139)", content: "Para 5 Consider.", subsection_id: "Para 5" }
        }
    ];

    for (const tc of roundTripCases) {
        it(`round-trip: ${tc.name} citation parses and navigates correctly`, () => {
            // Step 1: Parse subsection from citation
            const sub = parseSubsectionFromCitation(tc.citation);
            expect(sub).not.toBeNull();

            // Step 2: Parse citation label parts
            const labelParts = parseCitationLabelParts(tc.citation);
            expect(labelParts.document).toBeTruthy();

            // Step 3: Navigate to matching content
            const idx = findMatchingContentIndex(tc.citation, [tc.dp]);
            expect(idx).toBe(0);

            // Step 4: Extract subsection highlight
            if (sub) {
                const highlight = extractSubsectionContent(tc.dp.content, sub);
                expect(highlight).not.toBeNull();
                expect(highlight!.content).toContain(sub);
            }
        });
    }
});

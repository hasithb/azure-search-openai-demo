import { describe, expect, it } from "vitest";
import { formatAnswerParagraphs } from "../answerParagraphs";

describe("formatAnswerParagraphs", () => {
    it("keeps existing paragraph breaks", () => {
        const input = "First sentence.[1]\n\nSecond paragraph.[2]";
        expect(formatAnswerParagraphs(input)).toBe(input);
    });

    it("groups citation-ended sentences into paragraphs", () => {
        const input = "Sentence one.[1] Sentence two.[2] Sentence three.[3]";
        const expected = "Sentence one.[1] Sentence two.[2]\n\nSentence three.[3]";
        expect(formatAnswerParagraphs(input)).toBe(expected);
    });

    it("falls back to punctuation splitting when no citations", () => {
        const input = "First sentence. Second sentence. Third sentence.";
        const expected = "First sentence. Second sentence.\n\nThird sentence.";
        expect(formatAnswerParagraphs(input)).toBe(expected);
    });

    it("avoids reformatting lists", () => {
        const input = "- Item one\n- Item two\n- Item three";
        expect(formatAnswerParagraphs(input)).toBe(input);
    });

    it("handles filename-based citations without corruption", () => {
        const input =
            "Under employment law, every employee has the right not to be unfairly dismissed [Employment_Rights_Act_1996-5.pdf]. " +
            "An employer must show that the reason for dismissal relates to capability or qualifications [Employment_Rights_Act_1996-5.pdf]. " +
            "Additionally, employers have a duty to ensure the health and safety of all employees [Health_Safety_Regulations-3.pdf]. " +
            "Employment contracts must satisfy the requirements of offer, acceptance, and consideration to be valid [Contract_Law_Handbook-10.pdf].";
        const result = formatAnswerParagraphs(input);
        // Must preserve the full text including all citations
        expect(result).toContain("[Employment_Rights_Act_1996-5.pdf]");
        expect(result).toContain("[Health_Safety_Regulations-3.pdf]");
        expect(result).toContain("[Contract_Law_Handbook-10.pdf]");
        expect(result).toContain("Under employment law");
    });
});

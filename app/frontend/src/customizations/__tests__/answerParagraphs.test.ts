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
});

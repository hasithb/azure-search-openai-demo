import { describe, expect, it } from "vitest";

import { collapseAdjacentCitations } from "./AnswerParser";

describe("collapseAdjacentCitations", () => {
    it("deduplicates identical adjacent citations", () => {
        const input = "The rule is clear.[45][45]";
        const output = collapseAdjacentCitations(input);
        expect(output).toBe("The rule is clear.[45]");
    });

    it("preserves distinct adjacent citations", () => {
        const input = "Disclosure stays proportionate.[1][2]";
        const output = collapseAdjacentCitations(input);
        expect(output).toBe("Disclosure stays proportionate.[1][2]");
    });

    it("leaves non-adjacent citations untouched", () => {
        const input = "Explain the rule.[1] Additional detail here.[2]";
        const output = collapseAdjacentCitations(input);
        expect(output).toBe(input);
    });
});

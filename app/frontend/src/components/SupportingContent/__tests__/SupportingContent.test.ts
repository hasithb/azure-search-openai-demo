import { describe, expect, it } from "vitest";

import { buildDisplayedSupportingItems, resolveTargetSubsection } from "../SupportingContent";

describe("resolveTargetSubsection", () => {
    it("uses the subsection parsed from a citation label when available", () => {
        expect(resolveTargetSubsection("59.9, Part 59, Part 59 - Circuit Commercial Court")).toBe("59.9");
    });

    it("falls back to structured metadata when the citation reference is a content path", () => {
        expect(
            resolveTargetSubsection("/content/Part%2059%20-%20Circuit%20Commercial%20Court", {
                subsectionId: "59.9",
                sourcepage: "Part 59",
                sourcefile: "Part 59 - Circuit Commercial Court",
                category: "Civil Procedure Rules and Practice Directions",
                content: "59.9 If particulars of claim are not served with the claim form...",
                storageUrl: ""
            })
        ).toBe("59.9");
    });

    it("falls back to structured metadata when the citation reference is an external url", () => {
        expect(
            resolveTargetSubsection("https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part59", {
                subsectionId: "59.9",
                sourcepage: "Part 59",
                sourcefile: "Part 59 - Circuit Commercial Court",
                category: "Civil Procedure Rules and Practice Directions",
                content: "59.9 If particulars of claim are not served with the claim form...",
                storageUrl: "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part59"
            })
        ).toBe("59.9");
    });

    it("prefers structured metadata over a coarse citation label", () => {
        expect(
            resolveTargetSubsection("Part 46 – Costs special cases, Part 46", {
                subsectionId: "46.2",
                sourcepage: "Part 46 – Costs special cases",
                sourcefile: "Part 46",
                category: "Civil Procedure Rules and Practice Directions",
                content: "46.2 Where the court is considering whether to exercise its power...",
                storageUrl: "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part-46-costs-special-cases"
            })
        ).toBe("46.2");
    });

    it("returns null when neither the reference nor metadata provides a subsection", () => {
        expect(
            resolveTargetSubsection("/content/Part%2059%20-%20Circuit%20Commercial%20Court", {
                subsectionId: "",
                sourcepage: "Part 59",
                sourcefile: "Part 59 - Circuit Commercial Court",
                category: "Civil Procedure Rules and Practice Directions",
                content: "",
                storageUrl: ""
            })
        ).toBeNull();
    });
});

describe("buildDisplayedSupportingItems", () => {
    const normalizeUrl = (u?: string) => (u || "").toLowerCase();

    it("merges subsection-level items that belong to the same logical section", () => {
        const items = [
            {
                original_doc_id: "part59",
                sourcefile: "Part 59 - Circuit Commercial Court",
                sourcepage: "Part 59",
                subsection_id: "59.9",
                subsection_index: 0,
                content: "59.9 If particulars of claim are not served with the claim form..."
            },
            {
                original_doc_id: "part59",
                sourcefile: "Part 59 - Circuit Commercial Court",
                sourcepage: "Part 59",
                subsection_id: "59.10",
                subsection_index: 1,
                content: "59.10 The defendant must file an acknowledgement of service within 14 days."
            }
        ];

        const result = buildDisplayedSupportingItems(items, normalizeUrl);

        expect(result).toHaveLength(1);
        expect(result[0].full_content).toContain("59.9 If particulars of claim");
        expect(result[0].full_content).toContain("59.10 The defendant must file");
    });

    it("merges items with distinct citations from the same sourcepage into one full-section card", () => {
        const items = [
            {
                original_doc_id: "part59",
                sourcefile: "Part 59 - Circuit Commercial Court",
                sourcepage: "Part 59",
                citation: "59.1, Part 59, Part 59 - Circuit Commercial Court",
                subsection_id: "59.1",
                full_content: "59.1 This Part applies...\n\n59.2 These Rules apply...",
                content: "59.1 This Part applies to claims in Circuit Commercial Court."
            },
            {
                original_doc_id: "part59",
                sourcefile: "Part 59 - Circuit Commercial Court",
                sourcepage: "Part 59",
                citation: "59.2, Part 59, Part 59 - Circuit Commercial Court",
                subsection_id: "59.2",
                full_content: "59.1 This Part applies...\n\n59.2 These Rules apply...",
                content: "59.2 These Rules and their practice directions apply to Circuit Commercial claims."
            }
        ];

        const result = buildDisplayedSupportingItems(items, normalizeUrl);

        // Should merge into a single card showing the full section
        expect(result).toHaveLength(1);
        expect(result[0].full_content).toContain("59.1 This Part applies");
        expect(result[0].full_content).toContain("59.2 These Rules apply");
    });

    it("keeps distinct sourcepages from the same guide as separate cards", () => {
        const items = [
            {
                sourcefile: "King's Bench Division Guide",
                sourcepage: "Annex 6 - Notice of Proposed Allocation to the Multi-Track (p. 232)",
                citation: "rule 29.1, Annex 6 - Notice of Proposed Allocation to the Multi-Track (p. 232), King's Bench Division Guide",
                subsection_id: "rule 29.1",
                content: "rule 29.1 The notice of proposed allocation..."
            },
            {
                sourcefile: "King's Bench Division Guide",
                sourcepage: "The Urgent and Short Applications List, Master's Appointments (p. 74)",
                citation: "9.28, The Urgent and Short Applications List, Master's Appointments (p. 74), King's Bench Division Guide",
                subsection_id: "9.28",
                content: "9.28 Hearing dates for Masters' appointments..."
            }
        ];

        const result = buildDisplayedSupportingItems(items, normalizeUrl);

        expect(result).toHaveLength(2);
        expect(result[0].sourcepage).not.toBe(result[1].sourcepage);
    });

    it("keeps items from different documents as separate cards", () => {
        const items = [
            {
                original_doc_id: "part59",
                sourcefile: "Part 59 - Circuit Commercial Court",
                sourcepage: "Part 59",
                content: "59.1 This Part applies to claims in Circuit Commercial Court."
            },
            {
                original_doc_id: "part58",
                sourcefile: "Part 58 - Proceedings",
                sourcepage: "Part 58",
                content: "58.1 This Part contains rules about proceedings."
            }
        ];

        const result = buildDisplayedSupportingItems(items, normalizeUrl);

        expect(result).toHaveLength(2);
    });
});

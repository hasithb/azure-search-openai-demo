import React, { useEffect } from "react";

interface SupportingContentItem {
    title: string;
    content: string;
    page?: number;
}

interface Props {
    supportingContent: SupportingContentItem[];
    onInfo: (info: string) => void;
    onSupportingContentChange: (content: SupportingContentItem[]) => void;
}

// Add this helper function after the imports
const parseCitationReference = (reference: string): { section?: string; part?: string; title?: string; page?: number } => {
    const result: { section?: string; part?: string; title?: string; page?: number } = {};

    // Extract page number if present (e.g., "(p. 123)")
    const pageMatch = reference.match(/\(p\.\s*(\d+)\)/i);
    if (pageMatch) {
        result.page = parseInt(pageMatch[1]);
    }

    // Extract section number (e.g., "1.1(1)" or "31.1")
    const sectionMatch = reference.match(/^(\d+(?:\.\d+)?(?:\(\d+\))?)/);
    if (sectionMatch) {
        result.section = sectionMatch[1];
    }

    // Extract PART information
    const partMatch = reference.match(/PART\s+(\d+)/i);
    if (partMatch) {
        result.part = partMatch[1];
    }

    // Extract title
    const titleMatch = reference.match(/,\s*([^(]+)(?:\s*\(|$)/);
    if (titleMatch) {
        result.title = titleMatch[1].trim();
    }

    return result;
};

const SupportingContent: React.FC<Props> = ({ supportingContent, onInfo, onSupportingContentChange }) => {
    const activeCitationReference = "1.1(1), PART 1 - OVERRIDING OBJECTIVE (p. 1)";
    const activeCitationContent = "PRACTICE DIRECTION 31 A - DISCLOSURE AND INSPECTION";

    useEffect(() => {
        if (!activeCitationReference || !activeCitationContent || supportingContent.length === 0) {
            return;
        }

        console.log("Auto-scroll effect triggered:", {
            activeCitationReference,
            activeCitationContent: activeCitationContent.substring(0, 200) + "..."
        });

        const scrollTimeout = setTimeout(() => {
            // Parse the citation reference
            const citationParts = parseCitationReference(activeCitationReference);
            console.log("Parsed citation reference:", citationParts);

            // Log all supporting content items for debugging
            console.log("Supporting content items:");
            supportingContent.forEach((item, index) => {
                console.log(`[${index}] ${item.title.substring(0, 50)}...`);
            });

            // Find the best matching supporting content
            let bestMatchIndex = -1;
            let bestMatchScore = 0;

            supportingContent.forEach((item, index) => {
                let score = 0;
                const contentLower = item.content.toLowerCase();
                const titleLower = item.title.toLowerCase();

                // Prepare sectionRegex if needed for use in both scoring and logging
                let sectionRegex: RegExp | undefined = undefined;
                if (citationParts.section) {
                    sectionRegex = new RegExp(
                        `(?:^|\\s|rule\\s*|section\\s*)${citationParts.section.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:\\s|\\.|,|$)`,
                        "i"
                    );
                    if (sectionRegex.test(item.content) || sectionRegex.test(item.title)) {
                        // Make sure it's not a PRACTICE DIRECTION match when looking for section 1.1
                        if (!citationParts.section.startsWith("1.") || !titleLower.includes("practice direction")) {
                            score += 3;
                        }
                    }
                }

                // Check for part match - this should be weighted heavily
                if (citationParts.part) {
                    const partRegex = new RegExp(`\\bPART\\s+${citationParts.part}\\b(?!\\s*\\d)`, "i");
                    if (partRegex.test(item.title)) {
                        score += 10; // High weight for title match
                    } else if (partRegex.test(item.content)) {
                        score += 5;
                    }
                }

                // Check for title match - exact matches should score higher
                if (citationParts.title) {
                    const citationTitleLower = citationParts.title.toLowerCase();
                    if (titleLower.includes(citationTitleLower) || citationTitleLower.includes(titleLower)) {
                        score += 8; // High score for title match
                    } else {
                        // Partial word matching
                        const titleWords = citationParts.title.toLowerCase().split(/\s+/);
                        const matchingWords = titleWords.filter(word => word.length > 3 && (contentLower.includes(word) || titleLower.includes(word)));
                        score += matchingWords.length;
                    }
                }

                // Check for page number proximity
                if (citationParts.page && item.page) {
                    const pageDiff = Math.abs(item.page - citationParts.page);
                    if (pageDiff === 0) {
                        score += 5;
                    } else if (pageDiff <= 2) {
                        score += 3;
                    } else if (pageDiff <= 5) {
                        score += 1;
                    }
                }

                // Penalty for mismatched content type
                if (citationParts.part === "1" && titleLower.includes("practice direction")) {
                    score -= 5; // Penalty for practice direction when looking for Part 1
                }
                console.log(`Item ${index} "${item.title.substring(0, 50)}..." score:`, score, {
                    hasSection: citationParts.section && sectionRegex ? sectionRegex.test(item.content) || sectionRegex.test(item.title) : false,
                    hasPart: citationParts.part ? titleLower.includes(`part ${citationParts.part}`) : false,
                    hasTitle: citationParts.title ? titleLower.includes(citationParts.title.toLowerCase()) : false
                });

                if (score > bestMatchScore) {
                    bestMatchScore = score;
                    bestMatchIndex = index;
                }
            });

            if (bestMatchIndex >= 0 && bestMatchScore > 0) {
                console.log(
                    `Best match found at index ${bestMatchIndex} with score ${bestMatchScore}:`,
                    supportingContent[bestMatchIndex].title.substring(0, 50)
                );
                const element = document.querySelector(`[data-content-index="${bestMatchIndex}"]`);
                if (element) {
                    console.log("Scrolling to element:", element);
                    element.scrollIntoView({ behavior: "smooth", block: "center" });
                }
            } else {
                console.log("No suitable match found for citation");
            }
        }, 100);

        return () => clearTimeout(scrollTimeout);
    }, [activeCitationReference, activeCitationContent, supportingContent]);

    return <div>{/* Render supporting content here */}</div>;
};

export default SupportingContent;

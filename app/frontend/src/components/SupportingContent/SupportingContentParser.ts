import DOMPurify from "dompurify";
// import the correct member from "../../api"
import type * as ApiTypes from "../../api";

// Define SupportingContent type based on usage in the code
type SupportingContent = {
    id?: string;
    original_doc_id?: string;
    sourcefile?: string;
    sourcepage?: string;
    category?: string;
    updated?: string;
    last_updated?: string;
    date_updated?: string;
    storageurl?: string;
    storageUrl?: string;
    storage_url?: string;
    url?: string;
    content?: string;
    full_content?: string;
    date?: string;
};

export type ParsedSupportingContentItem = {
    title: string;
    content: string;
    date?: string;
    updated?: string;
    url?: string;
    sourcefile?: string;
    sourcepage?: string;
    category?: string;
    storageurl?: string;
    id?: string;
};

function extractInlineMetadata(content: string): {
    sourcefile?: string;
    sourcepage?: string;
    category?: string;
    cleanedContent: string;
} {
    if (!content) {
        return { cleanedContent: content };
    }

    const contentHead = content.slice(0, 2000);

    const extractField = (fieldName: "SOURCE" | "SOURCEPAGE" | "CATEGORY") => {
        const pattern = new RegExp(`\\b${fieldName}:\\s*(.+?)(?=\\s+\\b(?:SOURCE|SOURCEPAGE|CATEGORY|SECTION):|\\r?\\n|$)`, "i");
        const match = contentHead.match(pattern);
        return match?.[1]?.trim() || "";
    };

    const sourcefile = extractField("SOURCE");
    const sourcepage = extractField("SOURCEPAGE");
    const category = extractField("CATEGORY");

    let cleanedContent = content;
    if (sourcefile || sourcepage || category) {
        const markers = ["\n## ", "## ", "\nPart ", "Part 1 of", "Part 2 of"];
        const sourceTagIndex = cleanedContent.search(/\bSOURCE:\s*/i);
        const markerIndices = markers.map(marker => cleanedContent.indexOf(marker)).filter(index => index >= 0);
        const firstMarkerIndex = markerIndices.length ? Math.min(...markerIndices) : -1;

        if (sourceTagIndex >= 0 && firstMarkerIndex > sourceTagIndex) {
            cleanedContent = cleanedContent.slice(firstMarkerIndex).trimStart();
        } else {
            cleanedContent = cleanedContent
                .replace(/\bSOURCE:\s*.+?\bSOURCEPAGE:\s*.+?(?=(\n## |## |\nPart |Part 1 of|Part 2 of|$))/is, "")
                .replace(/\bCATEGORY:\s*.+?(?=(\n## |## |\nPart |Part 1 of|Part 2 of|$))/is, "")
                .trim();
        }
    }

    return {
        sourcefile: sourcefile || undefined,
        sourcepage: sourcepage || undefined,
        category: category || undefined,
        cleanedContent
    };
}

export function parseSupportingContentItem(item: any): ParsedSupportingContentItem {
    // Handle null/undefined cases
    if (!item) {
        return {
            title: "Unknown source",
            content: ""
        };
    }

    // For structured object format (which should be the primary format)
    if (typeof item === "object" && item !== null) {
        const inlineContent = item.full_content || item.content || "";
        const inlineMetadata = extractInlineMetadata(inlineContent);

        const sourcepage = item.sourcepage || inlineMetadata.sourcepage || "";
        const sourcefile = item.sourcefile || inlineMetadata.sourcefile || "";
        const category = item.category || inlineMetadata.category || "";
        const updated = item.updated || item.last_updated || item.date_updated || "";
        const storageurl = item.storageurl || item.storageUrl || item.storage_url || item.url || "";
        // Prefer full content if present
        const content = inlineMetadata.cleanedContent;

        // Create title from available fields
        const title = sourcefile || sourcepage || category || "Document Source";

        return {
            title: title,
            content: DOMPurify.sanitize(content),
            date: item.date,
            updated: updated,
            url: storageurl,
            sourcefile: sourcefile,
            sourcepage: sourcepage,
            category: category,
            storageurl: storageurl,
            id: item.id
        };
    }

    // Fallback for string format (legacy support)
    if (typeof item === "string") {
        let sourcepage = "";
        let sourcefile = "";
        let category = "";
        let content = item;

        // Extract from citation pattern: [citation]: content
        const citationMatch = item.match(/^\[([^\]]+)\]:\s*([\s\S]*)$/);
        if (citationMatch) {
            content = citationMatch[2];

            // Parse citation parts: [subsection, category, document]
            const citationParts = citationMatch[1].split(",").map(p => p.trim());

            if (citationParts.length >= 2) {
                sourcepage = citationParts[0]; // First part is sourcepage
                sourcefile = citationParts[citationParts.length - 1]; // Last part is sourcefile

                if (citationParts.length >= 3) {
                    category = citationParts[1]; // Middle part is category
                }
            }
        }

        const title = sourcefile || sourcepage || "Document Source";

        return {
            title: title,
            content: DOMPurify.sanitize(content),
            sourcefile: sourcefile,
            sourcepage: sourcepage,
            category: category
        };
    }

    // Ultimate fallback
    return {
        title: "Unknown source",
        content: DOMPurify.sanitize(String(item))
    };
}

export function extractSubsectionContent(fullContent: string, targetSubsection: string): { content: string; startIndex: number; endIndex: number } | null {
    if (!fullContent || !targetSubsection) {
        return null;
    }

    // Enhanced patterns to find the target subsection with more flexibility
    // Handles markdown headings (## 35.1), breadcrumbs ([PART 35 > 35.1]), and bare text
    const escapedSubsection = escapeRegExp(targetSubsection);
    const patterns = [
        // Markdown heading format: ## 35.1 or ### A.1
        new RegExp(`(^|\\n)\\s*#{1,6}\\s*${escapedSubsection}\\s*(\\n|\\s|$)`, "i"),
        // Breadcrumb format: [PART 35 > 35.1] or [CPR > 35.1]
        new RegExp(`(^|\\n)\\s*\\[[^\\]]*>\\s*${escapedSubsection}\\s*\\]\\s*(\\n|\\s|$)`, "i"),
        // Bracketed subsection markers used in some court guides: [D5.6] or [A.1]
        new RegExp(`(^|\\n)\\s*\\[\\s*${escapedSubsection}\\s*\\]\\s*(\\n|\\s|$)`, "i"),
        // Exact match at start of line or after newline
        new RegExp(`(^|\\n)\\s*${escapedSubsection}\\s*(\\n|\\s|$)`, "i"),
        // Match with optional formatting and punctuation
        new RegExp(`(^|\\n)\\s*${escapedSubsection}\\s*[.:]?\\s*(\\n|\\s|$)`, "i"),
        // Match as part of a larger heading (e.g., "B.7 London Circuit Commercial Court Triaging")
        new RegExp(`(^|\\n)\\s*${escapedSubsection}\\s+[A-Za-z]`, "i"),
        // Match with section markers and optional brackets
        new RegExp(`(^|\\n)\\s*\\(?${escapedSubsection}\\)?\\s*[-\\s]`, "i"),
        // Match anywhere in the content as a fallback
        new RegExp(`\\b${escapedSubsection}\\b`, "i")
    ];

    let targetMatch = null;
    let patternUsed = -1;

    for (let i = 0; i < patterns.length; i++) {
        targetMatch = fullContent.match(patterns[i]);
        if (targetMatch) {
            patternUsed = i;
            break;
        }
    }

    if (!targetMatch) {
        return null;
    }

    // Anchor the start at the actual subsection token so only the cited subsection is highlighted.
    const tokenRegex = new RegExp(`\\b${escapedSubsection}\\b`, "i");
    const localStartOffset = targetMatch[0].search(tokenRegex);
    const subsectionStartIndex = (targetMatch.index ?? 0) + (localStartOffset >= 0 ? localStartOffset : targetMatch[1] ? targetMatch[1].length : 0);
    const startIndex = subsectionStartIndex;

    // Enhanced patterns for finding the next subsection/title/divider boundary
    // Includes markdown headings (## subsection) and breadcrumbs ([PART X > subsection])
    const nextSubsectionPatterns = [
        // Markdown headings: ## 1.1, ### A.1, # Chapter 1
        /\n\s*(#{1,6}\s+(?:[A-Z]\.?\d+|[A-Z]\.?\s+[A-Z][a-z]+|\d+\.\d+|Rule\s+\d+|Chapter\s+\d+|Section\s+\d+|Part\s+\d+))/i,

        // Breadcrumb format: [PART 35 > 35.1], [CPR > A.1]
        /\n\s*(\[[^\]]+>\s*(?:[A-Z]\.?\d+|[A-Z]\.?\s+[A-Z][a-z]+|\d+\.\d+)\])/i,

        // Letter-Dot-Word patterns (A. Preliminary, B. Commencement)
        /\n\s*([A-Z]\.?\s+[A-Z][a-z]+)/i,

        // Letter-Number patterns (A.1, B.2, A1, B2)
        /\n\s*([A-Z]\.?\d+)/i,

        // Numeric patterns (1.1, 1.2.3)
        /\n\s*(\d+\.\d+(?:\.\d+)?)\s/i,

        // Rule/Para references
        /\n\s*(Rule\s+\d+(\.\d+)?|Para\s+\d+(\.\d+)?)\s/i,
        /\n\s*(rule\s+\d+(\.\d+)?|para\s+\d+(\.\d+)?)\s/i,

        // Chapter/Section/Part/Appendix markers
        /\n\s*(Chapter\s+\d+|Section\s+\d+|Part\s+\d+|Appendix\s+\d+)/i,

        // Section dividers
        /\n\s*---/i,
        /\n\s*===+/i,

        // Double newlines preceding another structured identifier (e.g., "D5.2" or "Rule 3.1")
        /\n\s*\n\s*((?:[A-Z]\.?\d+(?:\.\d+)*)|(?:\d+\.\d+(?:\.\d+)?)|(?:Rule\s+\d+)|(?:Para\s+\d+)|(?:Section\s+\d+)|(?:Chapter\s+\d+)|(?:Part\s+\d+))/i,

        // Court guide style: subsection markers separated by blank lines
        /\n\s*\n\s*((?:[A-Z]\.?\d+(?:\.\d+)*)|(?:\d+\.\d+(?:\.\d+)?)|(?:Rule\s+\d+)|(?:Para\s+\d+))/i
    ];

    // Search for the earliest boundary across all patterns (not the first that happens to match).
    // Use the subsection token as the anchor so including a preceding heading in the highlight
    // does not cause the boundary scan to re-read earlier title text.
    const boundarySearchStart = subsectionStartIndex + targetSubsection.length;
    const remainingContent = fullContent.substring(boundarySearchStart);
    let bestBoundaryIndex = Infinity;
    let bestMatch: RegExpMatchArray | null = null;
    let boundaryPatternUsed = -1;
    const sameSubsectionPattern = new RegExp(`\\b${escapedSubsection}\\b`, "i");

    const genericTitleBoundary =
        /\n\s*(#{1,6}\s+[^\n#][^\n]*)\s*\n\s*\n\s*#{1,6}\s*(?:\d+\.\d+(?:\.\d+)?|[A-Z]\.\d+(?:\.\d+)?|[A-Z]\d+(?:\.\d+)?|Rule\s+\d+(?:\.\d+)?|Para\s+\d+(?:\.\d+)?)/i;

    for (let i = 0; i < nextSubsectionPatterns.length; i++) {
        const m = remainingContent.match(nextSubsectionPatterns[i]);
        const boundaryText = (m?.[1] ?? m?.[0] ?? "").trim();
        if (m && boundaryText && sameSubsectionPattern.test(boundaryText)) {
            continue;
        }
        if (m && m.index !== undefined && m.index < bestBoundaryIndex) {
            bestBoundaryIndex = m.index;
            bestMatch = m;
            boundaryPatternUsed = i;
        }
    }

    const titleBoundaryMatch = remainingContent.match(genericTitleBoundary);
    if (titleBoundaryMatch && titleBoundaryMatch.index !== undefined && titleBoundaryMatch.index < bestBoundaryIndex) {
        bestBoundaryIndex = titleBoundaryMatch.index;
        bestMatch = titleBoundaryMatch;
        boundaryPatternUsed = nextSubsectionPatterns.length;
    }

    let endIndex: number;
    if (bestMatch) {
        // End at the earliest boundary (e.g., just before '---' or the next title/subsection)
        endIndex = boundarySearchStart + bestBoundaryIndex;
    } else {
        endIndex = fullContent.length;
    }

    const extractedContent = fullContent.substring(startIndex, endIndex).trim();

    return {
        content: extractedContent,
        startIndex,
        endIndex
    };
}

function escapeRegExp(string: string): string {
    const parts = string
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .map(part => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    if (parts.length === 0) {
        return "";
    }
    return parts.join("[\\s_]+");
}

export function parseSubsectionFromCitation(citation: string): string | null {
    if (!citation) return null;

    const normalizedCitation = citation.replace(/^\s*\d+\.\s+/, "");

    // Parse three-part citation format: [subsection, source, document]
    const citationParts = normalizedCitation
        .split(",")
        .map(p => p.trim())
        .filter(Boolean);
    const rawSubsection = citationParts.length > 0 ? citationParts[0] : "";

    // Support display labels like "D.7.5 - D. Case Management..." (dash-delimited)
    const subsection = rawSubsection.split(/\s*[-–]\s*/)[0].trim();

    if (subsection) {
        // Comprehensive validation for all legal document subsection formats
        const subsectionPatterns = [
            // Numeric patterns
            /^(\d+\.\d+(\.\d+)?)$/i, // 1.1, 1.2.3

            // Letter-Number combinations
            /^([A-Z]\.\d+)$/i, // A.1, B.2, D.5, E.3
            /^([A-Z]\.\d+(?:\.\d+)*)$/i, // D.7.1, A.1.2
            /^([A-Z]\d+\.?\d*)$/i, // A1, B2, D5, A1.1, B2.3

            // Letter-Dot-Word patterns (with flexible spacing)
            /^([A-Z]\.?\s+[A-Z][a-z]+.*?)$/i, // A. Preliminary, A.  Preliminary, B. Commencement, C. Particulars of Claim

            // Rule/Para/Section references
            /^(Rule\s+\d+(\.\d+)?)$/i, // Rule 1.2, Rule 5
            /^(Para\s+\d+(\.\d+)?)$/i, // Para 1.2, Para 3
            /^(rule\s+\d+(\.\d+)?)$/i, // rule 1.2 (lowercase)
            /^(para\s+\d+(\.\d+)?)$/i, // para 1.2 (lowercase)

            // Chapter/Section/Part/Appendix references
            /^(Chapter\s+\d+)$/i, // Chapter 1, Chapter 10
            /^(Section\s+\d+(\.\s+.+)?)$/i, // Section 1, Section 1. Introduction
            /^(Part\s+\d+)$/i, // Part 1, Part 25
            /^(Appendix\s+\d+)$/i, // Appendix 1, Appendix 10

            // Plain word titles (multi-word headings)
            /^([A-Z][a-z]+(\s+[A-Z][a-z]+)+)$/i, // Arbitration Claims, Civil Evidence Act, etc.

            // Catch remaining single letter patterns
            /^([A-Z])$/i // A, B, C (standalone letters)
        ];

        for (const pattern of subsectionPatterns) {
            if (pattern.test(subsection)) {
                return subsection;
            }
        }
    }

    return null;
}

export function parseSupportingContent(supportingContent: SupportingContent[]): ParsedSupportingContent[] {
    const parsedContent: ParsedSupportingContent[] = [];

    for (const item of supportingContent) {
        // Do NOT group/concatenate subsections. Return each item as a standalone entry
        const content = item.full_content || item.content || "";
        parsedContent.push({
            title: item.sourcefile || "Unknown Source",
            sourcefile: item.sourcefile,
            sourcepage: item.sourcepage,
            storageurl: item.storageurl || item.storageUrl || item.url,
            category: item.category,
            updated: item.updated,
            hasContent: !!content,
            fullContent: content,
            id: item.id || item.original_doc_id
        });
    }

    return parsedContent;
}

export interface ParsedSupportingContent {
    title: string;
    sourcefile?: string;
    sourcepage?: string;
    storageurl?: string;
    category?: string;
    updated?: string;
    hasContent: boolean;
    fullContent?: string;
    id?: string;
}

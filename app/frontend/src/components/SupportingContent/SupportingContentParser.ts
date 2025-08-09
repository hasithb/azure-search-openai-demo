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

export function parseSupportingContentItem(item: any): ParsedSupportingContentItem {
    // Handle null/undefined cases
    if (!item) {
        return {
            title: "Unknown source",
            content: ""
        };
    }

    console.log("Parsing item:", item);

    // For structured object format (which should be the primary format)
    if (typeof item === "object" && item !== null) {
        const sourcepage = item.sourcepage || "";
        const sourcefile = item.sourcefile || "";
        const category = item.category || "";
        const updated = item.updated || item.last_updated || item.date_updated || "";
        const storageurl = item.storageurl || item.storageUrl || item.storage_url || item.url || "";
        // Prefer full content if present
        const content = item.full_content || item.content || "";

        // Create title from available fields
        const title = sourcefile || sourcepage || category || "Document Source";

        console.log("Parsed from object:", {
            sourcepage,
            sourcefile,
            category,
            title,
            hasContent: !!content,
            updated,
            storageurl
        });

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

        console.log("Parsed from string:", {
            sourcepage,
            sourcefile,
            category,
            title,
            hasContent: !!content
        });

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

    console.log("Extracting subsection:", targetSubsection, "from content length:", fullContent.length);

    // Enhanced patterns to find the target subsection with more flexibility
    const escapedSubsection = escapeRegExp(targetSubsection);
    const patterns = [
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
            console.log(`Found subsection using pattern ${i}:`, targetMatch[0]);
            break;
        }
    }

    if (!targetMatch) {
        console.log("Target subsection not found with any pattern");
        return null;
    }

    // Anchor the start at the actual subsection token, not at the preceding newline/whitespace
    const tokenRegex = new RegExp(`\\b${escapedSubsection}\\b`, "i");
    const localStartOffset = targetMatch[0].search(tokenRegex);
    const startIndex = (targetMatch.index ?? 0) + (localStartOffset >= 0 ? localStartOffset : targetMatch[1] ? targetMatch[1].length : 0);

    // Enhanced patterns for finding the next subsection/title/divider boundary
    const nextSubsectionPatterns = [
        // Standard patterns for various numbering systems
        /\n\s*([A-Z]?\d+\.\d+(?:\.\d+)?)\s/i,
        /\n\s*([A-Z]\d*\.?\d+\.?\d*)\s/i,
        /\n\s*([A-Z]\.\d+)\s/i,
        /\n\s*(Rule\s+\d+\.\d+|Para\s+\d+\.\d+)\s/i,
        // Section dividers
        /\n\s*---/i,
        // Chapter/Part markers
        /\n\s*(Chapter\s+\d+|Part\s+[A-Z\d]+|Section\s+[A-Z\d]+)\s/i,
        // Double newlines as section breaks (likely titles)
        /\n\s*\n\s*[A-Z]/
    ];

    // Search for the earliest boundary across all patterns (not the first that happens to match)
    const remainingContent = fullContent.substring(startIndex + targetSubsection.length);
    let bestBoundaryIndex = Infinity;
    let bestMatch: RegExpMatchArray | null = null;
    let boundaryPatternUsed = -1;

    for (let i = 0; i < nextSubsectionPatterns.length; i++) {
        const m = remainingContent.match(nextSubsectionPatterns[i]);
        if (m && m.index !== undefined && m.index < bestBoundaryIndex) {
            bestBoundaryIndex = m.index;
            bestMatch = m;
            boundaryPatternUsed = i;
        }
    }

    let endIndex: number;
    if (bestMatch) {
        // End at the earliest boundary (e.g., just before '---' or the next title/subsection)
        endIndex = startIndex + targetSubsection.length + bestBoundaryIndex;
        console.log(`Found next subsection boundary at index ${endIndex}, pattern: ${boundaryPatternUsed}`);
    } else {
        endIndex = fullContent.length;
        console.log("No next subsection found, taking to end of content");
    }

    const extractedContent = fullContent.substring(startIndex, endIndex).trim();

    console.log("Found subsection:", {
        subsection: targetSubsection,
        startIndex,
        endIndex,
        contentLength: extractedContent.length,
        contentPreview: extractedContent.substring(0, 200) + "...",
        patternUsed: boundaryPatternUsed
    });

    return {
        content: extractedContent,
        startIndex,
        endIndex
    };
}

function escapeRegExp(string: string): string {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function parseSubsectionFromCitation(citation: string): string | null {
    if (!citation) return null;

    // Parse three-part citation format: [subsection, source, document]
    const citationParts = citation.split(",").map(p => p.trim());

    if (citationParts.length >= 3) {
        const subsection = citationParts[0].trim();

        // Enhanced validation for different subsection formats
        const subsectionPatterns = [
            /^(\d+\.\d+(\.\d+)?)$/i, // 1.1, 1.2.3
            /^([A-Z]\d*\.?\d+\.?\d*)$/i, // A1.1, B2, D5.6
            /^([A-Z]\.\d+)$/i, // D.5, E.3
            /^(Rule\s+\d+\.\d+)$/i, // Rule 1.2
            /^(Para\s+\d+\.\d+)$/i, // Para 1.2
            /^([A-Z]\d+)$/i // D5, E3
        ];

        for (const pattern of subsectionPatterns) {
            if (pattern.test(subsection)) {
                return subsection;
            }
        }

        console.log("Subsection format not recognized:", subsection);
    }

    return null;
}

export function parseSupportingContent(supportingContent: SupportingContent[]): ParsedSupportingContent[] {
    const parsedContent: ParsedSupportingContent[] = [];

    for (const item of supportingContent) {
        console.log("Parsing item:", item);

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

    console.log("Parsed supporting content (no grouping):", parsedContent);
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

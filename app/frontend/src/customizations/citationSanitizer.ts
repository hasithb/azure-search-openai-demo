/**
 * CUSTOM CITATION SANITIZER
 * =========================
 * This file contains all custom citation formatting logic.
 * It's designed to be merge-safe - upstream updates won't touch this file.
 *
 * To apply after upstream merge:
 * 1. Import sanitizeCitations from this file in AnswerParser.tsx
 * 2. Call sanitizeCitations(answerText) before parsing
 *
 * @author Your Name
 * @created 2025-12-14
 */

const ADJACENT_CITATIONS_REGEX = /\[\d+\](?:\s*\[\d+\])+/g;

/**
 * Collapse only repeated adjacent citations like [45][45] to a single [45].
 * Preserve distinct adjacent citations like [1][3], since those represent
 * multiple sources cited for the same statement.
 */
export function collapseAdjacentCitations(text: string): string {
    return text.replace(ADJACENT_CITATIONS_REGEX, match => {
        const numbers = match.match(/\d+/g);
        if (!numbers || numbers.length === 0) {
            return match;
        }

        const dedupedNumbers = numbers.filter((num, index) => index === 0 || num !== numbers[index - 1]);
        return dedupedNumbers.map(num => `[${num}]`).join("");
    });
}

/**
 * Fix malformed citation patterns where the model outputs unbracketed or incorrectly formatted citations.
 *
 * Patterns fixed:
 * - "1. 1" → "[1]" (duplicated source number without brackets, with space)
 * - "1.1" → "[1]" (duplicated source number without brackets, no space - like N.N)
 * - "proceedings 1." → "proceedings.[1]" (unbracketed citation at end of paragraph)
 * - "[69–81]" → "[69]" (range citations - take first number only)
 * - "[69-81]" → "[69]" (range with regular hyphen)
 * - "[1].[1]" → "[1]" (bracketed duplicates with period)
 *
 * This handles cases where the LLM gets confused between document paragraph numbers
 * (like "1.1", "1.2") and citation indices.
 */
export function fixMalformedCitations(text: string): string {
    let result = text;

    // 0. Fix bracketed duplicates: "[N].[N]", "[N] [N]", "[N]. [N]" or "[N][N]" → "[N]"
    // e.g., "[1].[1]" → "[1]", "[1][1]" → "[1]", "[1]. [1]" → "[1]"
    // Allow optional period and optional spaces between duplicates
    const bracketedDuplicatePattern = /\[(\d{1,3})\]\.?\s*\[\1\]/g;
    result = result.replace(bracketedDuplicatePattern, (_, num) => {
        return `[${num}]`;
    });

    // 1. Fix duplicated citation pattern: "N. N" where both numbers are identical (with space)
    // e.g., "...disclosure 1. 1" → "...disclosure [1]"
    const duplicatedCitationPattern = /(\s)(\d{1,3})\.\s+\2(?=\s|$|[.,;:!?)\]])/g;
    result = result.replace(duplicatedCitationPattern, (_, prefix, num) => {
        return `${prefix}[${num}]`;
    });

    // 1b. Fix duplicated citation pattern: "N.N" where both numbers are identical (NO space)
    // e.g., "...cost 1.1" → "...cost [1]" (but NOT "Section 1.2" which has different numbers)
    // Only match at end of sentence/paragraph or before punctuation
    const duplicatedNoSpacePattern = /(\s)(\d{1,3})\.(\2)(?=\s|$|[,;:!?)\]]|\n)/g;
    result = result.replace(duplicatedNoSpacePattern, (_, prefix, num) => {
        return `${prefix}[${num}]`;
    });

    // 2. Fix unbracketed citation ONLY at end of text (paragraph ending)
    // e.g., "...proceedings 1." at end → "...proceedings.[1]"
    // Only match if it's truly at the end ($ anchor) to avoid false positives
    // Must be preceded by at least 3 word characters to avoid matching "Section 1."
    const unbracketedEndPattern = /(\w{3,})\s+(\d{1,3})\.$/g;
    result = result.replace(unbracketedEndPattern, (_, word, num) => {
        if (String(word).toLowerCase() === "source") {
            return `${word} ${num}.`;
        }
        return `${word}.[${num}]`;
    });

    // 3. Fix range citations: "[N–M]" or "[N-M]" → "[N]" (take first number only)
    // Uses both en-dash (–) and regular hyphen (-)
    const rangeCitationPattern = /\[(\d{1,3})[–\-]\d{1,3}\]/g;
    result = result.replace(rangeCitationPattern, (_, firstNum) => {
        return `[${firstNum}]`;
    });

    // 4. Fix "source N" or "Source N" patterns → "[N]"
    // Matches: "source1", "source 1", "Source1", "Source 1", "(source 1)", etc.
    // Case-insensitive match for "source" followed by optional space and a number
    const sourceNPattern = /\(?source\s*(\d{1,3})\)?(?=\.?($|\s|[.,;:!?)\]]))/gi;
    result = result.replace(sourceNPattern, (_, num) => {
        return `[${num}]`;
    });

    // 5. Fix bare unbracketed citation numbers at sentence boundaries
    // e.g. "...occurs. 1 The Protocol..." → "...occurs.[1] The Protocol..."
    // Also handles chained citations after punctuation such as ". 4 1 Pre-action..."
    const bareCitationPattern = /((?:\]|(?<!\d)[.!?]))\s+(\d{1,2})(?=\s+[A-Z]|\s+\d{1,2}(?:\s|$)|\s*$)/gm;
    let previous = "";
    while (previous !== result) {
        previous = result;
        result = result.replace(bareCitationPattern, (_, before, num) => `${before}[${num}]`);
    }

    return result;
}

/**
 * CUSTOM: Remove CPR section/paragraph references in brackets that look like citations.
 * These are decimal-numbered references such as [7.3] or [7.3(2)] and should not be
 * treated as source citations by the frontend parser.
 */
export function removeDecimalBracketReferences(text: string): string {
    const decimalBracketPattern = /\[\d+\.\d+(?:\(\d+\))?(?:\([a-z]\))?\]\s*/g;
    return text.replace(decimalBracketPattern, "");
}

/**
 * CUSTOM: Remove specific display artifacts identified in legal domain responses.
 *
 * Artifacts removed:
 * 1. Markdown headers for rule numbers: "## 1.1" -> "1.1" (removes formatting, keeps number)
 * 2. Bracketed source headers: "[PART 1 – OVERRIDING OBJECTIVE > 1.1]" -> ""
 *
 * These checks are applied before other sanitizations to ensure cleaner text.
 */
export function removeArtifacts(text: string): string {
    let result = text;

    // Remove "## " prefix from "## 1.1" style headers but KEEP the number
    // Replaces "## 1.1" with "1.1"
    result = result.replace(/##\s+(?=\d)/g, "");

    // Remove "[PART ...]" style headers
    // Using \s* at end to clean up trailing space if the tag was at start of line
    result = result.replace(/\[PART\s+[^\]]+\]\s*/g, "");

    return result;
}

/**
 * CUSTOM: Remove trailing citation lists like:
 *
 * Citation:
 * 1. Source 1
 * 2. Source 2
 *
 * These lists are redundant because the UI renders citations separately.
 */
function removeTrailingCitationList(text: string): string {
    let result = text;

    // Remove trailing blocks that start with "Citation(s):", "Reference(s):", or "Source(s):"
    // followed by any numbered/bulleted list items (handles bare numbers, "Source N", filenames, etc.)
    // e.g. "Citation:\n1. 1" or "Citations:\n1. Source 1\n2. Source 2"
    const trailingCitationBlock = /(?:\r?\n)+\s*(?:Citations?|References?|Sources?)\s*:\s*(?:\r?\n[^\n]*)*\s*$/gi;
    result = result.replace(trailingCitationBlock, "");

    return result;
}

/**
 * Main sanitization function - applies all citation fixes in the correct order.
 * Call this on the raw LLM response before parsing into HTML.
 */
export function sanitizeCitations(text: string): string {
    // 0. Remove visual artifacts (hashes and PART brackets)
    let result = removeArtifacts(text);

    // 0b. Remove trailing citation lists (e.g., "Citation:\n1. Source 1")
    result = removeTrailingCitationList(result);

    // 0c. Remove CPR section references in brackets (e.g. [7.3], [9.1])
    // before attempting to repair malformed source citations.
    result = removeDecimalBracketReferences(result);

    // First fix malformed unbracketed citations like "1. 1" → "[1]"
    result = fixMalformedCitations(result);
    // Then collapse only repeated adjacent citations like [1][1] → [1]
    result = collapseAdjacentCitations(result);
    return result;
}

/**
 * CUSTOM: Normalize a citation string for fuzzy comparison.
 * Strips underscores, chunk IDs, file extensions, and lowercases.
 */
function normalizeCitationForMatch(s: string): string {
    return s
        .replace(/_?chunk_?\d+/gi, "") // remove chunk IDs before underscore conversion
        .replace(/___/g, " ")
        .replace(/_/g, " ")
        .replace(/\.\w{2,4}$/g, "") // remove file extension
        .replace(/,\s*$/g, "") // remove trailing comma from chunk removal
        .replace(/\s+/g, " ")
        .trim()
        .toLowerCase();
}

/**
 * CUSTOM: Find a matching citation in possibleCitations for a given LLM-generated part.
 *
 * The LLM often "humanizes" citations by:
 * - Removing underscores (Practice_Direction → Practice Direction)
 * - Dropping chunk IDs (_chunk_000)
 * - Dropping file extensions (.pdf)
 * - Abbreviating multi-part citations
 *
 * This function first tries the upstream exact endsWith match, then falls back
 * to normalized fuzzy matching.
 *
 * @returns The matched citation string from possibleCitations, or undefined if no match.
 */
export function findMatchingCitation(part: string, possibleCitations: string[]): string | undefined {
    // 1. Exact endsWith match (upstream behavior)
    const exactMatch = possibleCitations.find(c => c.endsWith(part));
    if (exactMatch) return exactMatch;

    // 2. Normalized fuzzy match
    const normalizedPart = normalizeCitationForMatch(part);
    if (normalizedPart.length < 3) return undefined; // too short, skip fuzzy match

    // Find all matches and prefer the one with highest overlap
    let bestMatch: string | undefined;
    let bestScore = 0;

    for (const citation of possibleCitations) {
        const normalizedCitation = normalizeCitationForMatch(citation);
        if (normalizedCitation.includes(normalizedPart)) {
            // Score by how much of the citation the part covers (higher = more specific match)
            const score = normalizedPart.length / normalizedCitation.length;
            if (score > bestScore) {
                bestScore = score;
                bestMatch = citation;
            }
        } else if (normalizedPart.includes(normalizedCitation)) {
            // LLM produced something more detailed than our citation
            const score = normalizedCitation.length / normalizedPart.length;
            if (score > bestScore) {
                bestScore = score;
                bestMatch = citation;
            }
        }
    }

    return bestMatch;
}

/**
 * INTEGRATION INSTRUCTIONS
 * ========================
 *
 * After merging upstream updates, add this import to AnswerParser.tsx:
 *
 *   import { sanitizeCitations, findMatchingCitation } from "../../customizations/citationSanitizer";
 *
 * Then find where answerText is first used and wrap it:
 *
 *   const sanitizedAnswer = sanitizeCitations(answerText);
 *
 * Use sanitizedAnswer for all subsequent processing.
 */

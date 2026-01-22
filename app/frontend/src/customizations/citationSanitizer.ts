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
 * Collapse adjacent bracketed citations like [1][2] or [45][45] to keep only the last one.
 * This prevents the UI from showing multiple redundant citation superscripts.
 */
export function collapseAdjacentCitations(text: string): string {
    return text.replace(ADJACENT_CITATIONS_REGEX, match => {
        const numbers = match.match(/\d+/g);
        if (!numbers || numbers.length === 0) {
            return match;
        }
        const lastCitation = numbers[numbers.length - 1];
        return `[${lastCitation}]`;
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
    const sourceNPattern = /\(?source\s*(\d{1,3})\)?/gi;
    result = result.replace(sourceNPattern, (_, num) => {
        return `[${num}]`;
    });

    return result;
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

    // Remove trailing blocks that start with "Citation:" or "Citations:" and list "Source N" items
    const sourceListBlock = /(?:\r?\n)+\s*(?:Citations?|Citation)\s*:\s*(?:\r?\n\s*\d+[\.)]\s*Source\s*\d+\s*)+\s*$/gi;
    result = result.replace(sourceListBlock, "");

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

    // First fix malformed unbracketed citations like "1. 1" → "[1]"
    result = fixMalformedCitations(result);
    // Then collapse any adjacent bracketed citations like [1][2] → [2]
    result = collapseAdjacentCitations(result);
    return result;
}

/**
 * INTEGRATION INSTRUCTIONS
 * ========================
 *
 * After merging upstream updates, add this import to AnswerParser.tsx:
 *
 *   import { sanitizeCitations } from "../../customizations/citationSanitizer";
 *
 * Then find where answerText is first used and wrap it:
 *
 *   const sanitizedAnswer = sanitizeCitations(answerText);
 *
 * Use sanitizedAnswer for all subsequent processing.
 */

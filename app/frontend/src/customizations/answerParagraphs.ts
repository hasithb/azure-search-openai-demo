// CUSTOM: Answer paragraph formatting helper
// Converts long, single-block answers into readable paragraphs.

const PARAGRAPH_BREAK_REGEX = /\n\s*\n/;
const LIST_MARKER_REGEX = /(^|\n)\s*([-*]|\d+\.)\s+/;

function groupSentences(sentences: string[]): string {
    if (sentences.length < 2) return sentences.join("");

    const sentencesPerParagraph = sentences.length >= 6 ? 3 : 2;
    const paragraphs: string[] = [];

    for (let i = 0; i < sentences.length; i += sentencesPerParagraph) {
        const group = sentences.slice(i, i + sentencesPerParagraph).join(" ");
        paragraphs.push(group);
    }

    return paragraphs.join("\n\n");
}

function splitByCitations(text: string): string[] | null {
    // Match sentences ending with numbered [1] or filename-based [doc.pdf] citations
    const sentenceRegex = /[\s\S]*?\[[^\]]+\](?=\s+|$)/g;
    const matches = [...text.matchAll(sentenceRegex)];
    if (matches.length < 2) return null;

    const sentences: string[] = [];
    let lastIndex = 0;

    for (const match of matches) {
        const raw = match[0];
        const index = match.index ?? 0;
        const trimmed = raw.trim();
        if (trimmed) {
            sentences.push(trimmed);
        }
        lastIndex = index + raw.length;
    }

    const remainder = text.slice(lastIndex).trim();
    if (remainder) {
        sentences.push(remainder);
    }

    return sentences.length >= 2 ? sentences : null;
}

function splitByPunctuation(text: string): string[] | null {
    // Replace citation brackets temporarily to prevent splitting on periods inside them
    // e.g. [Employment_Rights_Act_1996-5.pdf] should not be split at ".pdf"
    const placeholders: string[] = [];
    const masked = text.replace(/\[[^\]]+\]/g, match => {
        placeholders.push(match);
        return `\x00CIT${placeholders.length - 1}\x00`;
    });

    const sentenceRegex = /[^.!?]+[.!?]+(?=\s+|$)/g;
    const matches = masked.match(sentenceRegex) || [];
    if (matches.length < 3) return null;

    // Restore citation placeholders
    const sentences = matches
        .map(s => s.replace(/\x00CIT(\d+)\x00/g, (_, idx) => placeholders[parseInt(idx)]))
        .map(s => s.trim())
        .filter(Boolean);
    return sentences.length >= 3 ? sentences : null;
}

export function formatAnswerParagraphs(text: string): string {
    if (!text) return text;

    // Preserve existing paragraph structure
    if (PARAGRAPH_BREAK_REGEX.test(text)) return text;

    // Avoid reformatting lists or structured markdown
    if (LIST_MARKER_REGEX.test(text)) return text;

    const citationSentences = splitByCitations(text);
    if (citationSentences) {
        return groupSentences(citationSentences);
    }

    const fallbackSentences = splitByPunctuation(text);
    if (fallbackSentences) {
        return groupSentences(fallbackSentences);
    }

    return text;
}

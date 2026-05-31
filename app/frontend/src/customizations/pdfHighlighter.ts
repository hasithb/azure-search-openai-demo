// Primary Source PDF Highlighter
// ==============================
// PDF.js helpers for the Primary Source tab. Renders a single PDF page with a
// real selectable text layer, then locates the cited subsection text within that
// layer and highlights the matching spans so lawyers can validate AI findings
// against the original document.
//
// Part of the merge-safe customizations architecture: all PDF.js usage is
// isolated here and in PrimarySourceViewer.tsx so upstream files stay untouched.

import * as pdfjsLib from "pdfjs-dist";
// Vite resolves the `?url` suffix to the emitted worker asset URL.
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

export type PdfDocument = pdfjsLib.PDFDocumentProxy;

/** Outcome of attempting to highlight a target passage on a rendered page. */
export type HighlightResult = "exact" | "approximate" | "none";

/** Load a PDF document from a same-origin URL (e.g. the /content/ route). */
export async function loadPdfDocument(url: string): Promise<PdfDocument> {
    const loadingTask = pdfjsLib.getDocument({ url, isEvalSupported: false });
    return loadingTask.promise;
}

/**
 * Parse a 1-based page number from a citation path or sourcepage value that may
 * contain a `#page=N` fragment (e.g. "Part_35.pdf#page=3"). Returns null when absent.
 */
export function parsePageNumber(...candidates: (string | undefined)[]): number | null {
    for (const candidate of candidates) {
        if (!candidate) continue;
        const match = candidate.match(/#page=(\d+)/i);
        if (match) {
            const n = parseInt(match[1], 10);
            if (Number.isFinite(n) && n > 0) {
                return n;
            }
        }
    }
    return null;
}

/** Collapse whitespace and lowercase for tolerant text matching. */
function normalizeForMatch(text: string): string {
    return text.replace(/\s+/g, " ").trim().toLowerCase();
}

/**
 * Build candidate "needles" to search for within a page, in priority order.
 * We try the full extracted passage first (most precise), then progressively
 * shorter signatures so OCR/extraction differences still produce a usable hit.
 */
export function buildSearchNeedles(content?: string, subsectionId?: string): string[] {
    const needles: string[] = [];
    const normalizedContent = normalizeForMatch(content || "");

    if (normalizedContent.length >= 12) {
        needles.push(normalizedContent.slice(0, 240));
        needles.push(normalizedContent.slice(0, 120));
        needles.push(normalizedContent.slice(0, 60));
    }

    const sub = (subsectionId || "").trim().toLowerCase();
    if (sub) {
        // Subsection id alone (e.g. "35.4", "h1.7") as a last-resort anchor.
        needles.push(normalizeForMatch(sub));
    }

    // Deduplicate while preserving order.
    return Array.from(new Set(needles.filter(n => n.length >= 3)));
}

interface SpanCharMap {
    normStr: string;
    charSpan: HTMLElement[];
}

/**
 * Build a normalized string for the whole text layer together with a parallel
 * array mapping each normalized character back to its owning <span> element.
 */
function buildSpanCharMap(spans: HTMLElement[]): SpanCharMap {
    let normStr = "";
    const charSpan: HTMLElement[] = [];
    let prevWasSpace = true;

    for (const span of spans) {
        const text = span.textContent || "";
        for (const ch of text) {
            if (/\s/.test(ch)) {
                if (!prevWasSpace) {
                    normStr += " ";
                    charSpan.push(span);
                    prevWasSpace = true;
                }
            } else {
                normStr += ch.toLowerCase();
                charSpan.push(span);
                prevWasSpace = false;
            }
        }
        // Treat each span boundary as a soft space so adjacent spans don't merge words.
        if (!prevWasSpace) {
            normStr += " ";
            charSpan.push(span);
            prevWasSpace = true;
        }
    }

    return { normStr, charSpan };
}

/**
 * Highlight the cited passage within an already-rendered text layer.
 * Adds the provided CSS class to the matching spans and returns the first
 * highlighted span (for scrolling) plus the match quality.
 */
export function highlightInTextLayer(
    textLayerDiv: HTMLElement,
    needles: string[],
    highlightClass: string
): { firstSpan: HTMLElement | null; quality: HighlightResult } {
    const spans = Array.from(textLayerDiv.querySelectorAll<HTMLElement>("span"));
    if (spans.length === 0 || needles.length === 0) {
        return { firstSpan: null, quality: "none" };
    }

    const { normStr, charSpan } = buildSpanCharMap(spans);

    for (let i = 0; i < needles.length; i++) {
        const needle = needles[i];
        const idx = normStr.indexOf(needle);
        if (idx < 0) {
            continue;
        }
        const end = idx + needle.length;
        const matched = new Set<HTMLElement>();
        for (let c = idx; c < end && c < charSpan.length; c++) {
            matched.add(charSpan[c]);
        }
        let firstSpan: HTMLElement | null = null;
        for (const span of spans) {
            if (matched.has(span)) {
                span.classList.add(highlightClass);
                if (!firstSpan) {
                    firstSpan = span;
                }
            }
        }
        // The first (longest, most precise) needle is treated as an exact match;
        // shorter fallback signatures are flagged as approximate.
        return { firstSpan, quality: i === 0 ? "exact" : "approximate" };
    }

    return { firstSpan: null, quality: "none" };
}

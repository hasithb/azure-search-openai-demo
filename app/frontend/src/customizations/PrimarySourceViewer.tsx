// Primary Source Viewer
// =====================
// Renders the live primary source for a citation and highlights the cited
// section/subsection so lawyers can validate AI findings against the original
// document without manually hunting for it.
//
// Strategy ladder (best experience first):
//   1. In-app PDF  -> PDF.js render with a real text-layer highlight (primary)
//   2. PDF, no text match -> render page + extracted-passage banner (approximate)
//   3. HTML / Markdown / image -> embed + extracted-passage banner
//   4. External / iframe-blocked -> banner + "open in new tab"
//
// All PDF.js usage is isolated in pdfHighlighter.ts to keep the merge surface small.

import { useEffect, useMemo, useRef, useState } from "react";

import { isIframeBlocked } from "./externalSourceHandler";
import type { StructuredCitationMetadata } from "./citationMetadata";
import { loadPdfDocument, parsePageNumber, buildSearchNeedles, highlightInTextLayer, type PdfDocument, type HighlightResult } from "./pdfHighlighter";
import styles from "./PrimarySourceViewer.module.css";

import * as pdfjsLib from "pdfjs-dist";

interface PrimarySourceViewerProps {
    /** The citation path (e.g. "/content/Part_35.pdf#page=3") or an external URL. */
    citationFilePath?: string;
    /** Structured metadata carrying the extracted passage text and subsection id. */
    metadata?: StructuredCitationMetadata;
    /** Human-readable citation label for the toolbar. */
    citationLabel?: string;
    /** Height for embedded (non-PDF) sources. */
    height?: string;
    /**
     * Fired when the viewer finishes attempting to locate the cited passage in the
     * primary source. Lets the parent persist a verification tick on the citation.
     */
    onVerified?: (result: HighlightResult) => void;
}

type ViewerStatus = "loading" | "exact" | "approximate" | "none" | "error";

const HIGHLIGHT_CLASS = "pdfHighlight";

const stripFragment = (url: string) => url.split("#")[0].split("?")[0];

const getExtension = (url: string) => stripFragment(url).split(".").pop()?.toLowerCase() || "";

const docNameFromPath = (path: string) => {
    const clean = stripFragment(path);
    const last = clean.split("/").pop() || clean;
    try {
        return decodeURIComponent(last);
    } catch {
        return last;
    }
};

export function PrimarySourceViewer({ citationFilePath, metadata, citationLabel, height = "600px", onVerified }: PrimarySourceViewerProps) {
    const needles = useMemo(() => buildSearchNeedles(metadata?.content, metadata?.subsectionId), [metadata?.content, metadata?.subsectionId]);

    const extension = citationFilePath ? getExtension(citationFilePath) : "";
    const isPdf = extension === "pdf";

    // Non-PDF sources can't be auto-verified inside the panel (cross-document iframe);
    // report "none" once so the parent doesn't show a false verification tick.
    useEffect(() => {
        if (citationFilePath && !isPdf) {
            onVerified?.("none");
        }
    }, [citationFilePath, isPdf, onVerified]);

    if (!citationFilePath) {
        return <div className={styles.message}>No primary source available for this citation.</div>;
    }

    // PDF -> PDF.js text-layer highlight (primary experience). Always prefer an in-app PDF
    // (e.g. the court guides) even when the document also carries an external storageUrl.
    if (isPdf) {
        return (
            <PdfHighlightView citationFilePath={citationFilePath} metadata={metadata} citationLabel={citationLabel} needles={needles} onVerified={onVerified} />
        );
    }

    // Non-PDF: the live primary source is either an explicit external citation path or the
    // document's storageUrl. Most of the corpus (the Civil Procedure Rules) lives as HTML on
    // justice.gov.uk with no in-app PDF, so the citation path is a local /content/<sourcefile>
    // reference and the only embeddable/openable source is the external storageUrl. Resolve it
    // here so iframe-blocking is evaluated against the real source, not the local path.
    const isHttp = (value: string) => value.startsWith("http://") || value.startsWith("https://");
    const storageUrl = metadata?.storageUrl?.trim() || "";
    const externalSource = isHttp(citationFilePath) ? citationFilePath : isHttp(storageUrl) ? storageUrl : "";

    const openInNewTab = () => window.open(externalSource || citationFilePath, "_blank", "noopener,noreferrer");

    // External source that blocks embedding (justice.gov.uk, legislation.gov.uk, ...) -> banner + open in new tab.
    // This is the common case for the Civil Procedure Rules, whose only primary source is the iframe-blocked HTML page.
    if (externalSource && isIframeBlocked(externalSource)) {
        return (
            <div className={styles.primarySource}>
                <PassageBanner metadata={metadata} title="This source can't be embedded" />
                <div className={styles.fallbackBox}>
                    <div className={styles.fallbackIcon}>🚫</div>
                    <p>
                        <strong>The provider blocks display inside other sites.</strong>
                    </p>
                    <p>
                        We can take you straight to the cited passage in a new browser tab. The extracted text is shown above and on the Supporting content tab.
                    </p>
                    <button className={styles.fallbackButton} onClick={openInNewTab}>
                        Open primary source in new tab
                    </button>
                </div>
            </div>
        );
    }

    // Images -> embed directly with extracted-passage banner.
    const docName = citationLabel || docNameFromPath(citationFilePath);
    if (["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(extension)) {
        return (
            <div className={styles.primarySource}>
                <Toolbar docName={docName} where="" status="none" />
                <img src={citationFilePath} alt={docName} style={{ maxWidth: "100%", borderRadius: 8 }} />
            </div>
        );
    }

    // Embeddable HTML / Markdown / other -> embed the live source (external when available) with banner.
    const embedSrc = externalSource || citationFilePath;
    return (
        <div className={styles.primarySource}>
            <Toolbar docName={docName} where={metadata?.subsectionId ? `→ ${metadata.subsectionId}` : ""} status="none" />
            <PassageBanner metadata={metadata} title="Locate this passage in the document below" />
            <iframe title="Primary source" src={embedSrc} className={styles.frame} style={{ height }} />
        </div>
    );
}

// ---------------------------------------------------------------------------
// PDF view
// ---------------------------------------------------------------------------

interface PdfViewProps {
    citationFilePath: string;
    metadata?: StructuredCitationMetadata;
    citationLabel?: string;
    needles: string[];
    onVerified?: (result: HighlightResult) => void;
}

function PdfHighlightView({ citationFilePath, metadata, citationLabel, needles, onVerified }: PdfViewProps) {
    const scrollRef = useRef<HTMLDivElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const docRef = useRef<PdfDocument | null>(null);

    const [numPages, setNumPages] = useState(0);
    const [currentPage, setCurrentPage] = useState(1);
    const [targetPage, setTargetPage] = useState<number | null>(null);
    const [status, setStatus] = useState<ViewerStatus>("loading");

    const docName = citationLabel || docNameFromPath(citationFilePath);
    const pageHint = useMemo(() => parsePageNumber(citationFilePath, metadata?.sourcepage), [citationFilePath, metadata?.sourcepage]);

    // Load the document and resolve the initial target page.
    useEffect(() => {
        let cancelled = false;
        setStatus("loading");
        docRef.current = null;
        setNumPages(0);

        (async () => {
            try {
                const doc = await loadPdfDocument(stripFragment(citationFilePath));
                if (cancelled) return;
                docRef.current = doc;
                setNumPages(doc.numPages);

                let resolved = pageHint && pageHint <= doc.numPages ? pageHint : null;

                // No explicit page hint: scan pages for the strongest needle.
                if (!resolved && needles.length > 0) {
                    resolved = await findPageForNeedle(doc, needles[0]);
                }

                const initial = resolved || 1;
                setTargetPage(resolved);
                setCurrentPage(initial);
            } catch (err) {
                console.error("Failed to load primary source PDF", err);
                if (!cancelled) {
                    setStatus("error");
                    onVerified?.("none");
                }
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [citationFilePath, pageHint, needles]);

    // Render the current page and (when it is the target page) highlight the passage.
    useEffect(() => {
        const doc = docRef.current;
        const wrapper = wrapperRef.current;
        const scroll = scrollRef.current;
        if (!doc || !wrapper || !scroll || currentPage < 1) {
            return;
        }

        let cancelled = false;
        let renderTask: pdfjsLib.RenderTask | null = null;

        (async () => {
            try {
                const page = await doc.getPage(currentPage);
                if (cancelled) return;

                const availableWidth = Math.max(280, scroll.clientWidth - 32);
                const baseViewport = page.getViewport({ scale: 1 });
                const scale = Math.max(0.5, Math.min(2.2, availableWidth / baseViewport.width));
                const viewport = page.getViewport({ scale });
                const dpr = window.devicePixelRatio || 1;

                // Reset the wrapper for this page.
                wrapper.innerHTML = "";
                wrapper.style.width = `${viewport.width}px`;
                wrapper.style.height = `${viewport.height}px`;
                // pdf.js v4 text layer positioning relies on this CSS variable.
                wrapper.style.setProperty("--scale-factor", String(scale));

                const canvas = document.createElement("canvas");
                canvas.className = styles.pageCanvas;
                canvas.width = Math.floor(viewport.width * dpr);
                canvas.height = Math.floor(viewport.height * dpr);
                canvas.style.width = `${viewport.width}px`;
                canvas.style.height = `${viewport.height}px`;
                wrapper.appendChild(canvas);

                const ctx = canvas.getContext("2d");
                if (!ctx) return;
                ctx.scale(dpr, dpr);

                renderTask = page.render({ canvasContext: ctx, viewport });
                await renderTask.promise;
                if (cancelled) return;

                const textLayerDiv = document.createElement("div");
                textLayerDiv.className = styles.textLayer;
                wrapper.appendChild(textLayerDiv);

                const textContent = await page.getTextContent();
                if (cancelled) return;

                const textLayer = new pdfjsLib.TextLayer({
                    textContentSource: textContent,
                    container: textLayerDiv,
                    viewport
                });
                await textLayer.render();
                if (cancelled) return;

                // Highlight only when viewing the target page (or when no specific
                // target page was resolved but we have needles to try here).
                const shouldHighlight = needles.length > 0 && (targetPage === null || currentPage === targetPage);
                if (shouldHighlight) {
                    const { firstSpan, quality } = highlightInTextLayer(textLayerDiv, needles, HIGHLIGHT_CLASS);
                    applyStatus(quality);
                    onVerified?.(quality);
                    if (firstSpan) {
                        setTimeout(() => {
                            if (!cancelled) firstSpan.scrollIntoView({ behavior: "smooth", block: "center" });
                        }, 80);
                    }
                } else {
                    setStatus("none");
                    onVerified?.("none");
                }
            } catch (err) {
                if (!cancelled && (err as Error)?.name !== "RenderingCancelledException") {
                    console.error("Failed to render primary source page", err);
                    setStatus("error");
                    onVerified?.("none");
                }
            }
        })();

        return () => {
            cancelled = true;
            try {
                renderTask?.cancel();
            } catch {
                /* ignore */
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [currentPage, numPages, targetPage, needles]);

    const applyStatus = (quality: HighlightResult) => {
        setStatus(quality === "exact" ? "exact" : quality === "approximate" ? "approximate" : "none");
    };

    const where = metadata?.subsectionId ? `→ ${metadata.subsectionId}${pageHint ? ` (page ${pageHint})` : ""}` : pageHint ? `→ page ${pageHint}` : "";

    return (
        <div className={styles.primarySource}>
            <Toolbar docName={docName} where={where} status={status} />

            {status === "none" && needles.length > 0 && (
                <PassageBanner metadata={metadata} title="Exact passage not auto-located — extracted text shown for reference" />
            )}
            {status === "approximate" && <PassageBanner metadata={metadata} title="Approximate location highlighted — confirm the precise paragraph" />}

            <div className={styles.pdfScroll} ref={scrollRef} style={{ maxHeight: "560px" }}>
                {status === "error" ? (
                    <div className={styles.message}>Couldn't render this PDF in the panel. Use the Supporting content tab or open it in a new tab.</div>
                ) : (
                    <div className={styles.pageWrapper} ref={wrapperRef} />
                )}
            </div>

            {numPages > 1 && status !== "error" && (
                <div className={styles.pageNav}>
                    <button className={styles.navButton} disabled={currentPage <= 1} onClick={() => setCurrentPage(p => Math.max(1, p - 1))}>
                        ‹ Prev
                    </button>
                    <span>
                        Page {currentPage} / {numPages}
                        {targetPage && currentPage !== targetPage ? ` · cited on page ${targetPage}` : ""}
                    </span>
                    <button className={styles.navButton} disabled={currentPage >= numPages} onClick={() => setCurrentPage(p => Math.min(numPages, p + 1))}>
                        Next ›
                    </button>
                    {targetPage && currentPage !== targetPage && (
                        <button className={styles.navButton} onClick={() => setCurrentPage(targetPage)}>
                            Go to citation
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}

/** Scan up to a capped number of pages for the strongest needle. Returns a 1-based page or null. */
async function findPageForNeedle(doc: PdfDocument, needle: string): Promise<number | null> {
    const maxPages = Math.min(doc.numPages, 60);
    for (let p = 1; p <= maxPages; p++) {
        const page = await doc.getPage(p);
        const textContent = await page.getTextContent();
        const pageText = textContent.items
            .map((item: any) => ("str" in item ? item.str : ""))
            .join(" ")
            .replace(/\s+/g, " ")
            .toLowerCase();
        if (pageText.includes(needle)) {
            return p;
        }
    }
    return null;
}

// ---------------------------------------------------------------------------
// Shared sub-components
// ---------------------------------------------------------------------------

function Toolbar({ docName, where, status }: { docName: string; where: string; status: ViewerStatus }) {
    const statusLabel: Record<ViewerStatus, string> = {
        loading: "Locating…",
        exact: "✓ Verified in primary source",
        approximate: "Approximate location",
        none: "Showing source",
        error: "Couldn't render"
    };
    const statusClass: Record<ViewerStatus, string> = {
        loading: styles.statusLoading,
        exact: styles.statusFound,
        approximate: styles.statusApprox,
        none: styles.statusLoading,
        error: styles.statusBlocked
    };

    return (
        <div className={styles.toolbar}>
            <span className={styles.toolbarDoc} title={docName}>
                📄 {docName}
            </span>
            {where && <span className={styles.toolbarWhere}>{where}</span>}
            <span className={styles.spacer} />
            <span className={`${styles.status} ${statusClass[status]}`}>{statusLabel[status]}</span>
        </div>
    );
}

function PassageBanner({ metadata, title }: { metadata?: StructuredCitationMetadata; title: string }) {
    const quote = (metadata?.content || "").trim();
    if (!quote) return null;
    return (
        <div className={styles.banner}>
            <span className={styles.bannerTitle}>{title}</span>
            <div className={styles.bannerQuote}>{quote.length > 600 ? `${quote.slice(0, 600)}…` : quote}</div>
        </div>
    );
}

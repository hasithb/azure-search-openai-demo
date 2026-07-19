import React, { useEffect, useRef, useState, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { parseSupportingContentItem, extractSubsectionContent, parseSubsectionFromCitation } from "./SupportingContentParser";
import styles from "./SupportingContent.module.css";

// CUSTOM: Import external source handler and admin mode check
import {
    isIframeBlocked,
    isAdminMode,
    isFeatureEnabled,
    findBestMatch,
    deduplicatePreservingSubsections,
    CitationMetadataDisplay,
    extractMetadataFromDataPoint
} from "../../customizations";
import type { StructuredCitationMetadata } from "../../customizations";

// CUSTOM: Check if admin mode is enabled
const adminMode = isAdminMode();

interface SupportingContentProps {
    supportingContent: any;
    activeCitationReference?: string;
    activeCitationContent?: string;
    onViewSourceDocument?: (citation: string) => void;
    // CUSTOM: Structured metadata for precise subsection matching
    activeCitationMetadata?: StructuredCitationMetadata;
    // CUSTOM: Switch the analysis panel to the Primary Source tab (live PDF/HTML highlight)
    onShowInPrimarySource?: () => void;
    // CUSTOM: Primary-source verification result for the active citation
    verifiedStatus?: "exact" | "approximate" | "none";
}

export function resolveTargetSubsection(activeCitationReference?: string, activeCitationMetadata?: StructuredCitationMetadata): string | null {
    const metadataSubsection = activeCitationMetadata?.subsectionId?.trim();
    if (metadataSubsection) {
        return metadataSubsection;
    }

    const parsedFromReference = activeCitationReference ? parseSubsectionFromCitation(activeCitationReference) : null;
    return parsedFromReference || null;
}

export function buildDisplayedSupportingItems(
    supportingContent: any[],
    normalizeUrl: (u?: string) => string,
    chooseRepresentative?: (current: any, candidate: any) => any
): any[] {
    type Segment = { idx?: number | null; text: string };
    type DocRecord = {
        bestItem: any;
        hasAnyFull: boolean;
        bestFullText: string;
        segments: Segment[];
        seenTexts: Set<string>;
        bestUpdated: string;
    };
    const byDoc = new Map<string, DocRecord>();

    for (const it of supportingContent || []) {
        const parsed = parseSupportingContentItem(it);
        const docUrl = normalizeUrl(it.storageurl || it.url || parsed.storageurl || parsed.url || "");
        // CUSTOM: Group by logical section within a source document.
        // Merging by sourcefile alone collapses unrelated sections from the same guide into
        // one card, which produces the wrong title/content/highlight when a citation targets
        // a different section from the same source document.
        const sourcefile = (it.sourcefile || parsed.sourcefile || "").trim().toLowerCase();
        const sourcepage = (it.sourcepage || parsed.sourcepage || "").trim().toLowerCase();
        const citation = (it.citation || "").trim().toLowerCase();
        const subsectionId = (it.subsection_id || "").trim().toLowerCase();
        const baseDocKey =
            docUrl ||
            sourcefile ||
            String(it.original_doc_id || "")
                .trim()
                .toLowerCase();
        const sectionKey =
            sourcepage ||
            citation ||
            subsectionId ||
            String(it.original_doc_id || "")
                .trim()
                .toLowerCase();
        const docKey = sectionKey ? `${baseDocKey}::${sectionKey}` : baseDocKey;

        let rec = byDoc.get(docKey);
        const itemUpdated = (it.updated || it.last_updated || it.date_updated || "") as string;

        if (!rec) {
            rec = {
                bestItem: it,
                hasAnyFull: Boolean(it.full_content && it.full_content.length > 0),
                bestFullText: (it.full_content || "") as string,
                segments: [],
                seenTexts: new Set<string>(),
                bestUpdated: itemUpdated
            };
            byDoc.set(docKey, rec);
        } else {
            if (itemUpdated && !rec.bestUpdated) {
                rec.bestUpdated = itemUpdated;
            }
            if (chooseRepresentative) {
                rec.bestItem = chooseRepresentative(rec.bestItem, it);
            } else {
                const existingParsed = parseSupportingContentItem(rec.bestItem);
                const currentScore = (existingParsed.storageurl ? 1 : 0) + (existingParsed.sourcefile ? 1 : 0) + (existingParsed.sourcepage ? 1 : 0);
                const candidateScore = (parsed.storageurl ? 1 : 0) + (parsed.sourcefile ? 1 : 0) + (parsed.sourcepage ? 1 : 0);
                if (candidateScore > currentScore) {
                    rec.bestItem = it;
                }
            }
        }

        if (it.full_content && it.full_content.length > 0) {
            rec.hasAnyFull = true;
            if (it.full_content.length > (rec.bestFullText?.length || 0)) {
                rec.bestFullText = it.full_content;
            }
        }

        const candidateText = (it.content || parsed.content || "").trim();
        if (candidateText && !rec.seenTexts.has(candidateText)) {
            rec.seenTexts.add(candidateText);
            const idx: number | null = typeof it.subsection_index === "number" ? it.subsection_index : typeof it.index === "number" ? it.index : null;
            rec.segments.push({ idx, text: candidateText });
        }
    }

    return Array.from(byDoc.values()).map(rec => {
        const merged = { ...rec.bestItem };
        if (rec.bestUpdated && !merged.updated && !merged.last_updated && !merged.date_updated) {
            merged.updated = rec.bestUpdated;
        }
        if (rec.hasAnyFull && rec.bestFullText && rec.bestFullText.length > 0) {
            merged.full_content = rec.bestFullText;
        } else {
            const sorted = [...rec.segments].sort((a, b) => {
                if (a.idx == null && b.idx == null) return 0;
                if (a.idx == null) return 1;
                if (b.idx == null) return -1;
                return a.idx - b.idx;
            });
            merged.full_content = sorted.map(s => s.text).join("\n\n");
        }
        return merged;
    });
}

// CUSTOM: Helper to normalize DataPoints object or legacy array into a flat array
const toContentArray = (input: any): any[] => {
    if (!input) return [];
    if (Array.isArray(input)) return input;
    // New DataPoints object: merge text + external results
    const items: any[] = [];
    if (Array.isArray(input.text)) items.push(...input.text);
    if (Array.isArray(input.external_results_metadata)) items.push(...input.external_results_metadata);
    return items;
};

export const SupportingContent = ({
    supportingContent: rawSupportingContent,
    activeCitationReference,
    activeCitationContent,
    onViewSourceDocument,
    activeCitationMetadata,
    onShowInPrimarySource,
    verifiedStatus
}: SupportingContentProps) => {
    const supportingContent = toContentArray(rawSupportingContent);
    const { t } = useTranslation();
    const containerRef = useRef<HTMLDivElement>(null);
    const [activeCitation, setActiveCitation] = useState<string>();
    const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

    // Helper: normalize URLs for stable dedup keys
    const normalizeUrl = (u?: string) => {
        if (!u) return "";
        try {
            const url = new URL(u);
            // strip query/hash, keep origin+pathname, trim trailing slash
            let normalized = `${url.origin}${url.pathname}`.replace(/\/+$/, "");
            return normalized.toLowerCase();
        } catch {
            return (u || "")
                .toLowerCase()
                .replace(/[?#].*$/, "")
                .replace(/\/+$/, "");
        }
    };

    // Build a stable, deduplicated list by document and merge subsection chunks back into one display item.
    const displayedItems = useMemo(() => {
        return buildDisplayedSupportingItems(supportingContent, normalizeUrl);
    }, [supportingContent]);

    const formatDate = (dateString: string) => {
        if (!dateString || dateString === "") return "";
        try {
            const date = new Date(dateString);
            if (!isNaN(date.getTime())) {
                return date.toLocaleDateString();
            }
        } catch (e) {
            console.error("Error parsing date:", e);
        }
        return dateString;
    }; // Fixed missing closing brace

    // Add a tiny helper to remove only the optional leading "[n]: " prefix without touching other whitespace
    const stripLeadingIndexPrefix = (s: string) => s.replace(/^\[\d+\]:\s?/, "");

    // Visual-only cleanup for supporting content display (preserves original for matching/highlighting)
    const cleanSupportingContentForDisplay = (s: string) => {
        // Visual-only cleanup: preserve line structure for readability
        const lines = s.split(/\r?\n/);
        const cleaned = lines.map(line => {
            let updated = line;
            // Remove markdown heading markers but keep the canonical heading text.
            updated = updated.replace(/^#{1,6}\s*/g, "");
            // Remove inline bracketed metadata blocks, wherever they appear in the line
            updated = updated.replace(/\[[^\]]*(PRACTICE\s*DIRECTION|PD\s*\d+|PART\s+\d+|SECTION\s+\d+|APPENDIX|>)[^\]]*\]\s*/gi, "");
            return updated;
        });

        // Drop standalone bracketed metadata lines (e.g., [PRACTICE DIRECTION ... > 1.1 ...])
        // and header cruft from non-CPR sources (Document:, Section:, Part N of M, ==== dividers)
        const filtered = cleaned.filter(line => {
            const trimmed = line.trim();
            // Remove "Document: ..." header lines
            if (/^Document:\s/i.test(trimmed)) return false;
            // Remove "Section: ..." or "Section:..." header lines
            if (/^Section:\s*/i.test(trimmed)) return false;
            // Remove "Part N of M" pagination markers (but keep "Part 1" section headings)
            if (/^Part\s+\d+\s+of\s+\d+$/i.test(trimmed)) return false;
            // Remove ==== divider lines (4+ consecutive = chars)
            if (/^={4,}$/.test(trimmed)) return false;
            // Remove standalone "Contents" lines and "[Title] Contents" lines
            if (/^Contents$/i.test(trimmed)) return false;
            if (/\]\s*Contents$/i.test(trimmed) && trimmed.startsWith("[")) return false;
            // Keep the selected subsection heading even when the index uses a breadcrumb
            // form such as "[PART 24 > 24.2 ...]"; the browser oracle needs that identity.
            if (targetSubsection && normalizeSubsectionToken(trimmed).includes(normalizeSubsectionToken(targetSubsection))) {
                return true;
            }

            // Handle bracketed metadata
            if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
                return true;
            }
            // Keep numeric citations like [1] if they ever appear here
            if (/^\[\d+\]$/.test(trimmed)) {
                return true;
            }
            // Remove metadata lines with bracketed titles or section markers
            const isMetadata = /\b(PRACTICE\s*DIRECTION|PD\s*\d+|PART\s+\d+|SECTION\s+\d+|APPENDIX|>)/i.test(trimmed);
            return !isMetadata;
        });

        // Deduplicate identical non-empty lines (even across blanks, e.g., repeated document titles)
        const deduped: string[] = [];
        for (const line of filtered) {
            const trimmed = line.trim();
            if (trimmed) {
                // Compare against last non-blank line to catch dupes separated by blank lines
                const lastNonBlank = [...deduped].reverse().find(l => l.trim() !== "");
                if (lastNonBlank !== undefined && lastNonBlank.trim() === trimmed) {
                    continue;
                }
            }
            deduped.push(line);
        }

        const insertSubsectionBreaks = (line: string) => {
            let updated = line;

            // Insert breaks before inline numeric subsections like "1.2" when chained in one line
            updated = updated.replace(/(\s)(\d+\.\d+(?:\.\d+)?)(?=\s+[^\d])/g, "$1\n\n$2");

            // Insert breaks before inline court guide markers like "A.1", "A1.1", "D5.1"
            updated = updated.replace(/(\s)([A-Z]\.??\d+(?:\.\d+)*)(?=\s+[^\d])/g, "$1\n\n$2");

            // Insert breaks before single-level section headings like "3. Duties"
            updated = updated.replace(/(\s)(\d+\.)(?=\s+[A-Z])/g, "$1\n\n$2");

            // Insert breaks before inline structured markers like "Appendix 2" or "Part 7"
            updated = updated.replace(/(\s)((?:Appendix|Part|Section|Chapter|Rule|Para)\s+\d+)(?=\s*[:–-]?\s+)/gi, "$1\n\n$2");

            return updated;
        };

        // Add a blank line between numbered subsections for readability
        const withSpacing: string[] = [];
        for (let i = 0; i < deduped.length; i++) {
            const line = insertSubsectionBreaks(deduped[i]);
            const splitLines = line.split(/\n/);
            for (const part of splitLines) {
                const trimmed = part.trim();
                const isNumbered = /^\d+(?:\.\d+)?\b/.test(trimmed);
                const isLettered = /^[A-Z]\.??\d+(?:\.\d+)*\b/.test(trimmed);
                const prev = withSpacing.length > 0 ? withSpacing[withSpacing.length - 1] : "";
                const prevIsBlank = prev.trim() === "";
                if ((isNumbered || isLettered) && withSpacing.length > 0 && !prevIsBlank) {
                    withSpacing.push("");
                }
                withSpacing.push(part);
            }
        }

        return withSpacing.join("\n");
    };

    const normalizeSubsectionToken = (s?: string) =>
        (s || "")
            .trim()
            .replace(/\s+/g, " ")
            .replace(/[\s.:]+$/, "")
            .toLowerCase();

    const normalizeMatchText = (s?: string) => (s || "").toLowerCase().replace(/\s+/g, " ").trim();

    const formatSupportingContentHtml = (text: string, options?: { highlight?: boolean; sourceInfo?: string }) => {
        const normalized = text.replace(/\r\n/g, "\n").trim();
        if (!normalized) return "";

        const escapedSourceInfo = (options?.sourceInfo || "").replace(/"/g, "&quot;").replace(/'/g, "&#39;");

        const paragraphs = normalized.split(/\n\s*\n+/g);
        return paragraphs
            .map((paragraph, index) => {
                const lines = paragraph
                    .split(/\n/g)
                    .map(line => line.trimEnd())
                    .filter(Boolean);
                if (lines.length === 0) return "";
                const paragraphBody = lines.join("<br/>");
                if (!options?.highlight) {
                    return `<p>${paragraphBody}</p>`;
                }

                const markId = index === 0 ? ' id="highlighted-subsection"' : "";
                const markTitle = index === 0 && escapedSourceInfo ? ` title="${escapedSourceInfo}"` : "";
                return `<p><mark${markId}${markTitle} style="background-color:#3b82f6;color:#fff;padding:0 4px;border-radius:4px;display:inline;line-height:inherit;scroll-margin-top:20px;">${paragraphBody}</mark></p>`;
            })
            .filter(Boolean)
            .join("");
    };

    // Enhanced content rendering with subsection highlighting
    const renderContent = (content: string, isHighlighted: boolean = false, targetSubsection?: string, sourceInfo?: string) => {
        if (!content) return null;

        // NO CLEANING - Use the original content structure as created in the search index
        const originalContent = stripLeadingIndexPrefix(content); // Only drop a leading "[n]: " if present
        const displayContent = cleanSupportingContentForDisplay(originalContent);

        // If we have a target subsection and this item is highlighted, highlight that section within the full content
        if (isHighlighted && targetSubsection) {
            // Use the robust extractor to find the subsection block
            const section = extractSubsectionContent(originalContent, targetSubsection);

            if (section && section.content) {
                const beforeSubsection = originalContent.substring(0, section.startIndex);
                const subsectionContent = section.content;
                const afterSubsection = originalContent.substring(section.endIndex);

                const formattedBeforeContent = formatSupportingContentHtml(cleanSupportingContentForDisplay(beforeSubsection));
                const formattedHighlightedContent = formatSupportingContentHtml(cleanSupportingContentForDisplay(subsectionContent), {
                    highlight: true,
                    sourceInfo
                });
                const formattedAfterContent = formatSupportingContentHtml(cleanSupportingContentForDisplay(afterSubsection));
                const fullSectionWithHighlight = `${formattedBeforeContent}${formattedHighlightedContent}${formattedAfterContent}`;

                return (
                    <div className={styles.itemContent}>
                        <div style={{ fontFamily: "inherit", margin: 0, lineHeight: "1.4" }}>
                            <div dangerouslySetInnerHTML={{ __html: fullSectionWithHighlight }} />
                        </div>
                    </div>
                );
            } else {
                // CUSTOM: Fallback – the merged card's full_content may come from a different
                // chunk than the one that contains the cited subsection (e.g. PD 51R 2.1 is in
                // chunk_000 but the bestFullText may be chunk_006 covering section 7+).
                // Use activeCitationMetadata.content directly when it carries the subsection text.
                const fallbackRaw = activeCitationMetadata?.content;
                if (fallbackRaw && fallbackRaw.trim()) {
                    const fallbackOriginal = stripLeadingIndexPrefix(fallbackRaw);
                    const formattedHighlightedContent = formatSupportingContentHtml(cleanSupportingContentForDisplay(fallbackOriginal), {
                        highlight: true,
                        sourceInfo
                    });
                    return (
                        <div className={styles.itemContent}>
                            <div style={{ fontFamily: "inherit", margin: 0, lineHeight: "1.4" }}>
                                <div dangerouslySetInnerHTML={{ __html: formattedHighlightedContent }} />
                            </div>
                        </div>
                    );
                }
                const formattedDisplayContent = formatSupportingContentHtml(displayContent);
                return (
                    <div className={styles.itemContent}>
                        <div
                            id="highlighted-subsection"
                            style={{ fontFamily: "inherit", margin: 0, lineHeight: "1.4", scrollMarginTop: "20px" }}
                            dangerouslySetInnerHTML={{ __html: formattedDisplayContent }}
                        />
                    </div>
                );
            }
        }

        // Show original content without any cleaning to preserve index structure
        const formattedDisplayContent = formatSupportingContentHtml(displayContent);
        return (
            <div className={styles.itemContent}>
                <div style={{ fontFamily: "inherit", margin: 0, lineHeight: "1.4" }} dangerouslySetInnerHTML={{ __html: formattedDisplayContent }} />
            </div>
        );
    };

    // Enhanced function to find matching content with subsection awareness over displayedItems
    const findMatchingContentIndex = (citation: string): number => {
        if (!citation) return -1;

        // CUSTOM: Fast direct match — with the numbered citation pipeline, the citation
        // reference is the exact enhanced citation string (e.g., "35.1, Part 35, Part_35.pdf")
        // which matches the `citation` field on data points directly.
        const normalizedCitationFull = normalizeMatchText(citation);
        for (let i = 0; i < displayedItems.length; i++) {
            const itemCitation = normalizeMatchText(displayedItems[i]?.citation);
            if (itemCitation && itemCitation === normalizedCitationFull) {
                return i;
            }
            // Also check sourcepage match for simpler citations
            const parsedItem = parseSupportingContentItem(displayedItems[i]);
            const itemSourcepage = normalizeMatchText(parsedItem.sourcepage);
            if (itemSourcepage && itemSourcepage === normalizedCitationFull) {
                return i;
            }
        }

        let bestMatchIndex = -1;
        let bestMatchScore = 0;

        for (let i = 0; i < displayedItems.length; i++) {
            const parsedItem = parseSupportingContentItem(displayedItems[i]);
            const rawItem: any = displayedItems[i];
            let score = 0;

            const normalizedCitation = citation.replace(/^\s*\d+\.\s+/, "");
            const citationParts = normalizedCitation
                .split(",")
                .map(p => p.trim())
                .filter(Boolean);

            const parsedSubsection = parseSubsectionFromCitation(normalizedCitation) || "";

            if (citationParts.length >= 3) {
                const subsection = citationParts[0];
                const sourcePage = citationParts[1];
                const document = citationParts.slice(2).join(", ");

                const sourcepageMatches =
                    normalizeMatchText(parsedItem.sourcepage) === normalizeMatchText(sourcePage) ||
                    (parsedItem.sourcepage && sourcePage && normalizeMatchText(parsedItem.sourcepage).includes(normalizeMatchText(sourcePage))) ||
                    (parsedItem.sourcepage && sourcePage && normalizeMatchText(sourcePage).includes(normalizeMatchText(parsedItem.sourcepage)));

                const documentMatches =
                    normalizeMatchText(parsedItem.sourcefile) === normalizeMatchText(document) ||
                    (parsedItem.sourcefile && document && normalizeMatchText(parsedItem.sourcefile).includes(normalizeMatchText(document))) ||
                    (document && normalizeMatchText(document).includes(normalizeMatchText(parsedItem.sourcefile || "")));

                if (documentMatches) {
                    score += 10;
                } else if (!sourcepageMatches) {
                    continue;
                }

                // Sourcepage: prefer exact, otherwise light partial
                if (parsedItem.sourcepage && sourcePage) {
                    if (parsedItem.sourcepage === sourcePage) {
                        score += 50;
                    } else {
                        const sourcePageLower = sourcePage.toLowerCase();
                        const parsedSourcepageLower = parsedItem.sourcepage.toLowerCase();
                        if (sourcePageLower.length > 3 && parsedSourcepageLower.includes(sourcePageLower)) {
                            score += 10;
                        } else if (parsedSourcepageLower.length > 3 && sourcePageLower.includes(parsedSourcepageLower)) {
                            score += 10;
                        }
                    }
                } else {
                    // If we can't compare sourcepage, skip early for strictness
                    continue;
                }

                // STRICT: if we have a subsection, it MUST appear in the content to be a valid match
                const subsectionToken = parsedSubsection || subsection;
                if (subsectionToken && subsectionToken.length > 1) {
                    const content = parsedItem.content || "";
                    if (!content) {
                        continue;
                    }
                    const escaped = subsectionToken.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                    const patterns = [new RegExp(`(^|\\n)\\s*${escaped}\\b`, "i"), new RegExp(`\\b${escaped}\\b`, "i")];
                    const found = patterns.some(p => p.test(content));
                    if (!found) {
                        const normalizedSubsection = normalizeSubsectionToken(subsectionToken);
                        const normalizedMetaSubsection = normalizeSubsectionToken(rawItem?.subsection_id);
                        const sourcepageMatches =
                            parsedItem.sourcepage === sourcePage ||
                            (parsedItem.sourcepage && sourcePage && (parsedItem.sourcepage.includes(sourcePage) || sourcePage.includes(parsedItem.sourcepage)));
                        const subsectionMetaMatch = normalizedSubsection && normalizedMetaSubsection && normalizedSubsection === normalizedMetaSubsection;

                        if (sourcepageMatches || subsectionMetaMatch) {
                            score += 15;
                        } else {
                            continue; // do not consider this item at all
                        }
                    } else {
                        // If present, add a high score
                        score += 40;
                    }
                }
            } else if (citationParts.length === 2) {
                // Two-part citation: could be (subsection, sourcefile) or (sourcepage, sourcefile)
                const partA = citationParts[0];
                const partB = citationParts[1];
                const dashParts = partA.split(/\s*[-–]\s*/);
                const dashSubsection = dashParts[0]?.trim() || "";
                const dashDocument = dashParts.length > 1 ? dashParts.slice(1).join(" - ").trim() : "";
                const subsectionToken = parsedSubsection || dashSubsection || partA;
                const inferredDocument = dashDocument || "";
                const inferredSourcePage = partB;

                const sourcepageMatches =
                    normalizeMatchText(parsedItem.sourcepage) === normalizeMatchText(inferredSourcePage) ||
                    (parsedItem.sourcepage &&
                        inferredSourcePage &&
                        normalizeMatchText(parsedItem.sourcepage).includes(normalizeMatchText(inferredSourcePage))) ||
                    (parsedItem.sourcepage && inferredSourcePage && normalizeMatchText(inferredSourcePage).includes(normalizeMatchText(parsedItem.sourcepage)));

                const documentMatches =
                    !!inferredDocument &&
                    (normalizeMatchText(parsedItem.sourcefile) === normalizeMatchText(inferredDocument) ||
                        (parsedItem.sourcefile && normalizeMatchText(parsedItem.sourcefile).includes(normalizeMatchText(inferredDocument))) ||
                        normalizeMatchText(inferredDocument).includes(normalizeMatchText(parsedItem.sourcefile || "")));

                if (documentMatches) {
                    score += 30;
                } else if (sourcepageMatches) {
                    score += 25;
                } else {
                    continue;
                }

                // Check subsection presence in content or metadata
                if (subsectionToken && parsedItem.content) {
                    const escaped = subsectionToken.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                    const patterns = [new RegExp(`(^|\\n)\\s*${escaped}\\b`, "i"), new RegExp(`\\b${escaped}\\b`, "i")];
                    if (patterns.some(p => p.test(parsedItem.content || ""))) {
                        score += 25;
                    } else if (parsedSubsection) {
                        const normalizedSubsection = normalizeSubsectionToken(subsectionToken);
                        const normalizedMetaSubsection = normalizeSubsectionToken(rawItem?.subsection_id);
                        if (normalizedSubsection && normalizedMetaSubsection && normalizedSubsection === normalizedMetaSubsection) {
                            score += 15;
                        }
                    }
                }
            } else {
                // Single-part or legacy format
                if (parsedItem.sourcepage && citation.includes(parsedItem.sourcepage)) {
                    score += 15;
                } else if (parsedItem.sourcefile && citation.includes(parsedItem.sourcefile)) {
                    score += 10;
                } else {
                    continue;
                }
            }

            if (score > bestMatchScore && score >= 15) {
                bestMatchScore = score;
                bestMatchIndex = i;
            }
        }

        if (bestMatchIndex >= 0 && bestMatchScore >= 15) {
            return bestMatchIndex;
        }

        return -1;
    };

    // Auto-scroll uses displayedItems
    useEffect(() => {
        if ((activeCitationReference || activeCitationContent) && containerRef.current) {
            const matchIndex = findMatchingContentIndex(activeCitationReference || "");

            if (matchIndex >= 0) {
                const targetElement = containerRef.current.children[matchIndex] as HTMLElement;
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: "smooth", block: "start" });
                    targetElement.style.backgroundColor = "#dbeafe";
                    setTimeout(() => {
                        targetElement.style.backgroundColor = "";
                    }, 5000);

                    // Scroll to the subsection start, or to the content start when no subsection was found.
                    setTimeout(() => {
                        const highlightedMark = targetElement.querySelector("#highlighted-subsection");
                        if (highlightedMark) {
                            const elementTop = targetElement.getBoundingClientRect().top;
                            const markTop = highlightedMark.getBoundingClientRect().top;
                            const relativeTop = markTop - elementTop;
                            if (relativeTop > 180) {
                                highlightedMark.scrollIntoView({ behavior: "smooth", block: "start" });
                            }
                        }
                    }, 300);
                }
            }
        }
    }, [activeCitationReference, activeCitationContent, activeCitationMetadata, displayedItems]);

    // Handle view source document
    const handleViewSourceDocument = (parsedItem: any) => {
        // Handle both camelCase and lowercase field names for storage URL
        const documentUrl = parsedItem.storageurl || parsedItem.storageUrl || parsedItem.storage_url || parsedItem.url;
        if (documentUrl && onViewSourceDocument) {
            onViewSourceDocument(documentUrl);
        }
    };

    // Handle view source document in new tab
    const handleViewSourceDocumentNewTab = (parsedItem: any) => {
        // Handle both camelCase and lowercase field names for storage URL
        const documentUrl = parsedItem.storageurl || parsedItem.storageUrl || parsedItem.storage_url || parsedItem.url;
        if (documentUrl) {
            window.open(documentUrl, "_blank", "noopener,noreferrer");
        }
    };

    if (!displayedItems || displayedItems.length === 0) {
        return (
            <div className={styles.supportingContent}>
                <p>No supporting content available</p>
            </div>
        );
    }

    const targetSubsection = resolveTargetSubsection(activeCitationReference, activeCitationMetadata);

    // CUSTOM: Compute once instead of per-item to avoid O(n²) matching
    // When structured metadata is available, use findBestMatch for precise matching
    const structuredMatchIndex =
        activeCitationMetadata && isFeatureEnabled("structuredCitationMatching") ? findBestMatch(activeCitationMetadata, displayedItems) : -1;
    const textMatchIndex = activeCitationReference ? findMatchingContentIndex(activeCitationReference) : -1;
    const activeMatchIndex = structuredMatchIndex >= 0 ? structuredMatchIndex : textMatchIndex;

    return (
        <div className={styles.supportingContent} ref={containerRef}>
            {displayedItems.map((item, index) => {
                const parsedItem = parseSupportingContentItem(item);
                const isActive = activeMatchIndex === index;
                const rawItem = item as Record<string, string | undefined> | undefined;
                const sourcefile = parsedItem.sourcefile || rawItem?.sourcefile;
                const sourcepage = parsedItem.sourcepage || rawItem?.sourcepage;
                const category = parsedItem.category || rawItem?.category;
                const getDisplayTitle = () => {
                    const parts = [sourcefile, sourcepage, category].map(part => (part || "").trim()).filter(Boolean);
                    const uniqueParts: string[] = [];
                    for (const part of parts) {
                        const partLower = part.toLowerCase();
                        // Skip exact duplicates (case-insensitive)
                        if (uniqueParts.some(existing => existing.toLowerCase() === partLower)) {
                            continue;
                        }
                        // Skip truncated sourcefiles that are just a prefix of another part
                        // e.g., "Pre" when "Pre-Action Protocol for Disease..." is present
                        if (part.length <= 10) {
                            const isPrefix = parts.some(other => {
                                const otherLower = (other || "").toLowerCase().trim();
                                return otherLower !== partLower && otherLower.length > partLower.length && otherLower.startsWith(partLower);
                            });
                            if (isPrefix) {
                                continue;
                            }
                        }
                        uniqueParts.push(part);
                    }
                    return uniqueParts.join(", ") || "Document Source";
                };

                const displayTitle = getDisplayTitle();

                // Handle both camelCase and lowercase field names for storage URL
                // Cast to any to handle multiple possible field names from backend
                const itemAny = item as any;
                const parsedAny = parsedItem as any;
                const documentUrl = parsedAny.storageurl || parsedAny.storageUrl || parsedAny.storage_url || parsedAny.url;
                const hasDocumentUrl = Boolean(documentUrl);
                const isBlocked = documentUrl ? isIframeBlocked(documentUrl) : false;

                // Create a stable key per document (match dedup logic)
                const docUrl = normalizeUrl(
                    itemAny.storageurl ||
                        itemAny.storageUrl ||
                        itemAny.storage_url ||
                        itemAny.url ||
                        parsedAny.storageurl ||
                        parsedAny.storageUrl ||
                        parsedAny.storage_url ||
                        parsedAny.url ||
                        ""
                );
                const docKey = (itemAny.original_doc_id || docUrl || parsedItem.sourcefile || "") + `_${index}`;

                return (
                    <div key={docKey} className={`${styles.supportingItem} ${isActive ? styles.highlighted : ""}`}>
                        <div className={styles.itemHeader}>
                            <div className={styles.itemTitle} title={displayTitle} aria-label={displayTitle}>
                                {displayTitle}
                            </div>
                            {parsedItem.updated && (
                                <div className={styles.itemMeta}>
                                    <span className={styles.itemDate}>
                                        <strong>Updated:</strong> {formatDate(parsedItem.updated)}
                                    </span>
                                </div>
                            )}
                            {/* CUSTOM: Show structured metadata badges when enabled */}
                            {isFeatureEnabled("citationMetadataDisplay") && <CitationMetadataDisplay metadata={extractMetadataFromDataPoint(item)} />}
                        </div>

                        {/* Always render full content; highlight specific subsection if active */}
                        {renderContent(parsedItem.content, isActive, targetSubsection ?? undefined, displayTitle)}

                        <div className={styles.supportingContentActions}>
                            {/* CUSTOM: Jump to the live primary source with the cited section highlighted */}
                            {isActive && onShowInPrimarySource && (
                                <button
                                    className={`${styles.primarySourceButton} ${
                                        verifiedStatus === "exact"
                                            ? styles.primarySourceButtonVerified
                                            : verifiedStatus === "approximate"
                                              ? styles.primarySourceButtonApprox
                                              : ""
                                    }`}
                                    onClick={onShowInPrimarySource}
                                    title={
                                        verifiedStatus === "exact"
                                            ? "Verified: the cited text was located in the live primary source. Click to open the Primary Source tab and view it highlighted."
                                            : verifiedStatus === "approximate"
                                              ? "A close match was located in the live primary source. Click to open the Primary Source tab and review it."
                                              : "Open the Primary Source tab to load the live document and highlight this section"
                                    }
                                >
                                    {verifiedStatus === "exact"
                                        ? "Verified in primary source — open"
                                        : verifiedStatus === "approximate"
                                          ? "Likely match — open in primary source"
                                          : "Show in primary source"}
                                </button>
                            )}
                            {hasDocumentUrl && (
                                <>
                                    {/* Only show "View Source" (in-panel) for admins; everyone else gets "View Source in New Tab" */}
                                    {adminMode && !isBlocked && (
                                        <button
                                            className={styles.viewSourceButton}
                                            onClick={() => handleViewSourceDocument(parsedItem)}
                                            title="View Source Document"
                                        >
                                            View Source
                                        </button>
                                    )}
                                    {(!adminMode || isBlocked) && (
                                        <button
                                            className={styles.viewSourceButton}
                                            onClick={() => handleViewSourceDocumentNewTab(parsedItem)}
                                            title="View Source Document in New Tab"
                                        >
                                            View Source in New Tab
                                        </button>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

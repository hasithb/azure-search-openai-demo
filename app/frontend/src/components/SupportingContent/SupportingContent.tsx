import React, { useEffect, useRef, useState, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { parseSupportingContentItem, extractSubsectionContent, parseSubsectionFromCitation } from "./SupportingContentParser";
import styles from "./SupportingContent.module.css";

// CUSTOM: Import external source handler and admin mode check
import { isIframeBlocked, isAdminMode } from "../../customizations";

// CUSTOM: Check if admin mode is enabled
const adminMode = isAdminMode();

interface SupportingContentProps {
    supportingContent: any[];
    activeCitationReference?: string;
    activeCitationContent?: string;
    onViewSourceDocument?: (citation: string) => void;
}

export const SupportingContent = ({ supportingContent, activeCitationReference, activeCitationContent, onViewSourceDocument }: SupportingContentProps) => {
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

    // Build a stable, deduplicated list by document and MERGE all subsection chunks into a single full_content
    const displayedItems = useMemo(() => {
        type Segment = { idx?: number | null; text: string };
        type DocRecord = {
            bestItem: any;
            hasAnyFull: boolean;
            bestFullText: string;
            segments: Segment[];
            seenTexts: Set<string>;
        };
        const byDoc = new Map<string, DocRecord>();

        for (const it of supportingContent || []) {
            const parsed = parseSupportingContentItem(it);
            const docUrl = normalizeUrl(it.storageurl || it.url || parsed.storageurl || parsed.url || "");
            const docKey = it.original_doc_id || docUrl || (parsed.sourcefile || "").toLowerCase();

            let rec = byDoc.get(docKey);
            if (!rec) {
                rec = {
                    bestItem: it,
                    hasAnyFull: Boolean(it.full_content && it.full_content.length > 0),
                    bestFullText: (it.full_content || "") as string,
                    segments: [],
                    seenTexts: new Set<string>()
                };
                byDoc.set(docKey, rec);
            } else {
                // Prefer the item that has storageurl/sourcefile/sourcepage populated; otherwise keep first
                const existingParsed = parseSupportingContentItem(rec.bestItem);
                const currentScore = (existingParsed.storageurl ? 1 : 0) + (existingParsed.sourcefile ? 1 : 0) + (existingParsed.sourcepage ? 1 : 0);
                const candidateScore = (parsed.storageurl ? 1 : 0) + (parsed.sourcefile ? 1 : 0) + (parsed.sourcepage ? 1 : 0);
                if (candidateScore > currentScore) {
                    rec.bestItem = it;
                }
            }

            // Track full_content if any item provides it; prefer the longest
            if (it.full_content && it.full_content.length > 0) {
                rec.hasAnyFull = true;
                if (it.full_content.length > (rec.bestFullText?.length || 0)) {
                    rec.bestFullText = it.full_content;
                }
            }

            // Accumulate subsection content segments to reconstruct full content if backend didn't send it
            const candidateText = (it.content || parsed.content || "").trim();
            if (candidateText && !rec.seenTexts.has(candidateText)) {
                rec.seenTexts.add(candidateText);
                const idx: number | null = typeof it.subsection_index === "number" ? it.subsection_index : typeof it.index === "number" ? it.index : null;
                rec.segments.push({ idx, text: candidateText });
            }
        }

        // Finalize merged entries with injected full_content
        return Array.from(byDoc.values()).map(rec => {
            const merged = { ...rec.bestItem };
            if (rec.hasAnyFull && rec.bestFullText && rec.bestFullText.length > 0) {
                merged.full_content = rec.bestFullText;
            } else {
                // Sort by subsection_index when available, otherwise preserve insertion order
                const sorted = [...rec.segments].sort((a, b) => {
                    if (a.idx == null && b.idx == null) return 0;
                    if (a.idx == null) return 1;
                    if (b.idx == null) return -1;
                    return a.idx - b.idx;
                });
                // Join unique segments with double newline to preserve paragraph breaks
                merged.full_content = sorted.map(s => s.text).join("\n\n");
            }
            return merged;
        });
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
            // Remove markdown heading markers like "## 1.1" but keep the number/text
            updated = updated.replace(/^##\s*/g, "");
            // Remove inline bracketed metadata blocks, wherever they appear in the line
            updated = updated.replace(/\[[^\]]*(PRACTICE\s*DIRECTION|PD\s*\d+|PART\s+\d+|SECTION\s+\d+|APPENDIX|>)[^\]]*\]\s*/gi, "");
            return updated;
        });

        // Drop standalone bracketed metadata lines (e.g., [PRACTICE DIRECTION ... > 1.1 ...])
        const filtered = cleaned.filter(line => {
            const trimmed = line.trim();
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
        for (let i = 0; i < filtered.length; i++) {
            const line = insertSubsectionBreaks(filtered[i]);
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

    // CUSTOM: Convert cleaned supporting content into paragraph HTML using \n\n or markdown-style spacing
    const formatSupportingContentHtml = (text: string) => {
        const normalized = text.replace(/\r\n/g, "\n").trim();
        if (!normalized) return "";

        const paragraphs = normalized.split(/\n\s*\n+/g);
        return paragraphs
            .map(paragraph => {
                const lines = paragraph
                    .split(/\n/g)
                    .map(line => line.trimEnd())
                    .filter(Boolean);
                if (lines.length === 0) return "";
                return `<p>${lines.join("<br/>")}</p>`;
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
            console.log(`Attempting to highlight subsection: ${targetSubsection} in content length: ${originalContent.length}`);

            // Use the robust extractor to find the subsection block
            const section = extractSubsectionContent(originalContent, targetSubsection);

            if (section && section.content) {
                const beforeSubsection = originalContent.substring(0, section.startIndex);
                const subsectionContent = section.content;
                const afterSubsection = originalContent.substring(section.endIndex);

                // Reduce vertical padding to avoid overlapping the previous line
                // Add id for scrolling and title for hover tooltip with full source information
                // Escape quotes and HTML entities for the title attribute
                const escapedSourceInfo = (sourceInfo || "").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
                console.log("Setting highlight tooltip - sourceInfo:", sourceInfo, "escaped:", escapedSourceInfo);
                const highlightedSubsection =
                    `<mark id="highlighted-subsection" title="${escapedSourceInfo}" style="background-color:#3b82f6;color:#fff;padding:0 4px;border-radius:4px;display:inline;line-height:inherit;scroll-margin-top:20px;">` +
                    subsectionContent +
                    `</mark>`;
                const highlightedContent = cleanSupportingContentForDisplay(beforeSubsection + highlightedSubsection + afterSubsection);
                const formattedHighlightedContent = formatSupportingContentHtml(highlightedContent);

                return (
                    <div className={styles.itemContent}>
                        <div
                            style={{ fontFamily: "inherit", margin: 0, lineHeight: "1.4" }}
                            dangerouslySetInnerHTML={{ __html: formattedHighlightedContent }}
                        />
                    </div>
                );
            } else {
                console.warn(`Could not find subsection ${targetSubsection} in content`);
                console.log(`Content starts with: ${originalContent.substring(0, 200)}...`);
                const formattedDisplayContent = formatSupportingContentHtml(displayContent);
                return (
                    <div className={styles.itemContent}>
                        <div style={{ fontFamily: "inherit", margin: 0, lineHeight: "1.4" }} dangerouslySetInnerHTML={{ __html: formattedDisplayContent }} />
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

        console.log("Finding match for citation:", citation);

        let bestMatchIndex = -1;
        let bestMatchScore = 0;

        for (let i = 0; i < displayedItems.length; i++) {
            const parsedItem = parseSupportingContentItem(displayedItems[i]);
            const rawItem: any = displayedItems[i];
            let score = 0;

            console.log(`Checking displayed item ${i}:`, {
                sourcepage: parsedItem.sourcepage,
                sourcefile: parsedItem.sourcefile,
                category: parsedItem.category
            });

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

                console.log(`Citation parts:`, { subsection, sourcePage, document });

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
                    console.log(`Document mismatch for item ${i}: expected '${document}', got '${parsedItem.sourcefile}'`);
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
                    console.log(`Missing sourcepage for strict match on item ${i}`);
                    continue;
                }

                // STRICT: if we have a subsection, it MUST appear in the content to be a valid match
                const subsectionToken = parsedSubsection || subsection;
                if (subsectionToken && subsectionToken.length > 1) {
                    const content = parsedItem.content || "";
                    if (!content) {
                        console.log(`Item ${i} has no content to check subsection presence`);
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
                            console.log(`Subsection '${subsection}' not found in content for item ${i}; using metadata fallback`);
                            score += 15;
                        } else {
                            console.log(`Subsection '${subsection}' not found in item ${i} content, skipping`);
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

                console.log(`Two-part citation:`, { partA, partB, dashSubsection, dashDocument, inferredDocument, inferredSourcePage });

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
                    console.log(`Document match for two-part citation`);
                } else if (sourcepageMatches) {
                    score += 25;
                    console.log(`Sourcepage match for two-part citation`);
                } else {
                    console.log(`Document/sourcepage mismatch for two-part citation`);
                    continue;
                }

                // Check subsection presence in content or metadata
                if (subsectionToken && parsedItem.content) {
                    const escaped = subsectionToken.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                    const patterns = [new RegExp(`(^|\\n)\\s*${escaped}\\b`, "i"), new RegExp(`\\b${escaped}\\b`, "i")];
                    if (patterns.some(p => p.test(parsedItem.content || ""))) {
                        score += 25;
                        console.log(`Subsection '${subsectionToken}' found in content`);
                    } else if (parsedSubsection) {
                        const normalizedSubsection = normalizeSubsectionToken(subsectionToken);
                        const normalizedMetaSubsection = normalizeSubsectionToken(rawItem?.subsection_id);
                        if (normalizedSubsection && normalizedMetaSubsection && normalizedSubsection === normalizedMetaSubsection) {
                            score += 15;
                            console.log(`Subsection '${subsectionToken}' matched metadata`);
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
            console.log(`Best match (displayedItems) found at index ${bestMatchIndex} with score ${bestMatchScore}`);
            return bestMatchIndex;
        }

        console.log("No valid match found for citation (strict subsection check)");
        return -1;
    };

    // Auto-scroll uses displayedItems
    useEffect(() => {
        if ((activeCitationReference || activeCitationContent) && containerRef.current) {
            const matchIndex = findMatchingContentIndex(activeCitationReference || "");

            console.log("Auto-scroll effect (displayedItems):", {
                activeCitationReference,
                activeCitationContent,
                matchIndex,
                displayedItemsLength: displayedItems.length
            });

            if (matchIndex >= 0) {
                const targetElement = containerRef.current.children[matchIndex] as HTMLElement;
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: "smooth", block: "start" });
                    targetElement.style.backgroundColor = "#dbeafe";
                    setTimeout(() => {
                        targetElement.style.backgroundColor = "";
                    }, 5000);

                    // Scroll to the highlighted mark within the section if it exists
                    setTimeout(() => {
                        const highlightedMark = targetElement.querySelector("#highlighted-subsection");
                        if (highlightedMark) {
                            highlightedMark.scrollIntoView({ behavior: "smooth", block: "center" });
                        }
                    }, 300);
                }
            }
        }
    }, [activeCitationReference, activeCitationContent, displayedItems]);

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

    const targetSubsection = activeCitationReference ? parseSubsectionFromCitation(activeCitationReference) : null;

    return (
        <div className={styles.supportingContent} ref={containerRef}>
            {displayedItems.map((item, index) => {
                const parsedItem = parseSupportingContentItem(item);
                const isActive = !!(activeCitationReference && findMatchingContentIndex(activeCitationReference) === index);

                const getDisplayTitle = () => {
                    const parts: string[] = [];
                    const rawItem = item as Record<string, string | undefined> | undefined;
                    const sourcefile = parsedItem.sourcefile || rawItem?.sourcefile;
                    const sourcepage = parsedItem.sourcepage || rawItem?.sourcepage;
                    const category = parsedItem.category || rawItem?.category;
                    if (sourcefile) parts.push(sourcefile);
                    if (sourcepage) parts.push(sourcepage);
                    if (category) parts.push(category);
                    const title = parts.length > 0 ? parts.join(", ") : "Document Source";
                    console.log("getDisplayTitle:", {
                        sourcefile: parsedItem.sourcefile,
                        sourcepage: parsedItem.sourcepage,
                        category: parsedItem.category,
                        rawSourcefile: rawItem?.sourcefile,
                        rawSourcepage: rawItem?.sourcepage,
                        rawCategory: rawItem?.category,
                        title
                    });
                    return title;
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
                        </div>

                        {/* Always render full content; highlight specific subsection if active */}
                        {renderContent(parsedItem.content, isActive, targetSubsection ?? undefined, displayTitle)}

                        <div className={styles.supportingContentActions} style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
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

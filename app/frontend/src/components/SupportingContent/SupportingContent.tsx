import React, { useEffect, useRef, useState, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { parseSupportingContentItem, extractSubsectionContent, parseSubsectionFromCitation } from "./SupportingContentParser";
import styles from "./SupportingContent.module.css";

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

    // Enhanced content rendering with subsection highlighting
    const renderContent = (content: string, isHighlighted: boolean = false, targetSubsection?: string) => {
        if (!content) return null;

        // NO CLEANING - Use the original content structure as created in the search index
        const originalContent = stripLeadingIndexPrefix(content); // Only drop a leading "[n]: " if present

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
                const highlightedSubsection =
                    `<mark style="background-color:#3b82f6;color:#fff;padding:0 4px;border-radius:4px;display:inline;line-height:inherit;">` +
                    subsectionContent +
                    `</mark>`;
                const highlightedContent = beforeSubsection + highlightedSubsection + afterSubsection;

                return (
                    <div className={styles.itemContent}>
                        <div
                            style={{ whiteSpace: "pre-wrap", fontFamily: "inherit", margin: 0, lineHeight: "1.4" }}
                            dangerouslySetInnerHTML={{ __html: highlightedContent }}
                        />
                    </div>
                );
            } else {
                console.warn(`Could not find subsection ${targetSubsection} in content`);
                console.log(`Content starts with: ${originalContent.substring(0, 200)}...`);
            }
        }

        // Show original content without any cleaning to preserve index structure
        return (
            <div className={styles.itemContent}>
                <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit", margin: 0, lineHeight: "1.4" }}>{originalContent}</pre>
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
            let score = 0;

            console.log(`Checking displayed item ${i}:`, {
                sourcepage: parsedItem.sourcepage,
                sourcefile: parsedItem.sourcefile,
                category: parsedItem.category
            });

            const citationParts = citation.split(",").map(p => p.trim());

            if (citationParts.length >= 3) {
                const subsection = citationParts[0];
                const sourcePage = citationParts[1];
                const document = citationParts[2];

                console.log(`Citation parts:`, { subsection, sourcePage, document });

                // Document must match
                if (!(parsedItem.sourcefile === document || parsedItem.sourcefile?.includes(document))) {
                    console.log(`Document mismatch for item ${i}: expected '${document}', got '${parsedItem.sourcefile}'`);
                    continue;
                }
                score += 10;

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
                if (subsection && subsection.length > 1) {
                    const content = parsedItem.content || "";
                    if (!content) {
                        console.log(`Item ${i} has no content to check subsection presence`);
                        continue;
                    }
                    const escaped = subsection.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                    const patterns = [new RegExp(`(^|\\n)\\s*${escaped}\\b`, "i"), new RegExp(`\\b${escaped}\\b`, "i")];
                    const found = patterns.some(p => p.test(content));
                    if (!found) {
                        console.log(`Subsection '${subsection}' not found in item ${i} content, skipping`);
                        continue; // do not consider this item at all
                    }
                    // If present, add a high score
                    score += 40;
                }
            } else {
                // Legacy two-part citation
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
                }
            }
        }
    }, [activeCitationReference, activeCitationContent, displayedItems]);

    // Handle view source document
    const handleViewSourceDocument = (parsedItem: any) => {
        const documentUrl = parsedItem.storageurl || parsedItem.url;
        if (documentUrl && onViewSourceDocument) {
            onViewSourceDocument(documentUrl);
        }
    };

    // Handle view source document in new tab
    const handleViewSourceDocumentNewTab = (parsedItem: any) => {
        const documentUrl = parsedItem.storageurl || parsedItem.url;
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
                    if (parsedItem.sourcefile) parts.push(parsedItem.sourcefile);
                    if (parsedItem.sourcepage) parts.push(parsedItem.sourcepage);
                    if (parsedItem.category) parts.push(parsedItem.category);
                    return parts.length > 0 ? parts.join(", ") : "Document Source";
                };

                const documentUrl = parsedItem.storageurl || parsedItem.url;
                const hasDocumentUrl = Boolean(documentUrl);

                // Create a stable key per document (match dedup logic)
                const docUrl = normalizeUrl(item.storageurl || item.url || parsedItem.storageurl || parsedItem.url || "");
                const docKey = (item.original_doc_id || docUrl || parsedItem.sourcefile || "") + `_${index}`;

                return (
                    <div key={docKey} className={`${styles.supportingItem} ${isActive ? styles.highlighted : ""}`}>
                        <div className={styles.itemHeader}>
                            <div className={styles.itemTitle}>{getDisplayTitle()}</div>
                            {parsedItem.updated && (
                                <div className={styles.itemMeta}>
                                    <span className={styles.itemDate}>
                                        <strong>Updated:</strong> {formatDate(parsedItem.updated)}
                                    </span>
                                </div>
                            )}
                        </div>

                        {/* Always render full content; highlight specific subsection if active */}
                        {renderContent(parsedItem.content, isActive, targetSubsection ?? undefined)}

                        <div className={styles.supportingContentActions} style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                            {hasDocumentUrl && (
                                <>
                                    <button
                                        className={styles.viewSourceButton}
                                        onClick={() => handleViewSourceDocument(parsedItem)}
                                        title="View Source Document"
                                    >
                                        View Source
                                    </button>
                                    <button
                                        className={styles.viewSourceButton}
                                        onClick={() => handleViewSourceDocumentNewTab(parsedItem)}
                                        title="View Source Document in New Tab"
                                    >
                                        View Source in New Tab
                                    </button>
                                </>
                            )}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

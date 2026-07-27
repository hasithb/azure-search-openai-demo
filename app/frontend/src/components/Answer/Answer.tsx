import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@fluentui/react-components";
import { Copy24Regular, Checkmark24Regular, LightbulbFilament24Regular, ClipboardTextLtr24Regular } from "@fluentui/react-icons";
import { useTranslation } from "react-i18next";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

import styles from "./Answer.module.css";
import { ChatAppResponse, getCitationFilePath, SpeechConfig } from "../../api";
import { parseAnswerToHtml } from "./AnswerParser";
import { AnswerIcon } from "./AnswerIcon";
import { SpeechOutputBrowser } from "./SpeechOutputBrowser";
import { SpeechOutputAzure } from "./SpeechOutputAzure";
// CUSTOM: Import admin mode check to gate Thought Process visibility
import { buildCitationLabel, buildCitationPath, isAdminMode } from "../../customizations";
// CUSTOM: Import structured citation metadata type
import type { StructuredCitationMetadata } from "../../customizations";

interface Props {
    answer: ChatAppResponse;
    index: number;
    speechConfig: SpeechConfig;
    isSelected?: boolean;
    isStreaming: boolean;
    // CUSTOM: Optional second arg passes citation content for display enrichment
    //         Optional third arg passes structured metadata for precise matching
    onCitationClicked: (filePath: string, content?: string, metadata?: StructuredCitationMetadata, selectionId?: string) => void;
    onThoughtProcessClicked: () => void;
    onSupportingContentClicked: () => void;
    onFollowupQuestionClicked?: (question: string) => void;
    showFollowupQuestions?: boolean;
    showSpeechOutputBrowser?: boolean;
    showSpeechOutputAzure?: boolean;
}

export const Answer = ({
    answer,
    index,
    speechConfig,
    isSelected,
    isStreaming,
    onCitationClicked,
    onThoughtProcessClicked,
    onSupportingContentClicked,
    onFollowupQuestionClicked,
    showFollowupQuestions,
    showSpeechOutputAzure,
    showSpeechOutputBrowser
}: Props) => {
    const followupQuestions = answer.context?.followup_questions;
    const parsedAnswer = useMemo(() => parseAnswerToHtml(answer, isStreaming, onCitationClicked), [answer, isStreaming, onCitationClicked]);
    const { t } = useTranslation();
    const sanitizedAnswerHtml = DOMPurify.sanitize(parsedAnswer.answerHtml);
    const [copied, setCopied] = useState(false);
    // CUSTOM: Gate citation list visibility to avoid flicker during streaming
    const [showCitations, setShowCitations] = useState(false);
    useEffect(() => {
        if (isStreaming) {
            setShowCitations(false);
            return;
        }
        const timer = setTimeout(() => setShowCitations(true), 150);
        return () => clearTimeout(timer);
    }, [isStreaming]);
    // CUSTOM: Only show Thought Process button in admin mode
    const adminMode = isAdminMode();

    const getDataPointsArray = useCallback((dataPoints: any): any[] => {
        if (!dataPoints) {
            return [];
        }

        if (Array.isArray(dataPoints)) {
            return dataPoints;
        }

        if (dataPoints.text && Array.isArray(dataPoints.text)) {
            return dataPoints.text.map((textItem: any, itemIndex: number) => {
                if (typeof textItem === "string" && textItem.length > 0) {
                    const urlMatch = textItem.match(/^(https?:\/\/[^:]+):\s*/);
                    if (urlMatch) {
                        const content = textItem.substring(urlMatch[0].length);
                        return {
                            id: itemIndex,
                            content,
                            storageUrl: urlMatch[1],
                            sourcepage: `Source ${itemIndex + 1}`,
                            sourcefile: content.substring(0, 50) + "..."
                        };
                    }

                    return {
                        id: itemIndex,
                        content: textItem,
                        sourcepage: `Source ${itemIndex + 1}`,
                        sourcefile: textItem.substring(0, 50) + "..."
                    };
                }

                if (textItem && typeof textItem === "object") {
                    return {
                        id: itemIndex,
                        ...textItem,
                        content: textItem.content || "",
                        storageUrl: textItem.storageUrl || textItem.storageurl || textItem.url || "",
                        sourcepage: textItem.sourcepage || `Source ${itemIndex + 1}`,
                        sourcefile: textItem.sourcefile || ""
                    };
                }

                return {
                    id: itemIndex,
                    content: String(textItem || ""),
                    sourcepage: `Source ${itemIndex + 1}`,
                    sourcefile: ""
                };
            });
        }

        return [];
    }, []);

    type ParsedCitationLabel =
        | { kind: "full"; subsection: string; sourcePage: string; sourceFile: string }
        | { kind: "two"; partA: string; partB: string }
        | { kind: "single"; single: string };

    const parseCitationLabel = useCallback((citation: string): ParsedCitationLabel => {
        const parts = citation
            .split(",")
            .map(part => part.trim())
            .filter(Boolean);

        if (parts.length >= 3) {
            return {
                kind: "full",
                subsection: parts[0] ?? "",
                sourcePage: parts.slice(1, -1).join(", ") || "",
                sourceFile: parts[parts.length - 1] ?? ""
            };
        }

        if (parts.length === 2) {
            return {
                kind: "two",
                partA: parts[0] ?? "",
                partB: parts[1] ?? ""
            };
        }

        return { kind: "single", single: citation };
    }, []);

    const findMatchingSupportingContent = useCallback(
        (citation: string) => {
            if (!answer.context?.data_points) {
                return undefined;
            }

            const dataPointsArray = getDataPointsArray(answer.context.data_points);
            const parsedCitation = parseCitationLabel(citation);

            if (parsedCitation.kind === "full") {
                const { subsection, sourcePage, sourceFile } = parsedCitation;
                const escaped = subsection.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                const subsectionPattern = new RegExp(`(^|\\n)\\s*${escaped}\\b`, "i");

                // Stage 0: exact match on subsection_id + sourcepage + sourcefile
                const exactAll = dataPointsArray.find(dp => {
                    return (
                        String(dp.subsection_id || "").trim() === subsection &&
                        String(dp.sourcepage || "").trim() === sourcePage &&
                        String(dp.sourcefile || "").trim() === sourceFile
                    );
                });
                if (exactAll) return exactAll;

                // Stage 1: subsection in content start + exact page/file
                const contentMeta = dataPointsArray.find(dp => {
                    return (
                        subsectionPattern.test(String(dp.content || "")) &&
                        String(dp.sourcepage || "").trim() === sourcePage &&
                        String(dp.sourcefile || "").trim() === sourceFile
                    );
                });
                if (contentMeta) return contentMeta;

                // Stage 2: exact sourcepage + sourcefile
                const exact = dataPointsArray.find(dp => {
                    return String(dp.sourcepage || "").trim() === sourcePage && String(dp.sourcefile || "").trim() === sourceFile;
                });
                if (exact) return exact;

                // Stage 3: fuzzy includes (partial match on sourcepage or sourcefile)
                const fuzzy = dataPointsArray.find(dp => {
                    const dpSourcepage = String(dp.sourcepage || "").trim();
                    const dpSourcefile = String(dp.sourcefile || "").trim();
                    return (
                        dpSourcepage.includes(sourcePage) ||
                        sourcePage.includes(dpSourcepage) ||
                        dpSourcefile.includes(sourceFile) ||
                        sourceFile.includes(dpSourcefile)
                    );
                });
                if (fuzzy) return fuzzy;

                // Stage 4: court guide fallback — match by sourcefile for non-CPR sources
                const isCourtGuideCategory = (value?: string) => {
                    if (!value) return false;
                    return value.toLowerCase() !== "civil procedure rules and practice directions".toLowerCase();
                };
                const courtGuide = dataPointsArray.find(dp => {
                    const dpSourcefile = String(dp.sourcefile || "").trim();
                    const dpSourcepage = String(dp.sourcepage || "").trim();
                    const dpCategory = String(dp.category || "").trim();
                    if (!isCourtGuideCategory(dpCategory)) return false;
                    if (!dpSourcefile || dpSourcefile !== sourceFile) return false;
                    if (!sourcePage) return true;
                    return dpSourcepage.includes(sourcePage) || sourcePage.includes(dpSourcepage);
                });
                return courtGuide;
            }

            if (parsedCitation.kind === "two") {
                const { partA, partB } = parsedCitation;
                return dataPointsArray.find(dataPoint => {
                    const sourcepage = String(dataPoint.sourcepage || "").trim();
                    const sourcefile = String(dataPoint.sourcefile || "").trim();
                    return (sourcepage === partA && sourcefile === partB) || (sourcepage === partB && sourcefile === partA);
                });
            }

            return dataPointsArray.find(dataPoint => {
                const sourcepage = String(dataPoint.sourcepage || "").trim();
                const sourcefile = String(dataPoint.sourcefile || "").trim();
                return sourcepage === citation || sourcefile === citation;
            });
        },
        [answer.context?.data_points, getDataPointsArray, parseCitationLabel]
    );

    const handleCopy = () => {
        const tempElement = document.createElement("div");
        tempElement.innerHTML = sanitizedAnswerHtml;
        tempElement.querySelectorAll("sup").forEach(node => node.remove());
        tempElement.querySelectorAll(".citationStepBadge").forEach(node => node.remove());
        const textToCopy = tempElement.textContent ?? "";

        navigator.clipboard
            .writeText(textToCopy)
            .then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
            })
            .catch(err => console.error("Failed to copy text: ", err));
    };

    // CUSTOM: Event delegation for inline superscript citations.
    // renderToStaticMarkup strips React onClick handlers, so we use data-citation-path
    // attributes and delegate clicks from the parent container.
    const handleAnswerClick = useCallback(
        (e: React.MouseEvent<HTMLDivElement>) => {
            const target = e.target as HTMLElement;
            const supContainer = target.closest(".supContainer") as HTMLElement;
            if (supContainer) {
                const citationPath = supContainer.getAttribute("data-citation-path");
                const selectionId = supContainer.getAttribute("data-citation-selection-id") || undefined;
                const citationText = supContainer.getAttribute("data-citation-text") || supContainer.getAttribute("title") || "";
                const citationContent = supContainer.getAttribute("data-citation-content") || "";
                if (citationPath) {
                    e.preventDefault();
                    const matchingSupportingContent = citationText ? findMatchingSupportingContent(citationText) : undefined;
                    const finalCitationContent = citationContent || matchingSupportingContent?.content || "";
                    // CUSTOM: Extract structured metadata from data-attributes for precise SupportingContent matching
                    const metadata: StructuredCitationMetadata = {
                        subsectionId: supContainer.getAttribute("data-subsection-id") || "",
                        sourcepage: supContainer.getAttribute("data-sourcepage") || "",
                        sourcefile: supContainer.getAttribute("data-sourcefile") || "",
                        category: supContainer.getAttribute("data-category") || "",
                        content: finalCitationContent,
                        storageUrl: "",
                        fullContent: "",
                        sourceRevision: supContainer.getAttribute("data-source-revision") || "",
                        sourceId: supContainer.getAttribute("data-source-id") || "",
                        documentId: supContainer.getAttribute("data-document-id") || "",
                        canonicalTextSha256: supContainer.getAttribute("data-canonical-text-sha256") || ""
                    };
                    const hasMetadata = metadata.subsectionId || metadata.sourcepage || metadata.sourcefile || metadata.category;
                    onCitationClicked(citationPath, finalCitationContent || undefined, hasMetadata ? metadata : undefined, selectionId);
                }
            }
        },
        [findMatchingSupportingContent, onCitationClicked]
    );

    return (
        <div
            className={`${styles.answerContainer} ${isSelected ? styles.selected : ""}`}
            style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}
            data-answer-index={index}
        >
            <div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <AnswerIcon />
                    <div>
                        <Button
                            appearance="transparent"
                            style={{ color: "black" }}
                            icon={copied ? <Checkmark24Regular /> : <Copy24Regular />}
                            title={copied ? t("tooltips.copied") : t("tooltips.copy")}
                            aria-label={copied ? t("tooltips.copied") : t("tooltips.copy")}
                            onClick={handleCopy}
                        />
                        {adminMode && (
                            <Button
                                appearance="transparent"
                                style={{ color: "black" }}
                                icon={<LightbulbFilament24Regular />}
                                title={t("tooltips.showThoughtProcess")}
                                aria-label={t("tooltips.showThoughtProcess")}
                                onClick={() => onThoughtProcessClicked()}
                                disabled={!answer.context.thoughts?.length || isStreaming}
                            />
                        )}
                        <Button
                            appearance="transparent"
                            style={{ color: "black" }}
                            icon={<ClipboardTextLtr24Regular />}
                            title={t("tooltips.showSupportingContent")}
                            aria-label={t("tooltips.showSupportingContent")}
                            onClick={() => onSupportingContentClicked()}
                            disabled={!answer.context.data_points || isStreaming}
                        />
                        {showSpeechOutputAzure && (
                            <SpeechOutputAzure answer={sanitizedAnswerHtml} index={index} speechConfig={speechConfig} isStreaming={isStreaming} />
                        )}
                        {showSpeechOutputBrowser && <SpeechOutputBrowser answer={sanitizedAnswerHtml} />}
                    </div>
                </div>
            </div>

            <div style={{ flexGrow: 1 }}>
                <div className={styles.answerText} onClick={handleAnswerClick}>
                    <ReactMarkdown children={sanitizedAnswerHtml} rehypePlugins={[rehypeRaw]} remarkPlugins={[remarkGfm]} />
                </div>
            </div>

            {showCitations && !!parsedAnswer.citations.length && (
                <div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "5px" }}>
                        <span className={styles.citationLearnMore}>{t("citationWithColon")}</span>
                        {parsedAnswer.citations.map(citation => {
                            const isWeb = citation.isWeb;
                            const displayIndex = citation.index;
                            const reference = citation.reference;
                            if (isWeb) {
                                // Attempt to find the matching web data point to retrieve its title
                                const webEntry = answer.context.data_points.external_results_metadata?.find(w => w.url === reference);
                                const titleOrUrl = webEntry?.title?.trim() ? webEntry.title : reference;
                                return (
                                    <span key={`${reference}-${displayIndex}`} className={styles.citationEntry}>
                                        <a className={styles.citation} title={reference} href={reference} target="_blank" rel="noopener noreferrer">
                                            <span className={styles.citationIndexBadge}>[{displayIndex}]</span>
                                            <span className={styles.citationLabel}>{titleOrUrl}</span>
                                        </a>
                                    </span>
                                );
                            } else {
                                const path = getCitationFilePath(reference);
                                const matchingSupportingContent = findMatchingSupportingContent(reference);
                                // CUSTOM: Build metadata from citation detail for precise matching
                                const citMeta: StructuredCitationMetadata | undefined =
                                    citation.subsectionId || citation.sourcepage || citation.sourcefile || citation.category
                                        ? {
                                              subsectionId: citation.subsectionId || "",
                                              sourcepage: citation.sourcepage || "",
                                              sourcefile: citation.sourcefile || "",
                                              category: citation.category || "",
                                              content: citation.content || "",
                                              storageUrl: citation.storageUrl || "",
                                              fullContent: "",
                                              sourceRevision: citation.sourceRevision || "",
                                              sourceId: citation.sourceId || "",
                                              documentId: citation.documentId || "",
                                              canonicalTextSha256: citation.canonicalTextSha256 || ""
                                          }
                                        : undefined;
                                const citationPath = citMeta
                                    ? buildCitationPath({ sourcefile: citMeta.sourcefile, sourcepage: citMeta.sourcepage, storageurl: citMeta.storageUrl })
                                    : "";
                                const displayLabel = buildCitationLabel(citMeta, reference);
                                return (
                                    <span key={`${reference}-${displayIndex}`} className={styles.citationEntry}>
                                        <a
                                            className={styles.citation}
                                            title={reference}
                                            onClick={e => {
                                                e.preventDefault();
                                                onCitationClicked(
                                                    citationPath || path,
                                                    matchingSupportingContent?.content || undefined,
                                                    citMeta,
                                                    `${citationPath || path}::${displayIndex}`
                                                );
                                            }}
                                        >
                                            <span className={styles.citationIndexBadge}>[{displayIndex}]</span>
                                            <span className={styles.citationLabel}>{displayLabel}</span>
                                        </a>
                                    </span>
                                );
                            }
                        })}
                    </div>
                </div>
            )}

            {/* CUSTOM: Subtle AI disclaimer for legal compliance */}
            <div className={styles.aiDisclaimer}>AI-generated content may be incorrect. Always verify with the primary source documents cited above.</div>

            {!!followupQuestions?.length && showFollowupQuestions && onFollowupQuestionClicked && (
                <div>
                    <div
                        style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}
                        className={`${!!parsedAnswer.citations.length ? styles.followupQuestionsList : ""}`}
                    >
                        <span className={styles.followupQuestionLearnMore}>{t("followupQuestions")}</span>
                        {followupQuestions.map((x, i) => {
                            return (
                                <a key={i} className={styles.followupQuestion} title={x} onClick={() => onFollowupQuestionClicked(x)}>
                                    {`${x}`}
                                </a>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};

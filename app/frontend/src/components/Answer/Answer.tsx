import { useEffect, useMemo, useState, useCallback } from "react";
import { Stack, IconButton } from "@fluentui/react";
import { useTranslation } from "react-i18next";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

import styles from "./Answer.module.css";
import { ChatAppResponse, SpeechConfig } from "../../api";
import { parseAnswerToHtml } from "./AnswerParser";

// Simple AnswerIcon component (replace with your actual icon as needed)
const AnswerIcon = () => (
    <span role="img" aria-label="Answer">
        💡
    </span>
);

interface Props {
    answer: ChatAppResponse;
    index: number;
    speechConfig: SpeechConfig;
    isSelected?: boolean;
    isStreaming: boolean;
    onCitationClicked: (filePath: string, citationContent?: string) => void;
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
    const parsedAnswer = useMemo(() => parseAnswerToHtml(answer, isStreaming), [answer, isStreaming]);
    const { t } = useTranslation();

    // Process the HTML to make citations clickable
    const processedAnswerHtml = useMemo(() => {
        let html = DOMPurify.sanitize(parsedAnswer.answerHtml);

        // Add click handlers and proper styling to citation superscripts
        html = html.replace(
            /<sup class="citation-sup"([^>]*)>(\d+)<\/sup>/g,
            '<sup class="citation-sup citation-clickable"$1 style="cursor: pointer; color: #0066cc; margin-left: 2px;">$2</sup>'
        );

        return html;
    }, [parsedAnswer.answerHtml]);

    const [copied, setCopied] = useState(false);

    // Add defensive handling for data_points before any .find() calls
    const getDataPointsArray = (dataPoints: any): any[] => {
        if (!dataPoints) {
            return [];
        }

        if (Array.isArray(dataPoints)) {
            return dataPoints;
        }

        if (dataPoints.text && Array.isArray(dataPoints.text)) {
            // Convert text array to object format for backward compatibility
            return dataPoints.text.map((textItem: any, index: number) => {
                if (typeof textItem === "string" && textItem.length > 0) {
                    const urlMatch = textItem.match(/^(https?:\/\/[^:]+):\s*/);
                    if (urlMatch) {
                        const content = textItem.substring(urlMatch[0].length);
                        return {
                            id: index,
                            content,
                            storageUrl: urlMatch[1],
                            sourcepage: `Source ${index + 1}`,
                            sourcefile: content.substring(0, 50) + "..."
                        };
                    }
                    return {
                        id: index,
                        content: textItem,
                        sourcepage: `Source ${index + 1}`,
                        sourcefile: textItem.substring(0, 50) + "..."
                    };
                } else if (textItem && typeof textItem === "object") {
                    // IMPORTANT: preserve all original fields (including subsection_id, subsection_index, original_doc_id, storageurl/url)
                    return {
                        id: index,
                        ...textItem,
                        content: textItem.content || "",
                        storageUrl: textItem.storageUrl || textItem.storageurl || textItem.url || "",
                        sourcepage: textItem.sourcepage || `Source ${index + 1}`,
                        sourcefile: textItem.sourcefile || ""
                    };
                } else {
                    return {
                        id: index,
                        content: String(textItem || ""),
                        sourcepage: `Source ${index + 1}`,
                        sourcefile: ""
                    };
                }
            });
        }

        return [];
    };

    // Enhanced function to find matching content for a citation
    const findMatchingSupportingContent = useCallback(
        (citation: string) => {
            if (!answer.context?.data_points) {
                return undefined;
            }

            const dataPointsArray = getDataPointsArray(answer.context.data_points);
            const citationParts = citation.split(",").map(p => p.trim());

            console.log("Finding matching content for citation:", {
                citation,
                parts: citationParts,
                dataPointsLength: dataPointsArray.length
            });

            // Three-part citations: subsection, sourcePage, sourceFile
            if (citationParts.length >= 3) {
                const subsection = citationParts[0];
                const sourcePage = citationParts[citationParts.length - 2];
                const sourceFile = citationParts[citationParts.length - 1];

                // 0) Exact match on all three (subsection_id + sourcepage + sourcefile)
                const exactAll = dataPointsArray.find(dp => {
                    const dpSourcepage = String(dp.sourcepage || "").trim();
                    const dpSourcefile = String(dp.sourcefile || "").trim();
                    const dpSubsection = String(dp.subsection_id || "").trim();
                    return dpSubsection === subsection && dpSourcepage === sourcePage && dpSourcefile === sourceFile;
                });
                if (exactAll) {
                    console.log("Found exact match on subsection/sourcepage/sourcefile:", exactAll);
                    return exactAll;
                }

                // 1) Match subsection in content start + exact page/file
                const escaped = subsection.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                const startsWithPattern = new RegExp(`(^|\\n)\\s*${escaped}\\b`, "i");
                const contentAndMetaMatch = dataPointsArray.find(dp => {
                    const dpSourcepage = String(dp.sourcepage || "").trim();
                    const dpSourcefile = String(dp.sourcefile || "").trim();
                    const dpContent = String(dp.content || "");
                    return startsWithPattern.test(dpContent) && dpSourcepage === sourcePage && dpSourcefile === sourceFile;
                });
                if (contentAndMetaMatch) {
                    console.log("Found content+meta match:", contentAndMetaMatch);
                    return contentAndMetaMatch;
                }

                // 2) Exact match on sourcepage + sourcefile as last resort
                const exact = dataPointsArray.find(dp => {
                    const dpSourcepage = String(dp.sourcepage || "").trim();
                    const dpSourcefile = String(dp.sourcefile || "").trim();
                    return dpSourcepage === sourcePage && dpSourcefile === sourceFile;
                });
                if (exact) {
                    console.log("Found exact matching data point (page+file only):", exact);
                    return exact;
                }

                // 3) Fuzzy includes
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
                if (fuzzy) {
                    console.log("Found fuzzy matching data point:", fuzzy);
                    return fuzzy;
                }

                console.log("No matching data point found for three-part citation");
                return undefined;
            }

            // Two-part legacy citations
            if (citationParts.length === 2) {
                const [partA, partB] = citationParts;
                const twoPartExact = dataPointsArray.find(dp => {
                    const dpSourcepage = String(dp.sourcepage || "").trim();
                    const dpSourcefile = String(dp.sourcefile || "").trim();
                    return (dpSourcepage === partA && dpSourcefile === partB) || (dpSourcepage === partB && dpSourcefile === partA);
                });
                if (twoPartExact) return twoPartExact;
            }

            // Fallback
            for (const dp of dataPointsArray) {
                if (dp.sourcepage === citation || dp.sourcefile === citation) {
                    return dp;
                }
            }

            console.log("No matching content found for citation:", citation);
            return undefined;
        },
        [answer.context?.data_points]
    );

    const handleCopy = () => {
        // Single replace to remove all HTML tags to remove the citations
        const textToCopy = processedAnswerHtml.replace(/<sup [^>]*>\d+<\/sup>|<[^>]+>/g, "");

        navigator.clipboard
            .writeText(textToCopy)
            .then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
            })
            .catch(err => console.error("Failed to copy text: ", err));
    };

    // Get data points as array for supporting content button
    const dataPointsArray = getDataPointsArray(answer.context?.data_points);
    const hasDataPoints = dataPointsArray.length > 0;

    // Gate citation list visibility to avoid flicker during streaming
    const [showCitations, setShowCitations] = useState(false);
    useEffect(() => {
        if (isStreaming) {
            setShowCitations(false);
            return;
        }
        // Small delay to allow final citations to stabilize post-stream
        const timer = setTimeout(() => setShowCitations(true), 150);
        return () => clearTimeout(timer);
    }, [isStreaming]);

    // Enhanced click handler for superscript citation links
    useEffect(() => {
        const handleCitationClick = (e: Event) => {
            const target = e.target as HTMLElement;

            if (target.classList.contains("citation-sup") || target.closest(".citation-sup")) {
                e.preventDefault();
                e.stopPropagation();

                const citationElement = target.classList.contains("citation-sup") ? target : target.closest(".citation-sup");
                const citationText = citationElement?.getAttribute("data-citation-text");
                const citationContent = citationElement?.getAttribute("data-citation-content");

                if (citationText) {
                    console.log("Superscript citation clicked:", { citation: citationText });

                    const matchingSupportingContent = findMatchingSupportingContent(citationText);
                    let finalCitationContent = citationContent || matchingSupportingContent?.content || "";

                    // Build normalized label strictly from citation parts + index fields
                    let normalizedLabel = citationText;
                    const parts = citationText.split(",").map(p => p.trim());
                    if (parts.length >= 3) {
                        const subsection = parts[0]; // keep original subsection like "D5.6"
                        const sourcePage = parts[parts.length - 2];
                        const sourceFile = parts[parts.length - 1];
                        normalizedLabel = [subsection, sourcePage, sourceFile].filter(Boolean).join(", ");
                    } else if (matchingSupportingContent) {
                        const sourcePage = matchingSupportingContent.sourcepage || "";
                        const sourceFile = matchingSupportingContent.sourcefile || "";
                        normalizedLabel = [sourcePage, sourceFile].filter(Boolean).join(", ");
                    }

                    console.log("Citation click - content:", {
                        citation: normalizedLabel,
                        hasContent: !!finalCitationContent,
                        contentLength: finalCitationContent?.length || 0
                    });

                    onCitationClicked(normalizedLabel, finalCitationContent || undefined);
                }
            }
        };

        // Add event listener to the answer container
        const answerContainer = document.querySelector(`[data-answer-index="${index}"]`);
        if (answerContainer) {
            answerContainer.addEventListener("click", handleCitationClick);
            return () => {
                answerContainer.removeEventListener("click", handleCitationClick);
            };
        }
    }, [onCitationClicked, index, findMatchingSupportingContent]);

    const handleCitationLinkClick = useCallback(
        (citationFilePath: string, citationContent?: string) => {
            onCitationClicked(citationFilePath, citationContent);
        },
        [onCitationClicked]
    );

    return (
        <Stack className={`${styles.answerContainer} ${isSelected && styles.selected}`} verticalAlign="space-between" data-answer-index={index}>
            <Stack.Item>
                <Stack horizontal horizontalAlign="space-between">
                    <AnswerIcon />
                    <div>
                        <IconButton
                            style={{ color: "black" }}
                            iconProps={{ iconName: copied ? "CheckMark" : "Copy" }}
                            title={copied ? t("tooltips.copied") : t("tooltips.copy")}
                            ariaLabel={copied ? t("tooltips.copied") : t("tooltips.copy")}
                            onClick={handleCopy}
                        />
                        <IconButton
                            style={{ color: "black" }}
                            iconProps={{ iconName: "Lightbulb" }}
                            title={t("tooltips.showThoughtProcess")}
                            ariaLabel={t("tooltips.showThoughtProcess")}
                            onClick={() => onThoughtProcessClicked()}
                            disabled={!answer.context?.thoughts?.length || isStreaming}
                        />
                        <IconButton
                            style={{ color: "black" }}
                            iconProps={{ iconName: "ClipboardList" }}
                            title={t("tooltips.showSupportingContent")}
                            ariaLabel={t("tooltips.showSupportingContent")}
                            onClick={() => onSupportingContentClicked()}
                            disabled={!hasDataPoints || isStreaming}
                        />
                    </div>
                </Stack>
            </Stack.Item>

            <Stack.Item grow>
                <div className={styles.answerText}>
                    <ReactMarkdown children={processedAnswerHtml} rehypePlugins={[rehypeRaw]} remarkPlugins={[remarkGfm]} />
                </div>
            </Stack.Item>

            {!!parsedAnswer.citations.length && showCitations && (
                <Stack.Item>
                    <Stack horizontal wrap tokens={{ childrenGap: 5 }}>
                        <span className={styles.citationLearnMore}>{t("citationWithColon")}</span>
                        {parsedAnswer.citations.map((citation, i) => {
                            const matchingSupportingContent = findMatchingSupportingContent(citation);
                            const citationContent = matchingSupportingContent?.content || "";
                            const displayIndex = i + 1;

                            // Build normalized label from the citation's own parts (preserve subsection like "D5.6")
                            const parts = citation.split(",").map(p => p.trim());
                            let subsection = parts.length >= 3 ? parts[0] : "";
                            let sourcePage = parts.length >= 2 ? parts[parts.length - 2] : "";
                            let sourceFile = parts.length >= 1 ? parts[parts.length - 1] : "";

                            // If matching content exists, sync sourcepage/sourcefile from it (but keep subsection from citation)
                            if (matchingSupportingContent) {
                                sourcePage = matchingSupportingContent.sourcepage || sourcePage;
                                sourceFile = matchingSupportingContent.sourcefile || sourceFile;
                            }

                            const normalizedLabelParts = [subsection, sourcePage, sourceFile].filter(Boolean);
                            const normalizedLabel = normalizedLabelParts.join(", ");

                            // Display concise text: subsection - sourcePage
                            const displayText = subsection && sourcePage ? `${subsection} - ${sourcePage}` : normalizedLabel;

                            console.log("Rendering citation link:", {
                                citation,
                                normalizedLabel,
                                hasMatchingSupportingContent: !!matchingSupportingContent,
                                citationContentLength: citationContent.length,
                                displayText
                            });

                            return (
                                <a
                                    key={i}
                                    className={styles.citation}
                                    title={normalizedLabel}
                                    onClick={e => {
                                        e.preventDefault();
                                        console.log("Citation link clicked:", {
                                            normalizedLabel,
                                            matchingSupportingContent: !!matchingSupportingContent,
                                            citationContentLength: citationContent.length
                                        });
                                        handleCitationLinkClick(normalizedLabel, citationContent);
                                    }}
                                    style={{ cursor: "pointer" }}
                                >
                                    {`${displayIndex}. ${displayText}`}
                                </a>
                            );
                        })}
                    </Stack>
                </Stack.Item>
            )}

            {!!followupQuestions?.length && showFollowupQuestions && onFollowupQuestionClicked && (
                <Stack.Item>
                    <Stack horizontal wrap className={`${!!parsedAnswer.citations.length ? styles.followupQuestionsList : ""}`} tokens={{ childrenGap: 6 }}>
                        <span className={styles.followupQuestionLearnMore}>{t("followupQuestions")}</span>
                        {followupQuestions.map((x, i) => {
                            return (
                                <a key={i} className={styles.followupQuestion} title={x} onClick={() => onFollowupQuestionClicked(x)}>
                                    {`${x}`}
                                </a>
                            );
                        })}
                    </Stack>
                </Stack.Item>
            )}
        </Stack>
    );
};

// Helper function to extract subsection from content
function extractSubsection(content: string): string {
    if (!content) return "";

    const lines = content.split("\n");
    const firstLine = lines[0]?.trim();

    // Check if first line looks like a heading or rule number
    if (firstLine && firstLine.length < 100) {
        const cleaned = firstLine.replace(/^#+\s*/, "").trim();

        if (/^\d+\.\d+/.test(cleaned) || /^Rule \d+/.test(cleaned) || /^Part \d+/.test(cleaned)) {
            return cleaned;
        }

        return cleaned;
    }

    // Look for rule patterns in the content
    const ruleMatch = content.match(/(?:Rule\s+)?(\d+\.\d+(?:\(\d+\))?(?:\([a-z]\))?)/i);
    if (ruleMatch) {
        return ruleMatch[0];
    }

    // Look for part patterns
    const partMatch = content.match(/(Part\s+\d+[^.]*)/i);
    if (partMatch) {
        return partMatch[1].substring(0, 50);
    }

    return content.substring(0, 50).trim() + (content.length > 50 ? "..." : "");
}

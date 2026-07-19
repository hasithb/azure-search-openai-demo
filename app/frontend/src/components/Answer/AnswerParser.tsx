import { renderToStaticMarkup } from "react-dom/server";
import { ChatAppResponse, getCitationFilePath } from "../../api";
import { QueryPlanStep, getStepLabel, activityTypeLabels } from "../AnalysisPanel/agentPlanUtils";

// CUSTOM: Import citation sanitization from isolated customizations folder
import { sanitizeCitations, collapseAdjacentCitations, fixMalformedCitations, findMatchingCitation } from "../../customizations/citationSanitizer";
// CUSTOM: Paragraph formatting for long single-block answers
import { formatAnswerParagraphs, isFeatureEnabled } from "../../customizations";
// CUSTOM: Structured citation metadata for precise SupportingContent matching
import { extractMetadataFromDataPoint, buildCitationPath } from "../../customizations";

// Re-export for backward compatibility with existing tests
export { sanitizeCitations, collapseAdjacentCitations, fixMalformedCitations };

export type CitationDetail = {
    reference: string;
    index: number;
    isWeb: boolean;
    content?: string;
    activityId?: string;
    stepNumber?: number;
    stepLabel?: string;
    stepSource?: string;
    // CUSTOM: Structured metadata for precise SupportingContent matching
    subsectionId?: string;
    sourcepage?: string;
    sourcefile?: string;
    category?: string;
    storageUrl?: string;
};

type CitationFragment =
    | { type: "text"; value: string }
    | {
          type: "citation";
          detail: CitationDetail;
      };

type ActivityStepMeta = {
    stepNumber: number;
    stepLabel: string;
};

type NestedSubsectionSegment = {
    label: string;
    content: string;
};

type HtmlParsedAnswer = {
    answerHtml: string;
    citations: CitationDetail[];
};

const isWebCitation = (reference: string) => reference.startsWith("http://") || reference.startsWith("https://");

const extractSubsectionFromContent = (content: string): string => {
    if (!content) {
        return "";
    }

    const lines = content.split("\n");
    const firstLine = lines[0]?.trim();

    if (firstLine && firstLine.length < 100) {
        const cleaned = firstLine.replace(/^#+\s*/, "").trim();

        if (/^PART\s+\d+/i.test(cleaned)) {
            return cleaned;
        }

        if (/^\d+\.\d+/.test(cleaned) || /^Rule \d+/i.test(cleaned)) {
            return cleaned;
        }

        if (cleaned.length > 3 && cleaned.length < 80) {
            return cleaned;
        }
    }

    const partMatch = content.match(/PART\s+\d+[^.\n]*/i);
    if (partMatch) {
        return partMatch[0].trim();
    }

    const ruleMatch = content.match(/(?:Rule\s+)?(\d+\.\d+(?:\(\d+\))?(?:\([a-z]\))?)/i);
    if (ruleMatch) {
        return ruleMatch[0];
    }

    return content.substring(0, 50).trim() + (content.length > 50 ? "..." : "");
};

const normalizeContextTokens = (text: string): string[] => {
    return text
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, " ")
        .split(/\s+/)
        .filter(token => token.length > 2 && !["the", "and", "for", "that", "with", "must", "file", "after"].includes(token));
};

const extractNestedSubsectionSegments = (baseSubsection: string, content: string): NestedSubsectionSegment[] => {
    if (!baseSubsection || !content) {
        return [];
    }

    const escapedBase = baseSubsection.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const startsWithNestedSubsections = new RegExp(`^\\s*${escapedBase}\\s*\\((\\d+|[a-z])\\)`, "i").test(content);
    if (!startsWithNestedSubsections) {
        return [];
    }

    const markerRegex = /\((\d+|[a-z])\)\s+/gi;
    const matches = Array.from(content.matchAll(markerRegex));
    if (matches.length < 2) {
        return [];
    }

    return matches.map((match, index) => {
        const token = match[1];
        const start = (match.index ?? 0) + match[0].length;
        const end = matches[index + 1]?.index ?? content.length;
        return {
            label: `${baseSubsection}(${token})`,
            content: content.slice(start, end).trim()
        };
    });
};

const getSentenceContext = (text: string): string => {
    const paragraphs = text
        .split(/\n+/)
        .map(part => part.trim())
        .filter(Boolean);
    const lastParagraph = paragraphs[paragraphs.length - 1] ?? text.trim();
    const sentenceParts = lastParagraph.split(/(?<=[.!?])\s+/).filter(Boolean);
    return (sentenceParts[sentenceParts.length - 1] ?? lastParagraph).trim();
};

const findContextSubsection = (context: string): string | undefined => {
    const match = context.match(/\b(?:rule\s+)?(\d+\.\d+(?:\.\d+)?)(?:\b|\()/i);
    return match?.[1];
};

const resolveNumericCitationDataPoint = (textDataPoints: any[], citationIndex: number, answerContext: string): any => {
    const positionalDataPoint = textDataPoints[citationIndex];
    const contextSubsection = findContextSubsection(answerContext);
    if (!contextSubsection) {
        return positionalDataPoint;
    }

    const exactMatch = textDataPoints.find(dataPoint => String(dataPoint?.subsection_id || "").trim().toLowerCase() === contextSubsection.toLowerCase());
    return exactMatch || positionalDataPoint;
};

const resolveNestedSubsectionLabel = (baseSubsection: string, content: string, occurrenceIndex: number, answerContext: string): string | undefined => {
    const nestedSegments = extractNestedSubsectionSegments(baseSubsection, content);
    if (nestedSegments.length === 0) {
        return undefined;
    }

    const contextTokens = normalizeContextTokens(answerContext);
    if (contextTokens.length > 0) {
        let bestMatch: NestedSubsectionSegment | undefined;
        let bestScore = 0;

        for (const segment of nestedSegments) {
            const segmentTokens = new Set(normalizeContextTokens(segment.content));
            const score = contextTokens.reduce((total, token) => total + (segmentTokens.has(token) ? 1 : 0), 0);
            if (score > bestScore) {
                bestScore = score;
                bestMatch = segment;
            }
        }

        if (bestMatch && bestScore > 0) {
            return bestMatch.label;
        }
    }

    return nestedSegments[Math.min(occurrenceIndex - 1, nestedSegments.length - 1)]?.label;
};

const buildCitationFromDataPoint = (dataPoint: any, fallbackIndex: number, occurrenceIndex = 1, answerContext = ""): string | undefined => {
    if (!dataPoint || typeof dataPoint !== "object") {
        return undefined;
    }

    if (typeof dataPoint.citation === "string" && dataPoint.citation.trim()) {
        return dataPoint.citation.trim();
    }

    const dpContent = String(dataPoint.content || "");
    const baseSubsection = String(dataPoint.subsection_id || extractSubsectionFromContent(dpContent)).trim();
    const dpSubsection = resolveNestedSubsectionLabel(baseSubsection, dpContent, occurrenceIndex, answerContext) ?? baseSubsection;
    const dpSourcepage = String(dataPoint.sourcepage || "").trim();
    const dpSourcefile = String(dataPoint.sourcefile || "").trim();

    if (dpSubsection && dpSourcepage && dpSourcefile) {
        return `${dpSubsection}, ${dpSourcepage}, ${dpSourcefile}`;
    }

    if (dpSourcepage && dpSourcefile) {
        return `${dpSourcepage}, ${dpSourcefile}`;
    }

    if (dpSourcefile) {
        return dpSourcefile;
    }

    return `Source ${fallbackIndex}`;
};

const dataPointMatchesCitation = (dataPoint: any, citation: string): boolean => {
    if (!dataPoint || typeof dataPoint !== "object" || !citation) {
        return false;
    }

    const explicitCitation = typeof dataPoint.citation === "string" ? dataPoint.citation.trim() : "";
    if (explicitCitation && explicitCitation === citation) {
        return true;
    }

    const dpContent = String(dataPoint.content || "");
    const baseSubsection = String(dataPoint.subsection_id || extractSubsectionFromContent(dpContent)).trim();
    const sourcePage = String(dataPoint.sourcepage || "").trim();
    const sourceFile = String(dataPoint.sourcefile || "").trim();

    if (baseSubsection && sourcePage && sourceFile && `${baseSubsection}, ${sourcePage}, ${sourceFile}` === citation) {
        return true;
    }

    return extractNestedSubsectionSegments(baseSubsection, dpContent).some(segment => `${segment.label}, ${sourcePage}, ${sourceFile}` === citation);
};

const normalizeAnswerText = (answer: ChatAppResponse, isStreaming: boolean): string => {
    let parsedAnswer = answer.output_text.trim();

    // CUSTOM: Apply citation sanitization before parsing
    if (isFeatureEnabled("citationSanitizer")) {
        parsedAnswer = sanitizeCitations(parsedAnswer);
    }

    if (isStreaming) {
        let lastIndex = parsedAnswer.length;
        for (let i = parsedAnswer.length - 1; i >= 0; i--) {
            if (parsedAnswer[i] === "]") {
                break;
            } else if (parsedAnswer[i] === "[") {
                lastIndex = i;
                break;
            }
        }
        parsedAnswer = parsedAnswer.substring(0, lastIndex);
    }

    // CUSTOM: Add paragraph breaks to long single-block answers for readability
    if (!isStreaming && isFeatureEnabled("answerParagraphs")) {
        parsedAnswer = formatAnswerParagraphs(parsedAnswer);
    }

    return parsedAnswer;
};

const buildActivityStepMap = (answer: ChatAppResponse): Record<string, ActivityStepMeta> => {
    const mapping: Record<string, ActivityStepMeta> = {};
    const thoughts = answer.context?.thoughts;
    if (!Array.isArray(thoughts)) {
        return mapping;
    }

    const thoughtWithPlan = thoughts.find(thought => Array.isArray(thought.props?.query_plan));
    if (!thoughtWithPlan) {
        return mapping;
    }

    const planSteps = (thoughtWithPlan.props?.query_plan as QueryPlanStep[]) ?? [];
    planSteps.forEach((step, index) => {
        if (step && step.id !== undefined && step.id !== null) {
            mapping[String(step.id)] = {
                stepNumber: index + 1,
                stepLabel: getStepLabel(step)
            };
        }
    });

    return mapping;
};

const getLegacyCitationContext = (answer: ChatAppResponse): { enhancedCitations: string[]; citationMap: Record<string, string> } => {
    const legacyContext = (answer.context ?? {}) as Record<string, any>;
    return {
        enhancedCitations: Array.isArray(legacyContext.enhanced_citations) ? legacyContext.enhanced_citations : [],
        citationMap: legacyContext.citation_map && typeof legacyContext.citation_map === "object" ? legacyContext.citation_map : {}
    };
};

const collectCitations = (answer: ChatAppResponse, isStreaming: boolean): { fragments: CitationFragment[]; citations: CitationDetail[] } => {
    const legacyCitationContext = getLegacyCitationContext(answer);
    const possibleCitations = answer.context.data_points.citations || legacyCitationContext.enhancedCitations || [];
    const textDataPoints: any[] = answer.context.data_points?.text || [];
    const citationActivityDetails = answer.context.data_points.citation_activity_details ?? {};
    const activitySteps = buildActivityStepMap(answer);
    const externalResults = answer.context.data_points.external_results_metadata || [];
    const parsedAnswer = normalizeAnswerText(answer, isStreaming);

    if (!isStreaming && !/\[[^\]]+\]/.test(parsedAnswer) && legacyCitationContext.enhancedCitations.length > 0) {
        const citationList: CitationDetail[] = [];
        const seen = new Set<string>();

        for (const reference of legacyCitationContext.enhancedCitations) {
            if (!reference || seen.has(reference)) {
                continue;
            }
            seen.add(reference);
            citationList.push({
                reference,
                index: citationList.length + 1,
                isWeb: isWebCitation(reference)
            });
        }

        return {
            fragments: [{ type: "text", value: parsedAnswer }],
            citations: citationList
        };
    }

    const parts = parsedAnswer.split(/\[([^\]]+)\]/g);

    // Helper to resolve SharePoint filename to URL
    const resolveSharePointUrl = (citation: string): string => {
        // If it's already a URL, return as-is
        if (isWebCitation(citation)) {
            return citation;
        }
        // Check if this looks like a filename (has an extension)
        const hasFileExtension = /\.(pdf|docx?|xlsx?|pptx?|txt|html?|csv)$/i.test(citation);
        if (!hasFileExtension) {
            return citation;
        }

        // Look for matching SharePoint URL in external_results_metadata
        // Match by checking if the URL ends with the filename
        const matchingResult = externalResults.find(result => {
            if (!result.url) return false;
            const urlParts = result.url.split("/");
            const urlFilename = urlParts[urlParts.length - 1];
            return urlFilename === citation || decodeURIComponent(urlFilename) === citation;
        });

        return matchingResult?.url || citation;
    };

    const fragments: CitationFragment[] = [];
    const citationMap = new Map<string, CitationDetail>();
    const citationList: CitationDetail[] = [];
    const numericCitationOccurrences = new Map<number, number>();

    parts.forEach((part, index) => {
        if (index % 2 === 0) {
            fragments.push({ type: "text", value: part });
            return;
        }

        // CUSTOM: Handle numbered citations [1], [2], [3] from the prompt pipeline.
        // The backend formats sources as "[1]: content" so the LLM outputs simple numbers.
        // We resolve the number to the enhanced citation string via data_points.text[n-1].citation.
        let matchedCitation: string | undefined;
        let numericDataPoint: any;
        const numericMatch = part.match(/^\d+$/);
        if (numericMatch) {
            const citationIndex = parseInt(part) - 1;
            if (citationIndex >= 0 && citationIndex < textDataPoints.length) {
                const answerContext = getSentenceContext(parts[index - 1] ?? "");
                numericDataPoint = resolveNumericCitationDataPoint(textDataPoints, citationIndex, answerContext);
                const occurrenceIndex = (numericCitationOccurrences.get(citationIndex) ?? 0) + 1;
                numericCitationOccurrences.set(citationIndex, occurrenceIndex);
                matchedCitation = buildCitationFromDataPoint(numericDataPoint, citationIndex + 1, occurrenceIndex, answerContext);
            }

            if (!matchedCitation) {
                matchedCitation = legacyCitationContext.citationMap[part] || legacyCitationContext.citationMap[`[${part}]`];
            }

            if (!matchedCitation && citationIndex >= 0 && citationIndex < legacyCitationContext.enhancedCitations.length) {
                matchedCitation = legacyCitationContext.enhancedCitations[citationIndex];
            }
        }

        // Fallback: try matching against possibleCitations for upstream named citations
        if (!matchedCitation) {
            matchedCitation = legacyCitationContext.citationMap[part] || legacyCitationContext.citationMap[`[${part}]`];
        }
        if (!matchedCitation) {
            matchedCitation = findMatchingCitation(part, possibleCitations);
        }
        if (!matchedCitation) {
            fragments.push({ type: "text", value: `[${part}]` });
            return;
        }

        // Use part for exact endsWith matches (backward compat), matchedCitation for fuzzy/numeric
        const citationRef = matchedCitation.endsWith(part) ? part : matchedCitation;

        // Resolve SharePoint filename to URL if applicable
        const resolvedReference = resolveSharePointUrl(citationRef);

        // Check if this resolved reference already exists.
        // CUSTOM: Allow distinct citations when the same resolved reference maps to
        // different data point content (e.g., multiple chunks from the same document).
        const existing = citationMap.get(resolvedReference);
        if (existing) {
            // Determine content of the current data point (before matchingDataPoint is computed)
            const currentDpContent = numericMatch
                ? typeof numericDataPoint?.content === "string"
                    ? numericDataPoint.content
                    : undefined
                : undefined;
            const contentDiffers = currentDpContent && existing.content && currentDpContent !== existing.content;
            if (!contentDiffers) {
                fragments.push({ type: "citation", detail: existing });
                return;
            }
            // Different content from same source — fall through to create a new citation entry
        }

        // Try both keys for activity details lookup (fuzzy match may differ from LLM text)
        const backendDetail = citationActivityDetails?.[matchedCitation] ?? citationActivityDetails?.[part];
        const activityId = backendDetail?.id;
        const stepMeta = activityId ? activitySteps[String(activityId)] : undefined;

        // Get label from backend type using our mapping, or fallback to stepMeta
        const activityLabel = backendDetail?.type ? activityTypeLabels[backendDetail.type] || backendDetail.type : undefined;

        const matchingDataPoint = numericDataPoint || textDataPoints.find((dataPoint: any) => dataPointMatchesCitation(dataPoint, matchedCitation));

        // CUSTOM: Extract structured metadata from the matching data point
        const dpMetadata = extractMetadataFromDataPoint(matchingDataPoint);

        const detail: CitationDetail = {
            reference: resolvedReference,
            index: citationList.length + 1,
            isWeb: isWebCitation(resolvedReference),
            content: typeof matchingDataPoint?.content === "string" ? matchingDataPoint.content : undefined,
            activityId: activityId !== undefined ? String(activityId) : undefined,
            stepNumber: backendDetail?.number ?? stepMeta?.stepNumber,
            stepLabel: activityLabel ?? stepMeta?.stepLabel,
            stepSource: backendDetail?.source,
            // CUSTOM: Structured metadata for precise SupportingContent matching
            subsectionId: dpMetadata.subsectionId || undefined,
            sourcepage: dpMetadata.sourcepage || undefined,
            sourcefile: dpMetadata.sourcefile || undefined,
            category: dpMetadata.category || undefined,
            storageUrl: dpMetadata.storageUrl || undefined
        };

        // CUSTOM: Use a unique key when content differs for same resolved reference
        const mapKey = existing ? `${resolvedReference}#${citationList.length}` : resolvedReference;
        citationMap.set(mapKey, detail);
        citationList.push(detail);
        fragments.push({ type: "citation", detail });
    });

    return { fragments, citations: citationList };
};

// CUSTOM: The rendered HTML carries a stable selection id so same-document citations remain distinct.
const renderCitation = (detail: CitationDetail, onCitationClicked: (citationFilePath: string, content?: string) => void) => {
    const stepBadgeLabel = detail.stepSource ?? detail.stepLabel;
    const stepBadgeTitle =
        detail.stepNumber !== undefined
            ? `Linked to Step ${detail.stepNumber}${detail.stepLabel ? `: ${detail.stepLabel}` : ""}${detail.stepSource ? ` (${detail.stepSource})` : ""}`
            : stepBadgeLabel
              ? `Linked to ${stepBadgeLabel}`
              : undefined;
    const supElement = <sup title={stepBadgeTitle ?? undefined}>[{detail.index}]</sup>;
    const citationAriaLabel = `Citation ${detail.index}: ${detail.reference}`;

    if (detail.isWeb) {
        return renderToStaticMarkup(
            <span className="citationBadgeContainer">
                <a
                    className="supContainer"
                    aria-label={citationAriaLabel}
                    title={detail.reference}
                    data-citation-index={String(detail.index)}
                    data-citation-text={detail.reference}
                    data-citation-content={detail.content ?? ""}
                    href={detail.reference}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={e => e.stopPropagation()}
                >
                    {supElement}
                </a>
            </span>
        );
    }

    const path = getCitationFilePath(detail.reference);
    // CUSTOM: Use buildCitationPath for correct path resolution (storageUrl for legal docs, /content/ for PDFs)
    const structuredPath =
        isFeatureEnabled("structuredCitationMatching") && (detail.sourcefile || detail.sourcepage || detail.storageUrl)
            ? buildCitationPath({ sourcefile: detail.sourcefile, sourcepage: detail.sourcepage, storageurl: detail.storageUrl })
            : "";
    const citationPath = structuredPath || path;
    const selectionId = `${citationPath}::${detail.index}`;
    return renderToStaticMarkup(
        <span className="citationBadgeContainer">
            <a
                className="supContainer"
                aria-label={citationAriaLabel}
                title={detail.reference}
                data-citation-path={citationPath}
                data-citation-selection-id={selectionId}
                data-citation-index={String(detail.index)}
                data-citation-text={detail.reference}
                data-citation-content={detail.content ?? ""}
                data-subsection-id={detail.subsectionId ?? ""}
                data-sourcepage={detail.sourcepage ?? ""}
                data-sourcefile={detail.sourcefile ?? ""}
                data-category={detail.category ?? ""}
            >
                {supElement}
            </a>
        </span>
    );
};

export function parseAnswerToHtml(
    answer: ChatAppResponse,
    isStreaming: boolean,
    onCitationClicked: (citationFilePath: string, content?: string) => void
): HtmlParsedAnswer {
    const { fragments, citations } = collectCitations(answer, isStreaming);
    const answerHtml = fragments.map(fragment => (fragment.type === "text" ? fragment.value : renderCitation(fragment.detail, onCitationClicked))).join("");

    return {
        answerHtml,
        citations
    };
}

export function extractCitationDetails(answer: ChatAppResponse, isStreaming = false): CitationDetail[] {
    return collectCitations(answer, isStreaming).citations;
}

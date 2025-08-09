import { ChatAppResponse } from "../../api";

export interface ParsedAnswer {
    answerHtml: string;
    citations: string[];
}

export function parseAnswerToHtml(answer: ChatAppResponse, isStreaming: boolean): ParsedAnswer {
    const answerText = answer.message.content || "";
    const context = answer.context as any; // Use 'any' to bypass strict type checking for custom context fields
    const contextDataPoints = context?.data_points;

    // Get enhanced citations and citation map from context
    const enhancedCitations = context?.enhanced_citations || [];
    const citationMap = context?.citation_map || {};

    console.log("AnswerParser - Enhanced citations:", enhancedCitations);
    console.log("AnswerParser - Citation map:", citationMap);

    const citations: string[] = [];

    // Preserve whitespace in the rendered answer; do not globally trim
    let parsedAnswer = answerText;

    // Omit a citation that is still being typed during streaming
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

    // Check if we have enhanced citations but no numbered citations in content
    const hasNumberedCitations = /\[\d+\]/.test(parsedAnswer);
    console.log("Has numbered citations in content:", hasNumberedCitations);

    if (!hasNumberedCitations && enhancedCitations.length > 0) {
        // During streaming, suppress showing citations list based on enhanced_citations to avoid flicker
        if (isStreaming) {
            return {
                answerHtml: parsedAnswer,
                citations: []
            };
        }

        console.log("No numbered citations found in content, but enhanced citations available. Using enhanced citations directly.");
        enhancedCitations.forEach((citation: string) => {
            if (!citations.includes(citation)) {
                citations.push(citation);
            }
        });

        return {
            answerHtml: parsedAnswer,
            citations
        };
    }

    // Process numbered citations [1], [2], [3], etc.
    const citationMatches = parsedAnswer.match(/\[(\d+)\]/g) || [];

    citationMatches.forEach(match => {
        const citationNumber = match.match(/\d+/)?.[0];
        if (citationNumber) {
            let enhancedCitation = citationMap[citationNumber] || enhancedCitations[parseInt(citationNumber) - 1];

            // If no enhanced citation from backend, try to build from data_points
            if (!enhancedCitation && contextDataPoints) {
                const dataPoints = Array.isArray(contextDataPoints) ? contextDataPoints : contextDataPoints.text || [];
                const dataPoint = dataPoints[parseInt(citationNumber, 10) - 1];

                if (dataPoint && typeof dataPoint === "object") {
                    const dpContent = dataPoint.content || "";
                    const dpSubsection = dataPoint.subsection_id || extractSubsectionFromContent(dpContent);

                    // STRICT: sourcepage/sourcefile must come from index
                    const dpSourcepage = String(dataPoint.sourcepage || "").trim();
                    const dpSourcefile = String(dataPoint.sourcefile || "").trim();

                    if (dpSubsection && dpSourcepage && dpSourcefile) {
                        enhancedCitation = `${dpSubsection}, ${dpSourcepage}, ${dpSourcefile}`;
                    } else if (dpSourcepage && dpSourcefile) {
                        enhancedCitation = `${dpSourcepage}, ${dpSourcefile}`;
                    } else {
                        enhancedCitation = `Source ${citationNumber}`;
                    }
                } else {
                    enhancedCitation = `Source ${citationNumber}`;
                }
            }

            if (!enhancedCitation) {
                enhancedCitation = `Source ${citationNumber}`;
            }

            // NEW: Fix mixed or inconsistent citations using context.data_points
            const fixedCitation = fixInconsistentCitation(enhancedCitation, contextDataPoints);

            console.log(`Citation ${citationNumber} -> Enhanced: ${enhancedCitation} -> Fixed: ${fixedCitation}`);

            // Add to citations array if not already present
            if (!citations.includes(fixedCitation)) {
                citations.push(fixedCitation);
            }

            // Get citation content for highlighting (preview-only)
            const citationContent = getCitationContentFromContext(contextDataPoints, fixedCitation);

            // Replace [n] with clickable superscript using the fixed label
            const citationIndex = citations.indexOf(fixedCitation) + 1;
            parsedAnswer = parsedAnswer.replace(
                match,
                `<sup class="citation-sup" data-citation-text="${encodeHtml(fixedCitation)}" data-citation-content="${encodeHtml(
                    citationContent || ""
                )}">${citationIndex}</sup>`
            );
        }
    });

    console.log("AnswerParser - Final citations array:", citations);

    return {
        answerHtml: parsedAnswer,
        citations
    };
}

// Helper: classify subsection to validate compatibility
function classifySubsection(sub?: string): { kind: "alpha" | "numeric" | "rule" | "para" | "unknown"; major?: string; prefix?: string } {
    if (!sub) return { kind: "unknown" };
    const s = sub.trim();
    const mAlpha = s.match(/^([A-Z])(\d+)(?:\.(\d+))?/i); // D5.3, A1, B2.1
    if (mAlpha) return { kind: "alpha", prefix: mAlpha[1].toUpperCase(), major: mAlpha[2] };
    const mRule = s.match(/^Rule\s+(\d+)(?:\.(\d+))?/i);
    if (mRule) return { kind: "rule", major: mRule[1] };
    const mPara = s.match(/^Para(?:graph)?\s+(\d+)(?:\.(\d+))?/i);
    if (mPara) return { kind: "para", major: mPara[1] };
    const mNum = s.match(/^(\d+)(?:\.(\d+))?/);
    if (mNum) return { kind: "numeric", major: mNum[1] };
    return { kind: "unknown" };
}

// Helper: ensure array form for data_points
function asDataPointsArray(contextDataPoints: any): any[] {
    if (!contextDataPoints) return [];
    if (Array.isArray(contextDataPoints)) return contextDataPoints;
    if (contextDataPoints.text && Array.isArray(contextDataPoints.text)) return contextDataPoints.text;
    return [];
}

// Helper: try to fix mixed/misaligned citation labels using index fields
function fixInconsistentCitation(enhancedCitation: string, contextDataPoints: any): string {
    const parts = (enhancedCitation || "").split(",").map(p => p.trim());
    if (parts.length < 2) return enhancedCitation;

    // Extract subsection, sourcePage, document (if present)
    const subsection = parts.length >= 3 ? parts[0] : "";
    const sourcePage = parts.length >= 2 ? parts[parts.length - 2] : "";
    const document = parts.length >= 1 ? parts[parts.length - 1] : "";

    // If we don’t have a subsection, nothing to validate
    if (!subsection) return enhancedCitation;

    const cls = classifySubsection(subsection);
    const dps = asDataPointsArray(contextDataPoints);

    // Build quick checks
    const isPartDoc = /^Part\s+\d+/i.test(document);
    const isPDDoc = /^(PD|Practice\s*Direction)\s*\d+/i.test(document);

    // Inconsistency 1: alpha subsection (e.g., D5.3) paired with "Part N"/"PD N" document
    if (cls.kind === "alpha" && (isPartDoc || isPDDoc)) {
        // Find a data point from a Guide (or any non-Part) that contains the subsection marker
        const secTag = `${cls.prefix}.${cls.major}`; // e.g., D.5
        const secAltTag = `${cls.prefix}${cls.major}`; // e.g., D5
        const secRegex = new RegExp(`\\b${escapeHtmlRegex(secTag)}\\b`, "i");
        const secAltRegex = new RegExp(`\\b${escapeHtmlRegex(secAltTag)}\\b`, "i");

        const candidate = dps.find(dp => {
            const dpSourcefile = String(dp.sourcefile || "");
            const dpSourcepage = String(dp.sourcepage || "");
            const content = String(dp.content || "");
            // Avoid "Part"/"PD" documents; look for Guide or other non-Part docs
            const isNonPart = !/^Part\s+\d+/i.test(dpSourcefile) && !/^(PD|Practice\s*Direction)\s*\d+/i.test(dpSourcefile);
            const hasMarker = secRegex.test(dpSourcepage) || secAltRegex.test(dpSourcepage) || secRegex.test(content) || secAltRegex.test(content);
            return isNonPart && hasMarker;
        });

        if (candidate) {
            const sourcefile = String(candidate.sourcefile || "").trim();
            const sourcepageFixed = String(candidate.sourcepage || "").trim();
            // Keep the original subsection; rebind page/file from the correct dp
            return [subsection, sourcepageFixed, sourcefile].filter(Boolean).join(", ");
        }
        // If no candidate found, keep original to avoid accidental corruption
        return enhancedCitation;
    }

    // Inconsistency 2: numeric subsection (e.g., 59.4) but document not Part/PD matching the major number
    if (cls.kind === "numeric" && !isPartDoc && !isPDDoc) {
        const expectedPart = cls.major || "";
        const candidate = dps.find(dp => {
            const dpSourcefile = String(dp.sourcefile || "");
            return (
                new RegExp(`^Part\\s+${escapeHtmlRegex(expectedPart)}\\b`, "i").test(dpSourcefile) ||
                new RegExp(`^(PD|Practice\\s*Direction)\\s*${escapeHtmlRegex(expectedPart)}\\b`, "i").test(dpSourcefile)
            );
        });
        if (candidate) {
            const sourcefile = String(candidate.sourcefile || "").trim();
            const sourcepageFixed = String(candidate.sourcepage || "").trim();
            return [subsection, sourcepageFixed, sourcefile].filter(Boolean).join(", ");
        }
        return enhancedCitation;
    }

    // Inconsistency 3: Rule/Para subsection must have marker in page/content
    if (cls.kind === "rule" || cls.kind === "para") {
        const marker = cls.kind === "rule" ? "Rule" : "Para";
        const markerRegex = new RegExp(`\\b${escapeHtmlRegex(marker)}\\b`, "i");
        // If marker not present in current sourcePage, try to rebind
        if (!markerRegex.test(sourcePage)) {
            const candidate = dps.find(dp => {
                const dpSourcepage = String(dp.sourcepage || "");
                const content = String(dp.content || "");
                return markerRegex.test(dpSourcepage) || markerRegex.test(content);
            });
            if (candidate) {
                const sourcefile = String(candidate.sourcefile || "").trim();
                const sourcepageFixed = String(candidate.sourcepage || "").trim();
                return [subsection, sourcepageFixed, sourcefile].filter(Boolean).join(", ");
            }
        }
        return enhancedCitation;
    }

    return enhancedCitation;
}

// Reuse-safe escaping for building regex
function escapeHtmlRegex(s: string): string {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Helper function to extract subsection from content
function extractSubsectionFromContent(content: string): string {
    if (!content) return "";

    const lines = content.split("\n");
    const firstLine = lines[0]?.trim();

    // Check if first line looks like a heading or rule number
    if (firstLine && firstLine.length < 100) {
        const cleaned = firstLine.replace(/^#+\s*/, "").trim();

        // Look for specific patterns first
        if (/^PART\s+\d+/i.test(cleaned)) {
            return cleaned;
        }

        if (/^\d+\.\d+/.test(cleaned) || /^Rule \d+/.test(cleaned)) {
            return cleaned;
        }

        // Return first meaningful line if it's reasonably short
        if (cleaned.length > 3 && cleaned.length < 80) {
            return cleaned;
        }
    }

    // Look for PART patterns anywhere in content
    const partMatch = content.match(/PART\s+\d+[^.\n]*/i);
    if (partMatch) {
        return partMatch[0].trim();
    }

    // Look for rule patterns in the content
    const ruleMatch = content.match(/(?:Rule\s+)?(\d+\.\d+(?:\(\d+\))?(?:\([a-z]\))?)/i);
    if (ruleMatch) {
        return ruleMatch[0];
    }

    // Fallback to content preview
    return content.substring(0, 50).trim() + (content.length > 50 ? "..." : "");
}

// Helper function to get citation content from context (preview-only; does not mutate canonical content)
function getCitationContentFromContext(contextDataPoints: any, citation: string): string | undefined {
    if (!contextDataPoints) {
        return undefined;
    }

    // Handle new DataPoints structure
    let dataPointsArray: any[] = [];

    if (Array.isArray(contextDataPoints)) {
        dataPointsArray = contextDataPoints;
    } else if (contextDataPoints.text && Array.isArray(contextDataPoints.text)) {
        dataPointsArray = contextDataPoints.text;
    } else {
        return undefined;
    }

    console.log("Getting citation content for:", citation);
    console.log("Available data points:", dataPointsArray.length);

    // Handle simple "Source N" format
    const simpleSourceMatch = citation.match(/^Source (\d+)$/);
    if (simpleSourceMatch) {
        const sourceIndex = parseInt(simpleSourceMatch[1], 10) - 1;
        if (sourceIndex >= 0 && sourceIndex < dataPointsArray.length) {
            const dataPoint = dataPointsArray[sourceIndex];
            if (dataPoint && typeof dataPoint === "object" && dataPoint.content) {
                return getCitationContentPreview(dataPoint.content, 150);
            }
        }
    }

    // Parse the citation format - handle both two-part and three-part citations
    const citationParts = citation.split(",").map((p: string) => p.trim());

    if (citationParts.length < 2) {
        console.log("Citation has less than 2 parts, trying direct match");
        // Try direct matching with sourcepage or sourcefile
        for (let i = 0; i < dataPointsArray.length; i++) {
            const dataPoint = dataPointsArray[i];
            if (typeof dataPoint === "object" && dataPoint !== null) {
                const dpSourcepage = dataPoint.sourcepage?.trim() || "";
                const dpSourcefile = dataPoint.sourcefile?.trim() || "";

                // Check if citation matches sourcepage or sourcefile directly
                if (dpSourcepage === citation || dpSourcefile === citation) {
                    console.log("Found direct match with sourcepage/sourcefile");
                    return getCitationContentPreview(dataPoint.content || "", 150);
                }
            }
        }
        return undefined;
    }

    // For multi-part citations, extract the last two parts as source page and document
    const sourcePage = citationParts[citationParts.length - 2];
    const document = citationParts[citationParts.length - 1];

    console.log("Looking for sourcePage:", sourcePage, "document:", document);

    // Match based on sourcepage and sourcefile from search results
    for (let i = 0; i < dataPointsArray.length; i++) {
        const dataPoint = dataPointsArray[i];

        if (typeof dataPoint === "object" && dataPoint !== null) {
            const dpSourcepage = dataPoint.sourcepage?.trim() || "";
            const dpSourcefile = dataPoint.sourcefile?.trim() || "";

            console.log(`Checking data point ${i}: sourcepage='${dpSourcepage}', sourcefile='${dpSourcefile}'`);

            // Check if this data point matches the citation
            if (dpSourcepage === sourcePage && dpSourcefile === document) {
                console.log("Found matching data point");
                return getCitationContentPreview(dataPoint.content || "", 150);
            }

            // Also try partial matching for cases where citation uses simplified names
            if (dpSourcepage.includes(sourcePage) || sourcePage.includes(dpSourcepage)) {
                if (dpSourcefile === document || dpSourcefile.includes(document) || document.includes(dpSourcefile)) {
                    console.log("Found partial matching data point");
                    return getCitationContentPreview(dataPoint.content || "", 150);
                }
            }
        }
    }

    console.log("No matching data point found for citation");
    return undefined;
}

// Helper function to create preview content (preview-only normalization)
function getCitationContentPreview(content: string, maxLength: number = 150): string {
    if (!content) {
        return "";
    }

    // Clean the content - remove extra whitespace and normalize (for preview only)
    const cleaned = content.replace(/\s+/g, " ").trim();

    // If content is short enough, return it as-is
    if (cleaned.length <= maxLength) {
        return cleaned;
    }

    // Find a good breaking point (end of sentence)
    const truncated = cleaned.substring(0, maxLength);
    const lastPeriod = truncated.lastIndexOf(". ");

    if (lastPeriod > maxLength * 0.6) {
        // Only if we're not cutting too much
        return truncated.substring(0, lastPeriod + 1);
    }

    // Try to break at word boundary
    const lastSpace = truncated.lastIndexOf(" ");
    if (lastSpace > 0) {
        return truncated.substring(0, lastSpace) + "...";
    }

    return truncated + "...";
}

// Helper function to encode HTML entities
function encodeHtml(str: string): string {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

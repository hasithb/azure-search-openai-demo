import { Stack, Pivot, PivotItem } from "@fluentui/react";
import { useTranslation } from "react-i18next";
import styles from "./AnalysisPanel.module.css";

import { SupportingContent } from "../SupportingContent";
import { ChatAppResponse } from "../../api";
import { AnalysisPanelTabs } from "./AnalysisPanelTabs";
import { ThoughtProcess } from "./ThoughtProcess";
import { MarkdownViewer } from "../MarkdownViewer";
import { useMsal } from "@azure/msal-react";
import { getHeaders } from "../../api";
import { useLogin, getToken } from "../../authConfig";
import { useState, useEffect } from "react";

interface Props {
    className: string;
    activeTab: AnalysisPanelTabs;
    onActiveTabChanged: (tab: AnalysisPanelTabs) => void;
    activeCitation: string | undefined;
    citationHeight: string;
    answer: ChatAppResponse;
    activeCitationLabel?: string | undefined;
    activeCitationContent?: string | undefined;
    enableCitationTab?: boolean; // Add new prop
}

const pivotItemDisabledStyle = { disabled: true, style: { color: "grey" } };

export const AnalysisPanel = ({
    answer,
    activeTab,
    activeCitation,
    citationHeight,
    className,
    onActiveTabChanged,
    activeCitationLabel,
    activeCitationContent,
    enableCitationTab = false // Default to false
}: Props) => {
    // Add defensive handling for data_points structure first
    const getDataPointsArray = (dataPoints: any): any[] => {
        if (!dataPoints) {
            return [];
        }

        if (Array.isArray(dataPoints)) {
            return dataPoints;
        }

        if (dataPoints.text && Array.isArray(dataPoints.text)) {
            return dataPoints.text.map((textItem: any, index: number) => {
                // Add type checking before string operations
                if (typeof textItem === "string" && textItem.length > 0) {
                    const urlMatch = textItem.match(/^(https?:\/\/[^:]+):\s*/);
                    if (urlMatch) {
                        const content = textItem.substring(urlMatch[0].length);
                        return {
                            id: index,
                            content: content,
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
                    // If it's already an object, return it as-is with defaults
                    return {
                        id: index,
                        content: textItem.content || "",
                        storageUrl: textItem.storageUrl || "",
                        sourcepage: textItem.sourcepage || `Source ${index + 1}`,
                        sourcefile: textItem.sourcefile || ""
                    };
                } else {
                    // Fallback for any other type
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

    const dataPointsArray = getDataPointsArray(answer.context.data_points);

    // Helper to extract supporting content from context data_points or thoughts "Results" step
    const getSupportingContent = (answer: ChatAppResponse): any[] => {
        // 1. Prefer structured objects from context.data_points if available
        const dataPoints = answer.context?.data_points;

        console.log("Getting supporting content from:", {
            hasDataPoints: !!dataPoints,
            isArray: Array.isArray(dataPoints),
            type: typeof dataPoints,
            keys: dataPoints ? Object.keys(dataPoints) : []
        });

        if (Array.isArray(dataPoints) && dataPoints.length > 0) {
            console.log("Using data_points array with", dataPoints.length, "items");
            return dataPoints;
        }

        // 2. Handle data_points.text structure
        if (dataPoints && !Array.isArray(dataPoints) && typeof dataPoints === "object" && "text" in dataPoints && Array.isArray((dataPoints as any).text)) {
            console.log("Using data_points.text array with", (dataPoints as any).text.length, "items");
            return (dataPoints as any).text;
        }

        // 3. Fallback to thoughts "Results" step if available
        const thoughts = answer.context?.thoughts;
        if (Array.isArray(thoughts)) {
            const resultsStep = thoughts.find((t: any) => t.title === "Results" && Array.isArray(t.description));
            if (resultsStep && Array.isArray(resultsStep.description)) {
                console.log("Using thoughts Results step with", resultsStep.description.length, "items");
                return resultsStep.description;
            }
        }

        console.log("No supporting content found");
        return [];
    };

    const supportingContentItems = getSupportingContent(answer);

    const isDisabledThoughtProcessTab: boolean = !answer.context.thoughts;
    const isDisabledSupportingContentTab: boolean = supportingContentItems.length === 0;
    const isDisabledCitationTab: boolean = !activeCitation || !enableCitationTab; // Modified condition
    const [citation, setCitation] = useState("");

    const client = useLogin ? useMsal().instance : undefined;
    const { t } = useTranslation();

    const fetchCitation = async () => {
        const token = client ? await getToken(client) : undefined;
        if (activeCitation) {
            try {
                // Get hash from the URL as it may contain #page=N
                const originalHash = activeCitation.indexOf("#") > -1 ? activeCitation.split("#")[1] : "";

                // Check if this is an external URL that should go through the browser directly
                if (activeCitation.startsWith("http://") || activeCitation.startsWith("https://")) {
                    setCitation(activeCitation);
                    return;
                }

                // For internal citations in development, skip the backend fetch and just use the citation directly
                // This prevents 404 errors when the backend content endpoint doesn't exist
                if (import.meta.env.DEV) {
                    setCitation(activeCitation);
                    return;
                }

                // For production, use the backend content endpoint
                const decodedCitation = decodeURIComponent(activeCitation);
                let contentUrl = `/content/${encodeURIComponent(decodedCitation)}`;

                const response = await fetch(contentUrl, {
                    method: "GET",
                    headers: await getHeaders(token)
                });

                if (response.ok) {
                    const citationContent = await response.blob();
                    let citationObjectUrl = URL.createObjectURL(citationContent);
                    // Add hash back to the new blob URL
                    if (originalHash) {
                        citationObjectUrl += "#" + originalHash;
                    }
                    setCitation(citationObjectUrl);
                } else {
                    console.warn("Backend content endpoint not available, using direct citation");
                    setCitation(activeCitation);
                }
            } catch (error) {
                console.warn("Error fetching citation, using direct citation:", error);
                setCitation(activeCitation);
            }
        }
    };

    // Extract citation reference from the activeCitation
    const getActiveCitationReference = (): string | undefined => {
        // Always use the citation label if provided - this is the clean citation text
        if (activeCitationLabel) {
            return activeCitationLabel;
        }

        if (!activeCitation) return undefined;

        try {
            // Parse three-part citation format
            const citationParts = activeCitation.split(",").map(p => p.trim());
            if (citationParts.length >= 3) {
                // Return the full three-part citation for matching
                return activeCitation;
            }

            // If it's a URL, extract filename and decode it
            if (activeCitation.startsWith("http://") || activeCitation.startsWith("https://") || activeCitation.includes("/")) {
                const parts = activeCitation.split("/");
                const filename = parts[parts.length - 1];
                const decoded = decodeURIComponent(filename.split(".")[0].split("?")[0].split("#")[0]);
                return decoded;
            }

            // Otherwise assume it's already a citation label
            return activeCitation;
        } catch (error) {
            console.error("Error parsing citation reference:", error);
            return activeCitation; // Fallback to original citation
        }
    };

    useEffect(() => {
        fetchCitation();
    }, [activeCitation]);

    const renderFileViewer = () => {
        if (!activeCitation) {
            return null;
        }

        try {
            const fileExtension = activeCitation.split(".").pop()?.toLowerCase();

            // For external URLs, use iframe directly
            if (activeCitation.startsWith("http://") || activeCitation.startsWith("https://")) {
                return <iframe title="Citation" src={activeCitation} width="100%" height={citationHeight} />;
            }

            // For internal content, use the citation blob URL
            switch (fileExtension) {
                case "png":
                    return <img src={citation} className={styles.citationImg} alt="Citation Image" />;
                case "md":
                    return <MarkdownViewer src={activeCitation} />;
                default:
                    return <iframe title="Citation" src={citation} width="100%" height={citationHeight} />;
            }
        } catch (error) {
            console.error("Error rendering file viewer:", error);
            return <div>Error loading citation content</div>;
        }
    };

    // Handle viewing source document from supporting content
    const handleViewSourceDocument = (citation: string) => {
        try {
            // Set the citation URL for viewing
            if (citation.startsWith("http://") || citation.startsWith("https://")) {
                setCitation(citation);
            } else {
                // For non-URL citations, try to find the corresponding URL from dataPoints
                const dataPoint = dataPointsArray.find(
                    dp => dp.sourcepage === citation || dp.sourcefile === citation || `${dp.sourcepage}, ${dp.sourcefile}` === citation
                );

                if (dataPoint?.storageUrl) {
                    setCitation(dataPoint.storageUrl);
                } else {
                    setCitation(citation);
                }
            }
            onActiveTabChanged(AnalysisPanelTabs.CitationTab);
        } catch (error) {
            console.error("Error handling view source document:", error);
            // Fallback to just setting the citation
            setCitation(citation);
            onActiveTabChanged(AnalysisPanelTabs.CitationTab);
        }
    };

    return (
        <Pivot
            className={className}
            selectedKey={activeTab}
            onLinkClick={pivotItem => pivotItem && onActiveTabChanged(pivotItem.props.itemKey! as AnalysisPanelTabs)}
        >
            <PivotItem
                itemKey={AnalysisPanelTabs.ThoughtProcessTab}
                headerText={t("headerTexts.thoughtProcess")}
                headerButtonProps={isDisabledThoughtProcessTab ? pivotItemDisabledStyle : undefined}
            >
                <ThoughtProcess thoughts={answer.context.thoughts || []} />
            </PivotItem>
            <PivotItem
                itemKey={AnalysisPanelTabs.SupportingContentTab}
                headerText={t("headerTexts.supportingContent")}
                headerButtonProps={isDisabledSupportingContentTab ? pivotItemDisabledStyle : undefined}
            >
                <SupportingContent
                    supportingContent={supportingContentItems}
                    activeCitationReference={getActiveCitationReference()}
                    activeCitationContent={activeCitationContent}
                    onViewSourceDocument={handleViewSourceDocument}
                />
            </PivotItem>
            <PivotItem
                itemKey={AnalysisPanelTabs.CitationTab}
                headerText={t("headerTexts.citation")}
                headerButtonProps={isDisabledCitationTab ? pivotItemDisabledStyle : undefined}
            >
                {renderFileViewer()}
            </PivotItem>
        </Pivot>
    );
};

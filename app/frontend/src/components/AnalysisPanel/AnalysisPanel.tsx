import { Tab, TabList, SelectTabData, SelectTabEvent } from "@fluentui/react-components";
import { useTranslation } from "react-i18next";

import { ChatAppResponse } from "../../api";
import { isAdminMode } from "../../customizations/config";
import { isIframeBlocked } from "../../customizations/externalSourceHandler";
// CUSTOM: Import structured citation metadata type
import type { StructuredCitationMetadata } from "../../customizations";
import { MarkdownViewer } from "../MarkdownViewer";
import { SupportingContent } from "../SupportingContent";
import styles from "./AnalysisPanel.module.css";
import { AnalysisPanelTabs } from "./AnalysisPanelTabs";
import { ThoughtProcess } from "./ThoughtProcess";

interface Props {
    className: string;
    activeTab: AnalysisPanelTabs;
    onActiveTabChanged: (tab: AnalysisPanelTabs) => void;
    activeCitation: string | undefined;
    citationHeight: string;
    answer: ChatAppResponse;
    onCitationClicked?: (citationFilePath: string) => void;
    onViewSourceDocument?: (citationFilePath: string) => void;
    enableCitationTab?: boolean;
    // CUSTOM: Props for subsection highlighting in Supporting Content
    activeCitationLabel?: string | undefined;
    activeCitationContent?: string | undefined;
    // CUSTOM: Structured metadata for precise SupportingContent matching
    activeCitationMetadata?: StructuredCitationMetadata;
}

const adminMode = isAdminMode();

export const AnalysisPanel = ({
    answer,
    activeTab,
    activeCitation,
    citationHeight,
    className,
    onActiveTabChanged,
    onCitationClicked,
    onViewSourceDocument,
    enableCitationTab = false,
    activeCitationLabel,
    activeCitationContent,
    activeCitationMetadata
}: Props) => {
    const isDisabledThoughtProcessTab: boolean = !answer.context.thoughts;
    const dataPoints = answer.context.data_points;
    const hasSupportingContent = Boolean(
        dataPoints &&
            ((dataPoints.text && dataPoints.text.length > 0) ||
                (dataPoints.images && dataPoints.images.length > 0) ||
                (dataPoints.external_results_metadata && dataPoints.external_results_metadata.length > 0))
    );
    const isDisabledSupportingContentTab: boolean = !hasSupportingContent;
    const isBlockedCitation = Boolean(
        activeCitation && (activeCitation.startsWith("http://") || activeCitation.startsWith("https://")) && isIframeBlocked(activeCitation)
    );
    const isDisabledCitationTab: boolean = !activeCitation || isBlockedCitation;
    const showThoughtProcessTab = adminMode;
    const showCitationTab = adminMode && enableCitationTab && !isDisabledCitationTab;
    const effectiveActiveTab =
        activeTab === AnalysisPanelTabs.ThoughtProcessTab && !showThoughtProcessTab
            ? AnalysisPanelTabs.SupportingContentTab
            : activeTab === AnalysisPanelTabs.CitationTab && !showCitationTab
              ? AnalysisPanelTabs.SupportingContentTab
              : activeTab;

    const { t } = useTranslation();

    const renderCitationViewer = () => {
        if (!activeCitation) {
            return null;
        }

        if (isBlockedCitation) {
            return (
                <div style={{ padding: "1rem" }}>
                    <p>{t("headerTexts.citation")}</p>
                    <a href={activeCitation} target="_blank" rel="noopener noreferrer">
                        {activeCitation}
                    </a>
                </div>
            );
        }

        const citationWithoutQuery = activeCitation.split("?")[0].split("#")[0];
        const fileExtension = citationWithoutQuery.split(".").pop()?.toLowerCase();

        if (fileExtension === "md") {
            return <MarkdownViewer src={activeCitation} />;
        }

        if (["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(fileExtension || "")) {
            return <img src={activeCitation} className={styles.citationImg} alt={t("headerTexts.citation")} />;
        }

        return <iframe title={t("headerTexts.citation")} src={activeCitation} width="100%" height={citationHeight} style={{ border: "none" }} />;
    };

    return (
        <div className={className}>
            <TabList
                selectedValue={effectiveActiveTab}
                onTabSelect={(_ev: SelectTabEvent, data: SelectTabData) => onActiveTabChanged(data.value as AnalysisPanelTabs)}
            >
                {showThoughtProcessTab && (
                    <Tab value={AnalysisPanelTabs.ThoughtProcessTab} disabled={isDisabledThoughtProcessTab}>
                        {t("headerTexts.thoughtProcess")}
                    </Tab>
                )}
                <Tab value={AnalysisPanelTabs.SupportingContentTab} disabled={isDisabledSupportingContentTab}>
                    {t("headerTexts.supportingContent")}
                </Tab>
                {showCitationTab && (
                    <Tab value={AnalysisPanelTabs.CitationTab} disabled={isDisabledCitationTab}>
                        {t("headerTexts.citation")}
                    </Tab>
                )}
            </TabList>
            <div>
                {effectiveActiveTab === AnalysisPanelTabs.ThoughtProcessTab && showThoughtProcessTab && (
                    <ThoughtProcess thoughts={answer.context.thoughts || []} onCitationClicked={onCitationClicked} />
                )}
                {effectiveActiveTab === AnalysisPanelTabs.SupportingContentTab && (
                    <SupportingContent
                        supportingContent={answer.context.data_points}
                        activeCitationReference={activeCitationLabel}
                        activeCitationContent={activeCitationContent}
                        activeCitationMetadata={activeCitationMetadata}
                        onViewSourceDocument={onViewSourceDocument}
                    />
                )}
                {effectiveActiveTab === AnalysisPanelTabs.CitationTab && showCitationTab && renderCitationViewer()}
            </div>
        </div>
    );
};

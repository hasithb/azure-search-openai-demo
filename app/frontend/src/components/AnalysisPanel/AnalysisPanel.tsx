import { Tab, TabList, SelectTabData, SelectTabEvent } from "@fluentui/react-components";
import { Suspense, lazy, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { ChatAppResponse } from "../../api";
import { isAdminMode, isFeatureEnabled } from "../../customizations/config";
import { isIframeBlocked } from "../../customizations/externalSourceHandler";
// CUSTOM: Import structured citation metadata type
import type { StructuredCitationMetadata } from "../../customizations";
// CUSTOM: Primary source viewer (live PDF/HTML with the cited section highlighted).
// Lazy-loaded so PDF.js only ships to the browser when the Primary Source tab opens.
const PrimarySourceViewer = lazy(() => import("../../customizations/PrimarySourceViewer").then(m => ({ default: m.PrimarySourceViewer })));
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
    // CUSTOM: Primary-source verification of the active citation (text located in the live document)
    activeCitationVerified?: "exact" | "approximate" | "none";
    onCitationVerified?: (result: "exact" | "approximate" | "none") => void;
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
    activeCitationMetadata,
    activeCitationVerified,
    onCitationVerified
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
    // CUSTOM: The Primary Source tab is the lawyer-facing validation feature. Supporting Content can
    // hold several different sources, so the tab must NOT appear on its own — it only becomes available
    // once the user explicitly clicks "Show in primary source" on a specific citation. Selecting a
    // different citation hides the tab again until the user requests it for that source.
    const [primarySourceRequested, setPrimarySourceRequested] = useState(false);
    useEffect(() => {
        setPrimarySourceRequested(false);
    }, [activeCitation]);
    const primarySourceFeatureAvailable = isFeatureEnabled("primarySourceTab") && Boolean(activeCitation);
    const showPrimarySourceTab = primarySourceFeatureAvailable && primarySourceRequested;
    const openPrimarySource = () => {
        setPrimarySourceRequested(true);
        onActiveTabChanged(AnalysisPanelTabs.PrimarySourceTab);
    };
    const effectiveActiveTab =
        activeTab === AnalysisPanelTabs.ThoughtProcessTab && !showThoughtProcessTab
            ? AnalysisPanelTabs.SupportingContentTab
            : activeTab === AnalysisPanelTabs.CitationTab && !showCitationTab
              ? AnalysisPanelTabs.SupportingContentTab
              : activeTab === AnalysisPanelTabs.PrimarySourceTab && !showPrimarySourceTab
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
                {showPrimarySourceTab && <Tab value={AnalysisPanelTabs.PrimarySourceTab}>{t("headerTexts.primarySource")}</Tab>}
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
                        onShowInPrimarySource={primarySourceFeatureAvailable ? openPrimarySource : undefined}
                        verifiedStatus={activeCitationVerified}
                    />
                )}
                {effectiveActiveTab === AnalysisPanelTabs.PrimarySourceTab && showPrimarySourceTab && (
                    <Suspense fallback={<div style={{ padding: "1rem" }}>{t("headerTexts.primarySource")}…</div>}>
                        <PrimarySourceViewer
                            citationFilePath={activeCitation}
                            metadata={activeCitationMetadata}
                            citationLabel={activeCitationLabel}
                            height={citationHeight}
                            onVerified={onCitationVerified}
                        />
                    </Suspense>
                )}
                {effectiveActiveTab === AnalysisPanelTabs.CitationTab && showCitationTab && renderCitationViewer()}
            </div>
        </div>
    );
};

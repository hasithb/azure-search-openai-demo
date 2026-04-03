/**
 * ChatInputControls - Custom Dropdowns for Chat Input
 * =====================================================
 * FluentUI v9 rebuild of the category filter and search depth dropdowns.
 * Desktop: Two side-by-side dropdowns (Source filter + Depth)
 * Mobile: Single icon button toggling a dropdown panel
 *
 * CUSTOM: Isolated in /customizations/ for merge-safe architecture.
 */

import React, { useMemo } from "react";
import { Dropdown, Option, Button, Tooltip } from "@fluentui/react-components";
import { Settings20Regular } from "@fluentui/react-icons";
import { useTranslation } from "react-i18next";
import type { Category } from "./useCategories";
import { getAbbreviatedCategory } from "./useMobile";

// CUSTOM: Map category display names to add "Guide" suffix for courts
const DISPLAY_NAME_MAP: Record<string, string> = {
    "Commercial Court": "Commercial Court Guide",
    "Circuit Commercial Court": "Circuit Commercial Court Guide",
    "Technology and Construction Court": "Technology and Construction Court Guide",
    "King's Bench Division": "King's Bench Division Guide",
    "Chancery Division": "Chancery Guide",
    "Patents Court": "Patents Court Guide"
};

function enhanceDisplayName(text: string): string {
    return DISPLAY_NAME_MAP[text] || text;
}

function buildSourceOptionId(key: string): string {
    if (!key) return "source-filter-option-all-sources";

    return `source-filter-option-${key
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")}`;
}

export interface ChatInputControlsProps {
    categories: Category[];
    categoriesLoading: boolean;
    includeCategory: string;
    setIncludeCategory: (value: string) => void;
    allCategoriesSelected: boolean;
    setAllCategoriesSelected: (value: boolean) => void;
    agenticReasoningEffort: string;
    setAgenticReasoningEffort: (value: string) => void;
    showCategoryFilter: boolean;
    showAgenticRetrievalOption: boolean;
    useAgenticRetrieval: boolean;
    isLoading: boolean;
    isMobile: boolean;
    showMobileDropdown: boolean;
    setShowMobileDropdown: (value: boolean) => void;
}

/**
 * Desktop dropdowns rendered inline to the left of the send button.
 * Returns the JSX to pass as `leftOfSend` prop to QuestionInput.
 */
export const ChatInputControls: React.FC<ChatInputControlsProps> = ({
    categories,
    categoriesLoading,
    includeCategory,
    setIncludeCategory,
    allCategoriesSelected,
    setAllCategoriesSelected,
    agenticReasoningEffort,
    setAgenticReasoningEffort,
    showCategoryFilter,
    showAgenticRetrievalOption,
    useAgenticRetrieval,
    isLoading,
    isMobile,
    showMobileDropdown,
    setShowMobileDropdown
}) => {
    const { t } = useTranslation();

    // Parse includeCategory CSV into array of selected keys
    const includeKeys = useMemo(
        () =>
            includeCategory
                ? includeCategory
                      .split(",")
                      .map(s => s.trim())
                      .filter(Boolean)
                : [],
        [includeCategory]
    );

    // Build enhanced category options
    const categoryOptions = useMemo(
        () => [
            { key: "", text: "All Sources" },
            ...categories
                .filter(c => typeof c?.key === "string" && typeof c?.text === "string" && c.key !== "")
                .map(c => ({
                    key: c.key,
                    text: enhanceDisplayName(c.text)
                }))
        ],
        [categories]
    );

    // Selected options for v9 Dropdown (multiselect)
    const selectedOptions = useMemo(() => {
        if (allCategoriesSelected) return [""];
        return includeKeys;
    }, [allCategoriesSelected, includeKeys]);

    // Display value for the dropdown
    const categoryDisplayValue = useMemo(() => {
        if (allCategoriesSelected) return isMobile ? "All" : "All Sources";
        if (includeKeys.length === 0) return "Source";
        if (includeKeys.length === 1) {
            const match = categoryOptions.find(o => o.key === includeKeys[0]);
            const text = match?.text || includeKeys[0];
            return isMobile ? getAbbreviatedCategory(text) : text;
        }
        return `${includeKeys.length} selected`;
    }, [allCategoriesSelected, includeKeys, categoryOptions, isMobile]);

    const handleCategorySelect = (_ev: unknown, data: { optionValue?: string; selectedOptions: string[] }) => {
        const key = data.optionValue ?? "";

        // Selecting "All Sources" toggles it and clears specific selections
        if (key === "") {
            const wasSelected = allCategoriesSelected;
            setAllCategoriesSelected(!wasSelected);
            setIncludeCategory("");
            return;
        }

        // Selecting a specific category clears "All"
        setAllCategoriesSelected(false);

        const newSelected = data.selectedOptions.filter(k => k !== "");
        setIncludeCategory(newSelected.join(","));
    };

    // Depth dropdown display value
    const depthDisplayValue = useMemo(() => {
        const labels: Record<string, string> = {
            minimal: t("labels.agenticReasoningEffortOptions.minimal"),
            low: t("labels.agenticReasoningEffortOptions.low"),
            medium: t("labels.agenticReasoningEffortOptions.medium")
        };
        return labels[agenticReasoningEffort] || "Depth";
    }, [agenticReasoningEffort, t]);

    const handleDepthSelect = (_ev: unknown, data: { optionValue?: string }) => {
        setAgenticReasoningEffort(data.optionValue || "low");
    };

    const showAny = showCategoryFilter || (showAgenticRetrievalOption && useAgenticRetrieval);
    if (!showAny) return null;

    // ── Mobile: Single icon button ──
    if (isMobile) {
        return (
            <Tooltip content="Search settings" relationship="label">
                <Button
                    data-testid="chat-input-mobile-settings"
                    appearance="subtle"
                    icon={<Settings20Regular />}
                    onClick={() => setShowMobileDropdown(!showMobileDropdown)}
                    style={{
                        minWidth: "32px",
                        height: "32px",
                        color: showMobileDropdown ? "#0066cc" : "#666"
                    }}
                />
            </Tooltip>
        );
    }

    // ── Desktop: Side-by-side dropdowns ──
    return (
        <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
            {showCategoryFilter && (
                <Dropdown
                    data-testid="chat-source-filter-desktop"
                    button={{
                        id: "chat-source-filter-desktop-button",
                        "aria-label": "Source filter"
                    }}
                    multiselect
                    aria-label="Source filter"
                    value={categoryDisplayValue}
                    selectedOptions={selectedOptions}
                    onOptionSelect={handleCategorySelect}
                    disabled={isLoading || categoriesLoading}
                    placeholder="Source"
                    style={{ minWidth: 140, maxWidth: 180 }}
                    size="small"
                >
                    {categoryOptions.map(opt => (
                        <Option id={buildSourceOptionId(opt.key)} key={opt.key} value={opt.key} text={opt.text}>
                            {opt.text}
                        </Option>
                    ))}
                </Dropdown>
            )}
            {showAgenticRetrievalOption && useAgenticRetrieval && (
                <Dropdown
                    value={depthDisplayValue}
                    selectedOptions={[agenticReasoningEffort]}
                    onOptionSelect={handleDepthSelect}
                    disabled={isLoading}
                    placeholder="Depth"
                    style={{ minWidth: 90, maxWidth: 110 }}
                    size="small"
                >
                    <Option value="minimal" text={t("labels.agenticReasoningEffortOptions.minimal")}>
                        <div style={{ display: "flex", flexDirection: "column", padding: "4px 0" }}>
                            <span style={{ fontSize: "13px", fontWeight: 500 }}>{t("labels.agenticReasoningEffortOptions.minimal")}</span>
                            <span style={{ fontSize: "11px", color: "#666", marginTop: "2px" }}>Fast single search</span>
                        </div>
                    </Option>
                    <Option value="low" text={t("labels.agenticReasoningEffortOptions.low")}>
                        <div style={{ display: "flex", flexDirection: "column", padding: "4px 0" }}>
                            <span style={{ fontSize: "13px", fontWeight: 500 }}>{t("labels.agenticReasoningEffortOptions.low")}</span>
                            <span style={{ fontSize: "11px", color: "#666", marginTop: "2px" }}>Balanced search depth</span>
                        </div>
                    </Option>
                    <Option value="medium" text={t("labels.agenticReasoningEffortOptions.medium")}>
                        <div style={{ display: "flex", flexDirection: "column", padding: "4px 0" }}>
                            <span style={{ fontSize: "13px", fontWeight: 500 }}>{t("labels.agenticReasoningEffortOptions.medium")}</span>
                            <span style={{ fontSize: "11px", color: "#666", marginTop: "2px" }}>Comprehensive multi-source search</span>
                        </div>
                    </Option>
                </Dropdown>
            )}
        </div>
    );
};

/**
 * MobileDropdownPanel - Expanded panel shown below the input on mobile.
 * Contains full-width category and depth dropdowns.
 */
export interface MobileDropdownPanelProps {
    categories: Category[];
    categoriesLoading: boolean;
    includeCategory: string;
    setIncludeCategory: (value: string) => void;
    allCategoriesSelected: boolean;
    setAllCategoriesSelected: (value: boolean) => void;
    agenticReasoningEffort: string;
    setAgenticReasoningEffort: (value: string) => void;
    showCategoryFilter: boolean;
    showAgenticRetrievalOption: boolean;
    useAgenticRetrieval: boolean;
    isLoading: boolean;
}

export const MobileDropdownPanel: React.FC<MobileDropdownPanelProps> = ({
    categories,
    categoriesLoading,
    includeCategory,
    setIncludeCategory,
    allCategoriesSelected,
    setAllCategoriesSelected,
    agenticReasoningEffort,
    setAgenticReasoningEffort,
    showCategoryFilter,
    showAgenticRetrievalOption,
    useAgenticRetrieval,
    isLoading
}) => {
    const { t } = useTranslation();

    const includeKeys = useMemo(
        () =>
            includeCategory
                ? includeCategory
                      .split(",")
                      .map(s => s.trim())
                      .filter(Boolean)
                : [],
        [includeCategory]
    );

    const categoryOptions = useMemo(
        () => [
            { key: "", text: "All Sources" },
            ...categories
                .filter(c => typeof c?.key === "string" && typeof c?.text === "string" && c.key !== "")
                .map(c => ({
                    key: c.key,
                    text: enhanceDisplayName(c.text)
                }))
        ],
        [categories]
    );

    const selectedOptions = useMemo(() => {
        if (allCategoriesSelected) return [""];
        return includeKeys;
    }, [allCategoriesSelected, includeKeys]);

    const categoryDisplayValue = useMemo(() => {
        if (allCategoriesSelected) return "All Sources";
        if (includeKeys.length === 0) return "Select source";
        if (includeKeys.length === 1) {
            const match = categoryOptions.find(o => o.key === includeKeys[0]);
            return match?.text || includeKeys[0];
        }
        return `${includeKeys.length} sources selected`;
    }, [allCategoriesSelected, includeKeys, categoryOptions]);

    const handleCategorySelect = (_ev: unknown, data: { optionValue?: string; selectedOptions: string[] }) => {
        const key = data.optionValue ?? "";
        if (key === "") {
            setAllCategoriesSelected(!allCategoriesSelected);
            setIncludeCategory("");
            return;
        }
        setAllCategoriesSelected(false);
        const newSelected = data.selectedOptions.filter(k => k !== "");
        setIncludeCategory(newSelected.join(","));
    };

    const handleDepthSelect = (_ev: unknown, data: { optionValue?: string }) => {
        setAgenticReasoningEffort(data.optionValue || "low");
    };

    return (
        <div
            style={{
                background: "#fff",
                border: "1px solid #e0e0e0",
                borderRadius: "8px",
                padding: "12px",
                marginTop: "8px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)"
            }}
        >
            {showCategoryFilter && (
                <div style={{ marginBottom: "12px" }}>
                    <label style={{ display: "block", fontSize: "13px", fontWeight: 600, color: "#444", marginBottom: "6px" }}>Select Source</label>
                    <Dropdown
                        data-testid="chat-source-filter-mobile"
                        button={{
                            id: "chat-source-filter-mobile-button",
                            "aria-label": "Source filter"
                        }}
                        multiselect
                        aria-label="Source filter"
                        value={categoryDisplayValue}
                        selectedOptions={selectedOptions}
                        onOptionSelect={handleCategorySelect}
                        disabled={isLoading || categoriesLoading}
                        placeholder="Select source"
                        style={{ width: "100%" }}
                    >
                        {categoryOptions.map(opt => (
                            <Option id={buildSourceOptionId(opt.key)} key={opt.key} value={opt.key} text={opt.text}>
                                {opt.text}
                            </Option>
                        ))}
                    </Dropdown>
                </div>
            )}
            {showAgenticRetrievalOption && useAgenticRetrieval && (
                <div>
                    <label style={{ display: "block", fontSize: "13px", fontWeight: 600, color: "#444", marginBottom: "6px" }}>Search Depth</label>
                    <Dropdown
                        value={
                            {
                                minimal: t("labels.agenticReasoningEffortOptions.minimal"),
                                low: t("labels.agenticReasoningEffortOptions.low"),
                                medium: t("labels.agenticReasoningEffortOptions.medium")
                            }[agenticReasoningEffort] || "Depth"
                        }
                        selectedOptions={[agenticReasoningEffort]}
                        onOptionSelect={handleDepthSelect}
                        disabled={isLoading}
                        style={{ width: "100%" }}
                    >
                        <Option value="minimal" text={t("labels.agenticReasoningEffortOptions.minimal")}>
                            <div style={{ padding: "4px 0" }}>
                                <div style={{ fontWeight: 500 }}>{t("labels.agenticReasoningEffortOptions.minimal")}</div>
                                <div style={{ fontSize: "12px", color: "#666", marginTop: "2px" }}>Fast single search</div>
                            </div>
                        </Option>
                        <Option value="low" text={t("labels.agenticReasoningEffortOptions.low")}>
                            <div style={{ padding: "4px 0" }}>
                                <div style={{ fontWeight: 500 }}>{t("labels.agenticReasoningEffortOptions.low")}</div>
                                <div style={{ fontSize: "12px", color: "#666", marginTop: "2px" }}>Balanced search depth (recommended)</div>
                            </div>
                        </Option>
                        <Option value="medium" text={t("labels.agenticReasoningEffortOptions.medium")}>
                            <div style={{ padding: "4px 0" }}>
                                <div style={{ fontWeight: 500 }}>{t("labels.agenticReasoningEffortOptions.medium")}</div>
                                <div style={{ fontSize: "12px", color: "#666", marginTop: "2px" }}>Comprehensive multi-source search</div>
                            </div>
                        </Option>
                    </Dropdown>
                </div>
            )}
        </div>
    );
};

export default ChatInputControls;

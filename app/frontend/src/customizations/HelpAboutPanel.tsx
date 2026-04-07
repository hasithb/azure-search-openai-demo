// Help & About Panel Component
// ==============================
// Provides comprehensive help, usage instructions, and privacy information
// for non-technical users of the Civil Procedure Copilot.
// Migrated from @fluentui/react v8 to @fluentui/react-components v9.

import React, { useState } from "react";
import { OverlayDrawer, DrawerBody, DrawerHeader, DrawerHeaderTitle, Button, Text, TabList, Tab, Tooltip, Link, makeStyles } from "@fluentui/react-components";
import {
    DismissRegular,
    InfoRegular,
    LightbulbRegular,
    GridRegular,
    FlashRegular,
    ShieldRegular,
    CheckmarkRegular,
    CursorClickRegular,
    OpenRegular,
    DocumentRegular,
    ThumbLikeRegular,
    ThumbDislikeRegular,
    WarningRegular
} from "@fluentui/react-icons";

// Styles using makeStyles (v9 replacement for mergeStyles)
const useStyles = makeStyles({
    helpButton: {
        position: "fixed",
        bottom: "24px",
        right: "24px",
        zIndex: 100,
        backgroundColor: "#0078d4",
        borderRadius: "50%",
        width: "42px",
        height: "42px",
        minWidth: "42px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        boxShadow: "0 2px 8px rgba(0, 0, 0, 0.2)",
        color: "#fff",
        fontSize: "20px",
        fontWeight: 700,
        ":hover": {
            backgroundColor: "#106ebe",
            color: "#fff"
        }
    },
    helpButtonGlyph: {
        lineHeight: 1,
        transform: "translateY(-1px)"
    },
    panelContent: {
        paddingTop: "0",
        paddingRight: "24px",
        paddingBottom: "24px",
        paddingLeft: "24px"
    },
    section: {
        marginBottom: "24px",
        paddingTop: "16px",
        paddingRight: "16px",
        paddingBottom: "16px",
        paddingLeft: "16px",
        backgroundColor: "#f8f9fa",
        borderRadius: "8px",
        border: "1px solid #e1e4e8"
    },
    featureBox: {
        paddingTop: "16px",
        paddingRight: "16px",
        paddingBottom: "16px",
        paddingLeft: "16px",
        backgroundColor: "#fff",
        borderRadius: "8px",
        border: "1px solid #e1e4e8",
        marginBottom: "12px"
    },
    tipBox: {
        paddingTop: "12px",
        paddingRight: "16px",
        paddingBottom: "12px",
        paddingLeft: "16px",
        backgroundColor: "#fff4ce",
        borderLeft: "4px solid #ffb900",
        borderTopRightRadius: "8px",
        borderBottomRightRadius: "8px",
        marginBottom: "12px"
    },
    warningBox: {
        paddingTop: "12px",
        paddingRight: "16px",
        paddingBottom: "12px",
        paddingLeft: "16px",
        backgroundColor: "#fde7e9",
        borderLeft: "4px solid #d13438",
        borderTopRightRadius: "8px",
        borderBottomRightRadius: "8px",
        marginBottom: "12px"
    }
});

const iconBoxStyle: React.CSSProperties = {
    width: "48px",
    height: "48px",
    borderRadius: "8px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginRight: "16px",
    flexShrink: 0
};

const stepNumberStyle: React.CSSProperties = {
    width: "32px",
    height: "32px",
    borderRadius: "50%",
    backgroundColor: "#0078d4",
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 600,
    marginRight: "12px",
    flexShrink: 0
};

interface FeatureCardProps {
    icon: React.ReactElement;
    iconColor: string;
    title: string;
    description: string;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, iconColor, title, description }) => {
    const classes = useStyles();
    return (
        <div className={classes.featureBox}>
            <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start" }}>
                <div style={{ ...iconBoxStyle, backgroundColor: iconColor + "20" }}>{icon}</div>
                <div style={{ display: "flex", flexDirection: "column" }}>
                    <Text size={400} weight="semibold" style={{ marginBottom: 4 }}>
                        {title}
                    </Text>
                    <Text size={200} style={{ color: "#605e5c" }}>
                        {description}
                    </Text>
                </div>
            </div>
        </div>
    );
};

interface StepProps {
    number: number;
    title: string;
    description: string;
}

const Step: React.FC<StepProps> = ({ number, title, description }) => (
    <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start", marginBottom: 16 }}>
        <div style={stepNumberStyle}>{number}</div>
        <div style={{ display: "flex", flexDirection: "column" }}>
            <Text size={300} weight="semibold">
                {title}
            </Text>
            <Text size={200} style={{ color: "#605e5c" }}>
                {description}
            </Text>
        </div>
    </div>
);

export const HelpAboutPanel: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [selectedTab, setSelectedTab] = useState("about");
    const classes = useStyles();

    return (
        <>
            {/* Help Button - Bottom Right */}
            <Tooltip content="Help & About" relationship="label" positioning="before">
                <Button
                    icon={
                        <span aria-hidden="true" className={classes.helpButtonGlyph}>
                            ?
                        </span>
                    }
                    onClick={() => setIsOpen(true)}
                    aria-label="Help and About"
                    appearance="transparent"
                    className={classes.helpButton}
                />
            </Tooltip>

            {/* Main Drawer */}
            <OverlayDrawer
                open={isOpen}
                onOpenChange={(_, { open }) => {
                    if (!open) setIsOpen(false);
                }}
                position="end"
                size="medium"
            >
                <DrawerHeader>
                    <DrawerHeaderTitle action={<Button appearance="subtle" icon={<DismissRegular />} onClick={() => setIsOpen(false)} aria-label="Close" />}>
                        Civil Procedure Copilot
                    </DrawerHeaderTitle>
                </DrawerHeader>
                <DrawerBody>
                    <div className={classes.panelContent}>
                        <TabList selectedValue={selectedTab} onTabSelect={(_, data) => setSelectedTab(data.value as string)}>
                            <Tab value="about" icon={<InfoRegular />}>
                                About
                            </Tab>
                            <Tab value="howItWorks" icon={<LightbulbRegular />}>
                                How It Works
                            </Tab>
                            <Tab value="features" icon={<GridRegular />}>
                                Features
                            </Tab>
                            <Tab value="tips" icon={<FlashRegular />}>
                                Tips
                            </Tab>
                            <Tab value="privacy" icon={<ShieldRegular />}>
                                Privacy
                            </Tab>
                        </TabList>

                        {/* About Tab */}
                        {selectedTab === "about" && (
                            <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: 16 }}>
                                <div className={classes.section}>
                                    <Text size={400} weight="semibold" style={{ marginBottom: 12, display: "block" }}>
                                        🔨 What is this tool?
                                    </Text>
                                    <Text>
                                        This AI-powered research assistant helps you search and query the Civil Procedure Rules (CPR), Practice Directions, and
                                        Court Guides for England and Wales.
                                    </Text>
                                </div>

                                {/* INDEX-SOURCES: When adding or removing documents from the search index,
                                    update this list to match. See scripts/court_guides_processing_pipeline/outputs_azure_di/
                                    for the current set of processed court guides. */}
                                <div className={classes.section}>
                                    <Text size={400} weight="semibold" style={{ marginBottom: 12, display: "block" }}>
                                        📚 Available Documents
                                    </Text>
                                    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                        <Text size={200}>• Civil Procedure Rules (Parts 1-89) and Practice Directions</Text>
                                        <Text size={200}>• Commercial Court Guide (11th Edition, July 2023)</Text>
                                        <Text size={200}>• King's Bench Division Guide (2025 Edition)</Text>
                                        <Text size={200}>• Chancery Guide (2024 Edition)</Text>
                                        <Text size={200}>• Patents Court Guide (February 2025)</Text>
                                        <Text size={200}>• Technology & Construction Court Guide (October 2022)</Text>
                                        <Text size={200}>• Circuit Commercial Court Guide (August 2023)</Text>
                                        <Text size={200}>• Court of Appeal Civil Division Guide</Text>
                                        <Text size={200}>• Senior Courts Costs Office Guide</Text>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* How It Works Tab */}
                        {selectedTab === "howItWorks" && (
                            <div style={{ display: "flex", flexDirection: "column", gap: "20px", marginTop: 16 }}>
                                <div className={classes.section}>
                                    <Text size={400} weight="semibold" style={{ marginBottom: 16, display: "block" }}>
                                        🔄 Quick Start Guide
                                    </Text>
                                    <Step
                                        number={1}
                                        title="Type Your Question"
                                        description="Enter your legal research question in plain English. For example: 'What are the time limits for filing a defence?' or 'What documents are needed for a CMC in the Commercial Court?'"
                                    />
                                    <Step
                                        number={2}
                                        title="Review the Answer"
                                        description="The AI searches through CPR documents and provides a response with numbered citations [1], [2], [3] that reference specific source documents."
                                    />
                                    <Step
                                        number={3}
                                        title="Click Citations to Verify"
                                        description="Click on any numbered citation to open the Supporting Content panel, which shows the exact text from the source document that the AI used."
                                    />
                                    <Step
                                        number={4}
                                        title="Continue the Conversation"
                                        description="Ask follow-up questions to refine your understanding. The chat remembers context from previous questions in the same session."
                                    />
                                </div>

                                {/* Visual Guide: The Interface */}
                                <div className={classes.section}>
                                    <Text size={400} weight="semibold" style={{ marginBottom: 12, display: "block" }}>
                                        🖥️ Understanding the Interface
                                    </Text>

                                    {/* Source Dropdown */}
                                    <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginBottom: 16 }}>
                                        <Text weight="semibold">Source Filter (Optional)</Text>
                                        <div style={{ padding: "12px", backgroundColor: "#f8f9fa", borderRadius: "6px", border: "1px solid #e1e4e8" }}>
                                            <Text size={200}>
                                                Use the dropdown next to the input box to filter by document source: CPR & Practice Directions, or specific
                                                Court Guides (e.g., Commercial Court Guide, Chancery Guide). Select "All Sources" to search across all
                                                documents.
                                            </Text>
                                        </div>
                                    </div>

                                    {/* Search Depth */}
                                    <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginBottom: 16 }}>
                                        <Text weight="semibold">Search Depth</Text>
                                        <div style={{ padding: "12px", backgroundColor: "#f8f9fa", borderRadius: "6px", border: "1px solid #e1e4e8" }}>
                                            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                                <Text size={200}>
                                                    <strong>Quick:</strong> Fast single search - best for simple lookups like "What is CPR Part 31?"
                                                </Text>
                                                <Text size={200}>
                                                    <strong>Standard:</strong> Balanced search depth - recommended for most legal questions
                                                </Text>
                                                <Text size={200}>
                                                    <strong>Thorough:</strong> Comprehensive multi-source search - best for complex analysis spanning multiple
                                                    rules
                                                </Text>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Understanding Citations */}
                                <div className={classes.section}>
                                    <Text size={400} weight="semibold" style={{ marginBottom: 12, display: "block" }}>
                                        📖 Understanding Citations
                                    </Text>
                                    <Text size={200} style={{ marginBottom: 12, display: "block", color: "#666" }}>
                                        Every answer includes numbered citations that link to source documents:
                                    </Text>
                                    <div
                                        style={{
                                            padding: "16px",
                                            backgroundColor: "#fff",
                                            borderRadius: "8px",
                                            border: "1px solid #e1e4e8",
                                            marginBottom: "12px"
                                        }}
                                    >
                                        <div
                                            style={{
                                                padding: "12px",
                                                backgroundColor: "#f8f9fa",
                                                borderRadius: "6px",
                                                fontStyle: "italic"
                                            }}
                                        >
                                            "Standard disclosure requires a party to disclose documents on which it relies
                                            <span
                                                style={{
                                                    backgroundColor: "#deecf9",
                                                    padding: "2px 6px",
                                                    borderRadius: "4px",
                                                    margin: "0 4px",
                                                    cursor: "pointer",
                                                    fontWeight: 600
                                                }}
                                            >
                                                [1]
                                            </span>
                                            and documents which adversely affect its case
                                            <span
                                                style={{
                                                    backgroundColor: "#deecf9",
                                                    padding: "2px 6px",
                                                    borderRadius: "4px",
                                                    margin: "0 4px",
                                                    cursor: "pointer",
                                                    fontWeight: 600
                                                }}
                                            >
                                                [2]
                                            </span>
                                            ."
                                        </div>
                                    </div>
                                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                        <CursorClickRegular style={{ color: "#0078d4" }} />
                                        <Text size={200}>Click any blue citation number to view the supporting content from the source document</Text>
                                    </div>
                                </div>

                                {/* Supporting Content Panel */}
                                <div className={classes.section}>
                                    <Text size={400} weight="semibold" style={{ marginBottom: 12, display: "block" }}>
                                        📄 Supporting Content Panel
                                    </Text>
                                    <div style={{ padding: "12px", backgroundColor: "#f8f9fa", borderRadius: "6px", border: "1px solid #e1e4e8" }}>
                                        <Text size={200}>
                                            When you click a citation, a panel opens on the right showing the exact text from CPR, Practice Directions, or Court
                                            Guides that the AI used. This is your primary source for verification - always check that the AI's interpretation
                                            matches the original text.
                                        </Text>
                                    </div>
                                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px", marginTop: 12 }}>
                                        <OpenRegular style={{ color: "#0078d4" }} />
                                        <Text size={200}>Click "View Source in New Tab" to open the full source document</Text>
                                    </div>
                                </div>

                                {/* Feedback */}
                                <div className={classes.section}>
                                    <Text size={400} weight="semibold" style={{ marginBottom: 12, display: "block" }}>
                                        👍 Providing Feedback
                                    </Text>
                                    <Text size={200} style={{ marginBottom: 12, display: "block" }}>
                                        Use the thumbs up/down buttons below each answer to rate the response quality. Your feedback helps improve the accuracy
                                        of future answers.
                                    </Text>
                                </div>
                            </div>
                        )}

                        {/* Features Tab */}
                        {selectedTab === "features" && (
                            <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: 16 }}>
                                {/* Citations */}
                                <Text size={400} weight="semibold">
                                    📝 Understanding Citations
                                </Text>
                                <div className={classes.section}>
                                    <Text style={{ marginBottom: 12, display: "block" }}>
                                        Every answer includes numbered citations like <strong>[1]</strong>, <strong>[2]</strong>, <strong>[3]</strong> that link
                                        to source documents.
                                    </Text>
                                    <div
                                        style={{
                                            padding: "12px",
                                            backgroundColor: "#fff",
                                            borderRadius: "8px",
                                            border: "1px solid #0078d4"
                                        }}
                                    >
                                        <Text style={{ fontStyle: "italic" }}>
                                            "Standard disclosure requires a party to disclose documents on which it relies{" "}
                                            <span style={{ backgroundColor: "#deecf9", padding: "2px 6px", borderRadius: "4px" }}>[1]</span>, documents which
                                            adversely affect its case{" "}
                                            <span style={{ backgroundColor: "#deecf9", padding: "2px 6px", borderRadius: "4px" }}>[2]</span>, and documents
                                            which support another party's case{" "}
                                            <span style={{ backgroundColor: "#deecf9", padding: "2px 6px", borderRadius: "4px" }}>[3]</span>
                                            ."
                                        </Text>
                                    </div>
                                    <div style={{ display: "flex", flexDirection: "row", gap: "8px", marginTop: 12 }}>
                                        <CursorClickRegular style={{ color: "#0078d4" }} />
                                        <Text size={200}>
                                            <strong>Click any citation number</strong> to view the source document
                                        </Text>
                                    </div>
                                </div>

                                {/* Supporting Content */}
                                <Text size={400} weight="semibold">
                                    📄 Supporting Content Panel
                                </Text>
                                <div className={classes.section}>
                                    <div style={{ display: "flex", flexDirection: "row", gap: "16px" }}>
                                        <div
                                            style={{
                                                width: "60px",
                                                height: "80px",
                                                backgroundColor: "#f0f0f0",
                                                border: "1px solid #ccc",
                                                borderRadius: "4px",
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center"
                                            }}
                                        >
                                            <DocumentRegular style={{ fontSize: 24, color: "#666" }} />
                                        </div>
                                        <div style={{ display: "flex", flexDirection: "column" }}>
                                            <Text weight="semibold">What is it?</Text>
                                            <Text size={200}>The exact text passages from CPR documents that the AI used to generate its answer.</Text>
                                            <Text size={200} style={{ marginTop: 8 }}>
                                                <strong>This is the PRIMARY SOURCE</strong> - always verify the AI's interpretation against these original
                                                passages.
                                            </Text>
                                        </div>
                                    </div>
                                </div>

                                {/* Source Filter */}
                                <Text size={400} weight="semibold">
                                    🏷️ Source Filter
                                </Text>
                                <div className={classes.section}>
                                    <Text style={{ marginBottom: 12, display: "block" }}>
                                        Use the dropdown to narrow your search to specific document sources:
                                    </Text>
                                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                                        <Text size={200}>• CPR & Practice Directions</Text>
                                        <Text size={200}>• Commercial Court Guide</Text>
                                        <Text size={200}>• Other Court Guides (Chancery, TCC, etc.)</Text>
                                        <Text size={200}>• All Sources (default)</Text>
                                    </div>
                                </div>

                                {/* Feedback */}
                                <Text size={400} weight="semibold">
                                    👍👎 Feedback Buttons
                                </Text>
                                <div className={classes.section}>
                                    <Text style={{ marginBottom: 12, display: "block" }}>Help improve the tool by rating responses:</Text>
                                    <div style={{ display: "flex", flexDirection: "row", gap: "24px" }}>
                                        <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                            <ThumbLikeRegular style={{ color: "#107c10", fontSize: 20 }} />
                                            <Text size={200}>Accurate & helpful</Text>
                                        </div>
                                        <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                            <ThumbDislikeRegular style={{ color: "#d13438", fontSize: 20 }} />
                                            <Text size={200}>Inaccurate or unhelpful</Text>
                                        </div>
                                    </div>
                                    <Text size={200} style={{ marginTop: 12, color: "#605e5c" }}>
                                        You can optionally share your query to help us understand issues.
                                    </Text>
                                </div>
                            </div>
                        )}

                        {/* Tips Tab */}
                        {selectedTab === "tips" && (
                            <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: 16 }}>
                                <Text size={400} weight="semibold">
                                    ✅ Best Practices
                                </Text>

                                <div className={classes.tipBox}>
                                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                        <CheckmarkRegular style={{ color: "#107c10" }} />
                                        <Text size={200}>
                                            <strong>Be specific:</strong> "What is the time limit for filing an acknowledgment of service?"
                                        </Text>
                                    </div>
                                </div>

                                <div className={classes.tipBox}>
                                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                        <CheckmarkRegular style={{ color: "#107c10" }} />
                                        <Text size={200}>
                                            <strong>Use legal terminology:</strong> "disclosure obligations" rather than "sharing documents"
                                        </Text>
                                    </div>
                                </div>

                                <div className={classes.tipBox}>
                                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                        <CheckmarkRegular style={{ color: "#107c10" }} />
                                        <Text size={200}>
                                            <strong>Always verify:</strong> Click citations to check the source text matches the AI's summary
                                        </Text>
                                    </div>
                                </div>

                                <div className={classes.tipBox}>
                                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                        <CheckmarkRegular style={{ color: "#107c10" }} />
                                        <Text size={200}>
                                            <strong>Use follow-up questions:</strong> The chat remembers context from your conversation
                                        </Text>
                                    </div>
                                </div>

                                <Text size={400} weight="semibold" style={{ marginTop: 12 }}>
                                    ⚠️ Important Warnings
                                </Text>

                                <div className={classes.warningBox}>
                                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                        <WarningRegular style={{ color: "#d13438" }} />
                                        <Text size={200}>
                                            <strong>Not for deadline calculations:</strong> Always verify deadlines via official court channels
                                        </Text>
                                    </div>
                                </div>

                                <div className={classes.warningBox}>
                                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                        <WarningRegular style={{ color: "#d13438" }} />
                                        <Text size={200}>
                                            <strong>AI can make mistakes:</strong> Responses are assistive, not authoritative legal advice
                                        </Text>
                                    </div>
                                </div>

                                <div className={classes.warningBox}>
                                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                        <WarningRegular style={{ color: "#d13438" }} />
                                        <Text size={200}>
                                            <strong>Check currency:</strong> Verify Practice Direction dates are current before relying on them
                                        </Text>
                                    </div>
                                </div>

                                <Text size={400} weight="semibold" style={{ marginTop: 12 }}>
                                    💡 Example Queries
                                </Text>
                                <div className={classes.section}>
                                    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                        <Text size={200} style={{ fontStyle: "italic" }}>
                                            "What are the requirements for standard disclosure under CPR Part 31?"
                                        </Text>
                                        <Text size={200} style={{ fontStyle: "italic" }}>
                                            "How do I apply for summary judgment?"
                                        </Text>
                                        <Text size={200} style={{ fontStyle: "italic" }}>
                                            "What are the cost budgeting requirements in the Commercial Court?"
                                        </Text>
                                        <Text size={200} style={{ fontStyle: "italic" }}>
                                            "Explain the pre-action protocol requirements for professional negligence claims"
                                        </Text>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Privacy Tab */}
                        {selectedTab === "privacy" && (
                            <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: 16 }}>
                                <Text size={400} weight="semibold">
                                    🛡️ Data Protection
                                </Text>

                                <div className={classes.section}>
                                    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                                        <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                            <CheckmarkRegular style={{ color: "#107c10" }} />
                                            <Text size={200}>
                                                <strong>NOT used for AI training:</strong> Your queries never train AI models
                                            </Text>
                                        </div>
                                        <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                            <CheckmarkRegular style={{ color: "#107c10" }} />
                                            <Text size={200}>
                                                <strong>NOT shared:</strong> Your queries are isolated - others cannot see them
                                            </Text>
                                        </div>
                                        <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                            <CheckmarkRegular style={{ color: "#107c10" }} />
                                            <Text size={200}>
                                                <strong>NOT sent to OpenAI:</strong> Uses Azure OpenAI (separate enterprise service)
                                            </Text>
                                        </div>
                                        <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "8px" }}>
                                            <CheckmarkRegular style={{ color: "#107c10" }} />
                                            <Text size={200}>
                                                <strong>NOT stored:</strong> No chat history is retained after your session
                                            </Text>
                                        </div>
                                    </div>
                                </div>

                                <Text size={400} weight="semibold">
                                    💾 What is Stored
                                </Text>
                                <div className={classes.section}>
                                    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                        <Text size={200}>
                                            <strong>Legal documents:</strong> CPR, Practice Directions, Court Guides (permanent)
                                        </Text>
                                        <Text size={200}>
                                            <strong>Feedback (optional):</strong> Only if you submit feedback and consent to share your query
                                        </Text>
                                        <Text size={200}>
                                            <strong>Your queries:</strong> NOT stored - discarded after processing
                                        </Text>
                                    </div>
                                </div>

                                <Text size={400} weight="semibold">
                                    ⚙️ Technical Details
                                </Text>
                                <div className={classes.section}>
                                    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                        <Text size={200}>
                                            <strong>AI Models:</strong> GPT-5.4 (answers) · GPT-5.4-nano (query rewrite) via Azure OpenAI
                                        </Text>
                                        <Text size={200}>
                                            <strong>AI Processing:</strong> Global Standard (East US 2)
                                        </Text>
                                        <Text size={200}>
                                            <strong>App & Search:</strong> UK South
                                        </Text>
                                        <Text size={200}>
                                            <strong>Encryption:</strong> TLS 1.2+ in transit, AES-256 at rest
                                        </Text>
                                    </div>
                                </div>

                                <Text size={400} weight="semibold">
                                    📚 Official Documentation
                                </Text>
                                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                    <Link
                                        href="https://learn.microsoft.com/en-gb/legal/cognitive-services/openai/data-privacy"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                    >
                                        Azure OpenAI Data, Privacy & Security →
                                    </Link>
                                    <Link href="https://www.microsoft.com/en-gb/trust-center" target="_blank" rel="noopener noreferrer">
                                        Microsoft Trust Center (UK) →
                                    </Link>
                                </div>
                            </div>
                        )}
                    </div>
                </DrawerBody>
            </OverlayDrawer>
        </>
    );
};

// Data Privacy Notice Component
// ==============================
// Displays important data privacy and security information to users
// about Azure OpenAI data handling policies for the Civil Procedure Copilot.
// Migrated from @fluentui/react v8 to @fluentui/react-components v9.

import React, { useState } from "react";
import {
    OverlayDrawer,
    DrawerBody,
    DrawerHeader,
    DrawerHeaderTitle,
    Button,
    Text,
    MessageBar,
    MessageBarBody,
    MessageBarActions,
    Link
} from "@fluentui/react-components";
import {
    DismissRegular,
    ShieldRegular,
    CheckmarkRegular,
    InfoRegular,
    GlobeRegular,
    BookOpenRegular,
    LightbulbRegular,
    DatabaseRegular,
    ArrowRightRegular,
    GavelRegular
} from "@fluentui/react-icons";

import styles from "./DataPrivacyNotice.module.css";

interface DataPrivacyNoticeProps {
    /** If true, shows a small banner at the bottom of the screen */
    showBanner?: boolean;
}

export const DataPrivacyNotice: React.FC<DataPrivacyNoticeProps> = ({ showBanner = true }) => {
    const [isPanelOpen, setIsPanelOpen] = useState(false);
    const [bannerDismissed, setBannerDismissed] = useState(() => {
        return sessionStorage.getItem("privacyBannerDismissed") === "true";
    });

    const openPanel = () => setIsPanelOpen(true);
    const dismissPanel = () => setIsPanelOpen(false);

    const dismissBanner = () => {
        setBannerDismissed(true);
        sessionStorage.setItem("privacyBannerDismissed", "true");
    };

    return (
        <>
            {/* Privacy Banner */}
            {showBanner && !bannerDismissed && (
                <div className={styles.privacyBanner}>
                    <MessageBar intent="info">
                        <MessageBarBody>
                            <ShieldRegular style={{ marginRight: 8, verticalAlign: "middle" }} />
                            <strong>Data Privacy:</strong> Your queries are not used to train AI models and are not shared with third parties.{" "}
                            <Link onClick={openPanel}>Learn more</Link>
                        </MessageBarBody>
                        <MessageBarActions
                            containerAction={<Button appearance="transparent" icon={<DismissRegular />} onClick={dismissBanner} aria-label="Close" />}
                        />
                    </MessageBar>
                </div>
            )}

            {/* Privacy Info Button (always accessible) */}
            <Button
                icon={<ShieldRegular />}
                title="Data Privacy & Security"
                aria-label="Data Privacy & Security Information"
                onClick={openPanel}
                appearance="subtle"
                className={styles.privacyButton}
            />

            {/* Privacy Drawer */}
            <OverlayDrawer
                open={isPanelOpen}
                onOpenChange={(_, { open }) => {
                    if (!open) dismissPanel();
                }}
                position="end"
                size="medium"
            >
                <DrawerHeader>
                    <DrawerHeaderTitle action={<Button appearance="subtle" icon={<DismissRegular />} onClick={dismissPanel} aria-label="Close" />}>
                        Civil Procedure Copilot: Data Privacy & Security
                    </DrawerHeaderTitle>
                </DrawerHeader>
                <DrawerBody>
                    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }} className={styles.panelContent}>
                        {/* About This System */}
                        <section>
                            <Text size={400} weight="semibold" style={{ display: "block" }} className={styles.sectionTitle}>
                                <GavelRegular className={styles.sectionIcon} />
                                About This Legal Research Tool
                            </Text>
                            <Text style={{ display: "block" }}>
                                Civil Procedure Copilot helps lawyers and legal professionals search and query the <strong>Civil Procedure Rules (CPR)</strong>,
                                Practice Directions, and Court Guides for England and Wales using intelligent search.
                            </Text>
                            <div style={{ display: "flex", flexDirection: "column", gap: "4px", marginTop: 8 }}>
                                <Text style={{ display: "block" }}>
                                    <strong>Document sources indexed:</strong>
                                </Text>
                                <Text style={{ display: "block" }}>• Civil Procedure Rules (Parts 1-89) and Practice Directions</Text>
                                <Text style={{ display: "block" }}>• Commercial Court Guide (11th Edition, July 2023)</Text>
                                <Text style={{ display: "block" }}>• King's Bench Division Guide (2025 Edition)</Text>
                                <Text style={{ display: "block" }}>• Chancery Guide (2022)</Text>
                                <Text style={{ display: "block" }}>• Patents Court Guide (February 2025)</Text>
                                <Text style={{ display: "block" }}>• Technology and Construction Court Guide (October 2022)</Text>
                                <Text style={{ display: "block" }}>• Circuit Commercial Court Guide (August 2023)</Text>
                            </div>
                        </section>

                        {/* Key Assurances Section */}
                        <section>
                            <Text size={400} weight="semibold" style={{ display: "block" }} className={styles.sectionTitle}>
                                <ShieldRegular className={styles.sectionIcon} />
                                Key Data Protection Assurances
                            </Text>
                            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }} className={styles.assuranceList}>
                                <div className={styles.assuranceItem}>
                                    <CheckmarkRegular className={styles.checkIcon} />
                                    <Text>
                                        <strong>Your queries are NOT used for AI training:</strong> Questions you ask about CPR, Practice Directions, or Court
                                        procedures are NOT used to train or improve any AI models. Microsoft contractually prohibits this.
                                    </Text>
                                </div>
                                <div className={styles.assuranceItem}>
                                    <CheckmarkRegular className={styles.checkIcon} />
                                    <Text>
                                        <strong>NOT shared with anyone:</strong> Your queries are processed only within your session. Other users, other
                                        customers, and third parties cannot access your queries. No conversation history is retained after your session.
                                    </Text>
                                </div>
                                <div className={styles.assuranceItem}>
                                    <CheckmarkRegular className={styles.checkIcon} />
                                    <Text>
                                        <strong>Enterprise-grade security:</strong> This uses a private, enterprise deployment—your queries never go to public
                                        AI services. All data remains within secure Microsoft data centres.
                                    </Text>
                                </div>
                                <div className={styles.assuranceItem}>
                                    <CheckmarkRegular className={styles.checkIcon} />
                                    <Text>
                                        <strong>No memory between sessions:</strong> The system does not "remember" your previous questions. Each query is
                                        processed independently with no persistent memory.
                                    </Text>
                                </div>
                            </div>
                        </section>

                        {/* How It Works Section */}
                        <section>
                            <Text size={400} weight="semibold" style={{ display: "block" }} className={styles.sectionTitle}>
                                <ArrowRightRegular className={styles.sectionIcon} />
                                How Your Query is Processed
                            </Text>
                            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                <Text style={{ display: "block" }}>
                                    <strong>1. You ask a question</strong> about CPR rules, court procedures, or practice directions.
                                </Text>
                                <Text style={{ display: "block" }}>
                                    <strong>2. Intelligent search retrieves relevant passages</strong> from the indexed legal documents (CPR, Court Guides,
                                    etc.).
                                </Text>
                                <Text style={{ display: "block" }}>
                                    <strong>3. A response is generated</strong> based on the retrieved passages, with citations back to source documents.
                                </Text>
                                <Text style={{ display: "block" }}>
                                    <strong>4. Your query is immediately discarded</strong> after processing—nothing is stored or retained.
                                </Text>
                            </div>
                        </section>

                        {/* What Data is Stored */}
                        <section>
                            <Text size={400} weight="semibold" style={{ display: "block" }} className={styles.sectionTitle}>
                                <DatabaseRegular className={styles.sectionIcon} />
                                What Data is Stored
                            </Text>
                            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                <Text style={{ display: "block" }}>
                                    <strong>Legal document content:</strong> The CPR, Practice Directions, and Court Guides are stored securely. These are
                                    publicly available legal documents.
                                </Text>
                                <Text style={{ display: "block" }}>
                                    <strong>Conversation history:</strong> Conversation history storage is <strong>not enabled</strong> for this environment.
                                    Your queries are not saved after your browser session ends.
                                </Text>
                                <Text style={{ display: "block" }}>
                                    <strong>Feedback reports:</strong> If you submit feedback using the thumbs up/down buttons, the following is logged for
                                    quality improvement purposes:
                                </Text>
                                <div style={{ display: "flex", flexDirection: "column", gap: "4px", marginLeft: 16 }}>
                                    <Text style={{ display: "block" }}>• Your rating (helpful/unhelpful)</Text>
                                    <Text style={{ display: "block" }}>• Issue categories you selected (e.g., "incorrect citation", "outdated law")</Text>
                                    <Text style={{ display: "block" }}>• Any comments you provide</Text>
                                    <Text style={{ display: "block" }}>• A message ID (anonymous identifier for the response)</Text>
                                    <Text style={{ display: "block" }}>
                                        • <strong>Optionally:</strong> If you choose to share your query and search details when reporting an issue, your
                                        question and conversation history will also be included. You will be shown exactly what will be shared and must
                                        explicitly consent before this data is included.
                                    </Text>
                                </div>
                                <Text style={{ display: "block", fontStyle: "italic" }}>
                                    Note: Feedback is voluntary and anonymous. Your name and email are not attached to feedback reports. Sharing your query is
                                    optional and requires explicit consent.
                                </Text>
                                <Text style={{ display: "block" }}>
                                    <strong>Authentication:</strong> Your Microsoft Entra ID sign-in session is managed securely.
                                </Text>
                                <Text style={{ display: "block", marginTop: 8, fontStyle: "italic" }}>
                                    <strong>Your queries are NOT stored:</strong> The system does not store or retain your queries. Your questions are processed
                                    and immediately discarded—nothing is saved (unless you choose to share them in a feedback report).
                                </Text>
                            </div>
                        </section>

                        {/* Content Safety */}
                        <section>
                            <Text size={400} weight="semibold" style={{ display: "block" }} className={styles.sectionTitle}>
                                <InfoRegular className={styles.sectionIcon} />
                                Content Safety
                            </Text>
                            <Text style={{ display: "block" }}>
                                The system includes content safety measures to prevent misuse. For legitimate legal research queries about CPR and court
                                procedures, this typically has no impact.
                            </Text>
                        </section>

                        {/* Best Practices for Lawyers */}
                        <section>
                            <Text size={400} weight="semibold" style={{ display: "block" }} className={styles.sectionTitle}>
                                <LightbulbRegular className={styles.sectionIcon} />
                                Best Practices for Lawyers
                            </Text>
                            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                <Text style={{ display: "block" }}>
                                    ✅ <strong>Safe to use for:</strong> Researching CPR procedures, court rules, Practice Directions, costs rules, disclosure
                                    requirements, case management, and general procedural questions.
                                </Text>
                                <Text style={{ display: "block" }}>
                                    ⚠️ <strong>Recommendations:</strong>
                                </Text>
                                <Text style={{ display: "block" }}>• Avoid including real client names or case-specific identifiers when possible</Text>
                                <Text style={{ display: "block" }}>
                                    • Use generic placeholders for sensitive details (e.g., "the claimant" rather than specific names)
                                </Text>
                                <Text style={{ display: "block" }}>• Always verify citations against the source documents</Text>
                                <Text style={{ display: "block" }}>• Remember that responses are assistive, not authoritative legal advice</Text>
                            </div>
                        </section>

                        {/* Data Residency */}
                        <section>
                            <Text size={400} weight="semibold" style={{ display: "block" }} className={styles.sectionTitle}>
                                <GlobeRegular className={styles.sectionIcon} />
                                Data Residency & Encryption
                            </Text>
                            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                <Text style={{ display: "block" }}>• All data is processed within secure Microsoft data centres</Text>
                                <Text style={{ display: "block" }}>• Data transmission is encrypted using industry-standard protocols</Text>
                                <Text style={{ display: "block" }}>• Data at rest is encrypted</Text>
                                <Text style={{ display: "block" }}>
                                    • Note: This is a test environment. Production deployments can be configured for UK data centres if required.
                                </Text>
                            </div>
                        </section>

                        {/* Official Documentation Links */}
                        <section>
                            <Text size={400} weight="semibold" style={{ display: "block" }} className={styles.sectionTitle}>
                                <BookOpenRegular className={styles.sectionIcon} />
                                Further Information
                            </Text>
                            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                <Link
                                    href="https://learn.microsoft.com/en-gb/azure/ai-services/openai/concepts/data-privacy"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    Microsoft Data Privacy & Security Documentation →
                                </Link>
                                <Link href="https://www.microsoft.com/en-gb/trust-center" target="_blank" rel="noopener noreferrer">
                                    Microsoft Trust Centre →
                                </Link>
                            </div>
                        </section>

                        {/* Legal Disclaimer */}
                        <section className={styles.disclaimer}>
                            <Text size={200} style={{ display: "block" }}>
                                <strong>Disclaimer:</strong> This summary is provided for informational purposes to help you understand how your data is handled
                                when using Civil Procedure Copilot. For complete and authoritative information about Microsoft's data handling practices, please
                                refer to the official Microsoft documentation linked above. Responses should always be verified against source documents and are
                                not a substitute for professional legal judgement.
                            </Text>
                        </section>
                    </div>
                </DrawerBody>
            </OverlayDrawer>
        </>
    );
};

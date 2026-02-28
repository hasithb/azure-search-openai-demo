import { useState } from "react";
import {
    Button,
    Checkbox,
    Dialog,
    DialogActions,
    DialogBody,
    DialogContent,
    DialogSurface,
    DialogTitle,
    DialogTrigger,
    Divider,
    MessageBar,
    MessageBarBody,
    Radio,
    RadioGroup,
    Text,
    Textarea,
    Tooltip
} from "@fluentui/react-components";
import { ThumbLike24Regular, ThumbLike24Filled, ThumbDislike24Regular, ThumbDislike24Filled } from "@fluentui/react-icons";
import { Thoughts } from "../api";

interface ConversationMessage {
    role: "user" | "assistant";
    content: string;
}

interface Props {
    messageId: string;
    /** The user's prompt for this specific response */
    userPrompt?: string;
    /** The AI's response content */
    aiResponse?: string;
    /** Full conversation history up to this point */
    conversationHistory?: ConversationMessage[];
    /** Thought process data (search queries, retrieved docs, etc.) */
    thoughts?: Thoughts[];
}

export const LegalFeedback = ({ messageId, userPrompt, aiResponse, conversationHistory, thoughts }: Props) => {
    const [status, setStatus] = useState<"none" | "helpful" | "unhelpful">("none");
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [issues, setIssues] = useState<string[]>([]);
    const [comment, setComment] = useState("");
    const [shareContext, setShareContext] = useState<string | undefined>(undefined); // undefined = not chosen yet

    const handleVote = (vote: "helpful" | "unhelpful") => {
        setStatus(vote);
        if (vote === "unhelpful") setIsDialogOpen(true);
        else submitFeedback("helpful", [], "", false);
    };

    const submitFeedback = async (rating: string, selectedIssues: string[], text: string, includeContext: boolean) => {
        try {
            const payload: Record<string, unknown> = {
                message_id: messageId,
                rating,
                issues: selectedIssues,
                comment: text,
                context_shared: includeContext
            };

            // Only include context data if user consented
            if (includeContext) {
                payload.user_prompt = userPrompt || "";
                payload.ai_response = aiResponse || "";
                payload.conversation_history = conversationHistory || [];
                payload.thoughts = thoughts || [];
            }

            await fetch("/api/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
        } catch (e) {
            console.error("Failed to send feedback", e);
        }
        setIsDialogOpen(false);
        // Reset for next time
        setShareContext(undefined);
        setIssues([]);
        setComment("");
    };

    const toggleIssue = (issue: string, checked: boolean) => {
        if (checked) setIssues([...issues, issue]);
        else setIssues(issues.filter(i => i !== issue));
    };

    const canSubmit = shareContext !== undefined; // Must choose yes or no

    const closeDialog = () => {
        setIsDialogOpen(false);
        setShareContext(undefined);
    };

    return (
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Tooltip content="Accurate & Helpful" relationship="label">
                <Button
                    appearance="subtle"
                    icon={status === "helpful" ? <ThumbLike24Filled /> : <ThumbLike24Regular />}
                    onClick={() => handleVote("helpful")}
                    style={{ color: status === "helpful" ? "green" : "inherit" }}
                />
            </Tooltip>
            <Tooltip content="Report Issue" relationship="label">
                <Button
                    appearance="subtle"
                    icon={status === "unhelpful" ? <ThumbDislike24Filled /> : <ThumbDislike24Regular />}
                    onClick={() => handleVote("unhelpful")}
                    style={{ color: status === "unhelpful" ? "red" : "inherit" }}
                />
            </Tooltip>

            <Dialog
                open={isDialogOpen}
                onOpenChange={(_, data) => {
                    if (!data.open) closeDialog();
                }}
            >
                <DialogSurface style={{ minWidth: 600, maxWidth: 800 }}>
                    <DialogBody>
                        <DialogTitle>Report an Issue</DialogTitle>
                        <DialogContent>
                            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                                {/* Issue Selection */}
                                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                    <Text weight="semibold" size={400}>
                                        What was wrong with this response?
                                    </Text>
                                    <Checkbox label="Incorrect Citation / Reference" onChange={(_, data) => toggleIssue("wrong_citation", !!data.checked)} />
                                    <Checkbox label="Hallucinated / Fake Citation" onChange={(_, data) => toggleIssue("hallucination", !!data.checked)} />
                                    <Checkbox label="Outdated Law" onChange={(_, data) => toggleIssue("outdated", !!data.checked)} />
                                    <Checkbox label="Missing Key Information" onChange={(_, data) => toggleIssue("missing_info", !!data.checked)} />
                                    <div>
                                        <Text weight="regular" size={300} style={{ display: "block", marginBottom: 4 }}>
                                            Correction / Additional Notes
                                        </Text>
                                        <Textarea resize="vertical" rows={3} onChange={(_, data) => setComment(data.value)} style={{ width: "100%" }} />
                                    </div>
                                </div>

                                <Divider />

                                {/* Context Sharing - REQUIRED */}
                                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                                    <Text weight="semibold" size={400}>
                                        Share your prompt and search details? <span style={{ color: "red" }}>*</span>
                                    </Text>
                                    <MessageBar intent="info">
                                        <MessageBarBody>
                                            To help us understand and fix this issue, we need to see what you asked and how the system searched for an answer.
                                            This is optional but very helpful for diagnosis.
                                        </MessageBarBody>
                                    </MessageBar>

                                    <RadioGroup value={shareContext || ""} onChange={(_, data) => setShareContext(data.value)} required>
                                        <Radio value="yes" label="Yes, include my prompt and search details to help diagnose the issue" />
                                        <Radio value="no" label="No, submit feedback without my prompt" />
                                    </RadioGroup>

                                    {/* Preview of what will be shared */}
                                    {shareContext === "yes" && (
                                        <div
                                            style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                gap: 12,
                                                backgroundColor: "#f5f5f5",
                                                padding: 16,
                                                borderRadius: 4,
                                                maxHeight: 300,
                                                overflowY: "auto"
                                            }}
                                        >
                                            <Text weight="semibold">📋 Data that will be shared:</Text>

                                            {/* User's Prompt */}
                                            <div>
                                                <Text weight="semibold">Your Question:</Text>
                                                <div
                                                    style={{
                                                        backgroundColor: "#fff",
                                                        padding: 8,
                                                        borderRadius: 4,
                                                        marginTop: 4,
                                                        border: "1px solid #ddd"
                                                    }}
                                                >
                                                    <Text>{userPrompt || "(No prompt available)"}</Text>
                                                </div>
                                            </div>

                                            {/* Conversation History (if multi-turn) */}
                                            {conversationHistory && conversationHistory.length > 1 && (
                                                <div>
                                                    <Text weight="semibold">Conversation History ({conversationHistory.length} messages):</Text>
                                                    <div
                                                        style={{
                                                            backgroundColor: "#fff",
                                                            padding: 8,
                                                            borderRadius: 4,
                                                            marginTop: 4,
                                                            border: "1px solid #ddd",
                                                            maxHeight: 150,
                                                            overflowY: "auto"
                                                        }}
                                                    >
                                                        {conversationHistory.map((msg, idx) => (
                                                            <div key={idx} style={{ marginBottom: 8 }}>
                                                                <Text weight="semibold" style={{ color: msg.role === "user" ? "#0066cc" : "#666" }}>
                                                                    {msg.role === "user" ? "You:" : "Assistant:"}
                                                                </Text>
                                                                <Text style={{ marginLeft: 8, display: "block" }}>
                                                                    {msg.content.substring(0, 200)}
                                                                    {msg.content.length > 200 ? "..." : ""}
                                                                </Text>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {shareContext === "no" && (
                                        <MessageBar intent="warning">
                                            <MessageBarBody>
                                                Without your prompt and search details, it may be difficult to diagnose and fix this issue.
                                            </MessageBarBody>
                                        </MessageBar>
                                    )}
                                </div>
                            </div>
                        </DialogContent>
                        <DialogActions>
                            <Button
                                appearance="primary"
                                onClick={() => submitFeedback("unhelpful", issues, comment, shareContext === "yes")}
                                disabled={!canSubmit}
                            >
                                Submit Report
                            </Button>
                            <DialogTrigger disableButtonEnhancement>
                                <Button appearance="secondary" onClick={closeDialog}>
                                    Cancel
                                </Button>
                            </DialogTrigger>
                        </DialogActions>
                    </DialogBody>
                </DialogSurface>
            </Dialog>
        </div>
    );
};

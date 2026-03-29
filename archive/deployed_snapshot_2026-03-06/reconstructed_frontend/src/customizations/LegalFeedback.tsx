import { useState } from "react";
import {
    Stack,
    IconButton,
    Dialog,
    DialogFooter,
    PrimaryButton,
    DefaultButton,
    Checkbox,
    TextField,
    Text,
    ChoiceGroup,
    IChoiceGroupOption,
    MessageBar,
    MessageBarType,
    Separator
} from "@fluentui/react";
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
    const [isDialogVisible, setIsDialogVisible] = useState(false);
    const [issues, setIssues] = useState<string[]>([]);
    const [comment, setComment] = useState("");
    const [shareContext, setShareContext] = useState<string | undefined>(undefined); // undefined = not chosen yet

    const handleVote = (vote: "helpful" | "unhelpful") => {
        setStatus(vote);
        if (vote === "unhelpful") setIsDialogVisible(true);
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
        setIsDialogVisible(false);
        // Reset for next time
        setShareContext(undefined);
        setIssues([]);
        setComment("");
    };

    const toggleIssue = (issue: string, checked?: boolean) => {
        if (checked) setIssues([...issues, issue]);
        else setIssues(issues.filter(i => i !== issue));
    };

    const shareContextOptions: IChoiceGroupOption[] = [
        { key: "yes", text: "Yes, include my prompt and search details to help diagnose the issue" },
        { key: "no", text: "No, submit feedback without my prompt" }
    ];

    const canSubmit = shareContext !== undefined; // Must choose yes or no

    return (
        <Stack horizontal tokens={{ childrenGap: 8 }} verticalAlign="center">
            <IconButton
                iconProps={{ iconName: status === "helpful" ? "LikeSolid" : "Like" }}
                title="Accurate & Helpful"
                onClick={() => handleVote("helpful")}
                style={{ color: status === "helpful" ? "green" : "inherit" }}
            />
            <IconButton
                iconProps={{ iconName: status === "unhelpful" ? "DislikeSolid" : "Dislike" }}
                title="Report Issue"
                onClick={() => handleVote("unhelpful")}
                style={{ color: status === "unhelpful" ? "red" : "inherit" }}
            />

            <Dialog
                hidden={!isDialogVisible}
                onDismiss={() => {
                    setIsDialogVisible(false);
                    setShareContext(undefined);
                }}
                dialogContentProps={{ title: "Report an Issue" }}
                minWidth={600}
                maxWidth={800}
            >
                <Stack tokens={{ childrenGap: 16 }}>
                    {/* Issue Selection */}
                    <Stack tokens={{ childrenGap: 8 }}>
                        <Text variant="mediumPlus" style={{ fontWeight: 600 }}>
                            What was wrong with this response?
                        </Text>
                        <Checkbox label="Incorrect Citation / Reference" onChange={(_, c) => toggleIssue("wrong_citation", c)} />
                        <Checkbox label="Hallucinated / Fake Citation" onChange={(_, c) => toggleIssue("hallucination", c)} />
                        <Checkbox label="Outdated Law" onChange={(_, c) => toggleIssue("outdated", c)} />
                        <Checkbox label="Missing Key Information" onChange={(_, c) => toggleIssue("missing_info", c)} />
                        <TextField label="Correction / Additional Notes" multiline rows={3} onChange={(_, v) => setComment(v || "")} />
                    </Stack>

                    <Separator />

                    {/* Context Sharing - REQUIRED */}
                    <Stack tokens={{ childrenGap: 12 }}>
                        <Text variant="mediumPlus" style={{ fontWeight: 600 }}>
                            Share your prompt and search details? <span style={{ color: "red" }}>*</span>
                        </Text>
                        <MessageBar messageBarType={MessageBarType.info}>
                            To help us understand and fix this issue, we need to see what you asked and how the system searched for an answer. This is optional
                            but very helpful for diagnosis.
                        </MessageBar>

                        <ChoiceGroup options={shareContextOptions} selectedKey={shareContext} onChange={(_, option) => setShareContext(option?.key)} required />

                        {/* Preview of what will be shared */}
                        {shareContext === "yes" && (
                            <Stack
                                tokens={{ childrenGap: 12 }}
                                style={{
                                    backgroundColor: "#f5f5f5",
                                    padding: 16,
                                    borderRadius: 4,
                                    maxHeight: 300,
                                    overflowY: "auto"
                                }}
                            >
                                <Text variant="medium" style={{ fontWeight: 600 }}>
                                    📋 Data that will be shared:
                                </Text>

                                {/* User's Prompt */}
                                <div>
                                    <Text style={{ fontWeight: 600 }}>Your Question:</Text>
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
                                        <Text style={{ fontWeight: 600 }}>Conversation History ({conversationHistory.length} messages):</Text>
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
                                                    <Text style={{ fontWeight: 600, color: msg.role === "user" ? "#0066cc" : "#666" }}>
                                                        {msg.role === "user" ? "You:" : "Assistant:"}
                                                    </Text>
                                                    <Text block style={{ marginLeft: 8 }}>
                                                        {msg.content.substring(0, 200)}
                                                        {msg.content.length > 200 ? "..." : ""}
                                                    </Text>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </Stack>
                        )}

                        {shareContext === "no" && (
                            <MessageBar messageBarType={MessageBarType.warning}>
                                Without your prompt and search details, it may be difficult to diagnose and fix this issue.
                            </MessageBar>
                        )}
                    </Stack>
                </Stack>

                <DialogFooter>
                    <PrimaryButton
                        text="Submit Report"
                        onClick={() => submitFeedback("unhelpful", issues, comment, shareContext === "yes")}
                        disabled={!canSubmit}
                    />
                    <DefaultButton
                        text="Cancel"
                        onClick={() => {
                            setIsDialogVisible(false);
                            setShareContext(undefined);
                        }}
                    />
                </DialogFooter>
            </Dialog>
        </Stack>
    );
};

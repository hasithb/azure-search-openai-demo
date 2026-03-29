import { useState, useEffect, useContext } from "react";
import { Stack, TextField } from "@fluentui/react";
import { Button, Tooltip } from "@fluentui/react-components";
import { Send28Filled } from "@fluentui/react-icons";
import { useTranslation } from "react-i18next";

import styles from "./QuestionInput.module.css";
import { SpeechInput } from "./SpeechInput";
import { LoginContext } from "../../loginContext";
import { requireLogin } from "../../authConfig";

interface Props {
    onSend: (question: string) => void;
    disabled: boolean;
    initQuestion?: string;
    placeholder?: string;
    clearOnSend?: boolean;
    showSpeechInput?: boolean;
    leftOfSend?: React.ReactNode;
    autoFocus?: boolean;
}

export const QuestionInput = ({ onSend, disabled, placeholder, clearOnSend, initQuestion, showSpeechInput, leftOfSend, autoFocus }: Props) => {
    const [question, setQuestion] = useState<string>("");
    const { loggedIn } = useContext(LoginContext);
    const { t } = useTranslation();
    const [isComposing, setIsComposing] = useState(false);

    useEffect(() => {
        initQuestion && setQuestion(initQuestion);
    }, [initQuestion]);

    const sendQuestion = () => {
        if (disabled || !question.trim()) {
            return; // Don't send if disabled or no question
        }

        onSend(question);

        // Only clear if clearOnSend is true - let parent handle clearing on success
        // This prevents clearing the question when validation fails
        if (clearOnSend) {
            setQuestion("");
        }
    };

    const onEnterPress = (ev: React.KeyboardEvent<Element>) => {
        if (isComposing) return;

        if (ev.key === "Enter" && !ev.shiftKey) {
            ev.preventDefault();
            sendQuestion();
        }
    };

    const handleCompositionStart = () => {
        setIsComposing(true);
    };
    const handleCompositionEnd = () => {
        setIsComposing(false);
    };

    const onQuestionChange = (_ev: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string) => {
        if (!newValue) {
            setQuestion("");
        } else if (newValue.length <= 1000) {
            setQuestion(newValue);
        }
    };

    const disableRequiredAccessControl = requireLogin && !loggedIn;
    const sendQuestionDisabled = disabled || !question.trim() || disableRequiredAccessControl;

    if (disableRequiredAccessControl) {
        placeholder = "Please login to continue...";
    } else if (disabled && placeholder?.includes("category")) {
        // Keep the category-related placeholder if that's why it's disabled
        // placeholder already set by parent component
    }

    return (
        <Stack horizontal className={styles.questionInputContainer} tokens={{ childrenGap: 8 }}>
            <TextField
                className={styles.questionInputTextArea}
                disabled={disableRequiredAccessControl}
                placeholder={placeholder}
                multiline
                resizable={false}
                borderless
                autoFocus={autoFocus}
                value={question}
                onChange={onQuestionChange}
                onKeyDown={onEnterPress}
                onCompositionStart={handleCompositionStart}
                onCompositionEnd={handleCompositionEnd}
                styles={{
                    root: { flex: 1, minWidth: 0 },
                    fieldGroup: { minHeight: 44 },
                    field: {
                        minHeight: 44,
                        maxHeight: 120,
                        overflowY: "auto",
                        overflowX: "hidden"
                    }
                }}
            />
            <div className={styles.questionInputButtonsContainer}>
                {leftOfSend && <div style={{ marginRight: 8 }}>{leftOfSend}</div>}
                <Tooltip content={t("tooltips.submitQuestion")} relationship="label">
                    <Button
                        icon={<Send28Filled primaryFill="rgba(115, 118, 225, 1)" />}
                        disabled={sendQuestionDisabled}
                        onClick={sendQuestion}
                        style={{
                            backgroundColor: "transparent",
                            border: "none",
                            borderRadius: "4px",
                            padding: "4px",
                            minWidth: "auto",
                            minHeight: "auto",
                            transition: "all 0.15s ease",
                            cursor: sendQuestionDisabled ? "not-allowed" : "pointer",
                            opacity: sendQuestionDisabled ? 0.4 : 1,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center"
                        }}
                        onMouseEnter={e => {
                            if (!sendQuestionDisabled) {
                                const currentIcon = e.currentTarget.querySelector("svg");
                                if (currentIcon) {
                                    currentIcon.style.fill = "rgba(95, 98, 205, 1)";
                                }
                            }
                        }}
                        onMouseLeave={e => {
                            const currentIcon = e.currentTarget.querySelector("svg");
                            if (currentIcon) {
                                currentIcon.style.fill = "rgba(115, 118, 225, 1)";
                            }
                        }}
                    />
                </Tooltip>
            </div>
            {showSpeechInput && <SpeechInput updateQuestion={setQuestion} />}
        </Stack>
    );
};

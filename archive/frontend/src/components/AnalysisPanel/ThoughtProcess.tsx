import { Stack } from "@fluentui/react";
import styles from "./AnalysisPanel.module.css";
import { Thoughts } from "../../api";
import { TokenUsageGraph } from "./TokenUsageGraph";
import { AgentPlan } from "./AgentPlan";

interface Props {
    thoughts: Thoughts[];
}

export const ThoughtProcess = ({ thoughts }: Props) => {
    const renderCode = (content: any) => {
        // Simple fallback without syntax highlighting
        return (
            <pre
                className={styles.tCodeBlock}
                style={{
                    backgroundColor: "#f8f8f8",
                    padding: "10px",
                    borderRadius: "4px",
                    overflow: "auto",
                    fontSize: "0.85em",
                    lineHeight: "1.4"
                }}
            >
                {JSON.stringify(content, null, 2)}
            </pre>
        );
    };

    const blockedTitles = ["Prompt to generate answer", "Prompt to rewrite query", "Query rewrite", "Thought step"];

    return (
        <ul className={styles.tList}>
            {thoughts
                .filter(t => !blockedTitles.includes(t.title))
                .map((t, ind) => {
                    return (
                        <li className={styles.tListItem} key={ind}>
                            <div className={styles.tStep}>{t.title}</div>
                            <Stack horizontal tokens={{ childrenGap: 5 }}>
                                {t.props &&
                                    (Object.keys(t.props).filter(k => k !== "token_usage" && k !== "query_plan") || []).map((k: any) => (
                                        <span className={styles.tProp} key={k}>
                                            {k}: {JSON.stringify(t.props?.[k])}
                                        </span>
                                    ))}
                            </Stack>
                            {t.props?.token_usage && <TokenUsageGraph tokenUsage={t.props.token_usage} reasoningEffort={t.props.reasoning_effort} />}
                            {t.props?.query_plan && <AgentPlan query_plan={t.props.query_plan} description={t.description} />}
                            {Array.isArray(t.description) ? renderCode(t.description) : <div>{t.description}</div>}
                        </li>
                    );
                })}
        </ul>
    );
};

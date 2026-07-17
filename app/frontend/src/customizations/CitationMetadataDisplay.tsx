// Citation Metadata Display
// =========================
// Renders structured citation metadata (subsection, sourcepage, sourcefile,
// category) as styled badges in the SupportingContent item header.
// Only renders fields that have non-empty values.
// Part of the merge-safe customizations architecture.

import React from "react";
import type { StructuredCitationMetadata } from "./citationMetadata";

interface Props {
    metadata: StructuredCitationMetadata;
}

const badgeStyle: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    padding: "2px 8px",
    borderRadius: "4px",
    fontSize: "12px",
    lineHeight: "1.4",
    whiteSpace: "nowrap"
};

const subsectionBadge: React.CSSProperties = {
    ...badgeStyle,
    background: "#e8f4fd",
    color: "#0078d4",
    fontWeight: 600
};

const sourcepageBadge: React.CSSProperties = {
    ...badgeStyle,
    background: "#f3f2f1",
    color: "#323130"
};

const sourcefileBadge: React.CSSProperties = {
    ...badgeStyle,
    background: "#f3f2f1",
    color: "#605e5c"
};

const categoryBadge: React.CSSProperties = {
    ...badgeStyle,
    background: "#fff4ce",
    color: "#6b5900"
};

const containerStyle: React.CSSProperties = {
    display: "flex",
    flexWrap: "wrap",
    gap: "6px",
    marginBottom: "8px"
};

export const CitationMetadataDisplay: React.FC<Props> = ({ metadata }) => {
    const { subsectionId, sourcepage, sourcefile, category } = metadata;
    const hasAny = subsectionId || sourcepage || sourcefile || category;
    if (!hasAny) return null;

    return (
        <div style={containerStyle}>
            {subsectionId && <span style={subsectionBadge}>§ {subsectionId}</span>}
            {sourcepage && <span style={sourcepageBadge}>{sourcepage}</span>}
            {sourcefile && <span style={sourcefileBadge}>{sourcefile}</span>}
            {category && <span style={categoryBadge}>{category}</span>}
        </div>
    );
};

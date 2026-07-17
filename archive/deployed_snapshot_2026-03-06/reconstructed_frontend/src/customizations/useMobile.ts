/**
 * Mobile Detection Hook
 * =====================
 * Provides responsive breakpoint detection for mobile-specific UI adaptations.
 */

import { useState, useEffect } from "react";

const MOBILE_BREAKPOINT = 768;

export function useIsMobile(): boolean {
    const [isMobile, setIsMobile] = useState(() => {
        if (typeof window !== "undefined") {
            return window.innerWidth < MOBILE_BREAKPOINT;
        }
        return false;
    });

    useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth < MOBILE_BREAKPOINT);
        };

        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    return isMobile;
}

/**
 * Source Abbreviations for Mobile
 * Maps full source names to shorter abbreviations for small screens.
 */
export const CATEGORY_ABBREVIATIONS: Record<string, string> = {
    // CPR Parts
    "CPR Part 1 - Overriding Objective": "CPR 1",
    "CPR Part 2 - Application and Interpretation": "CPR 2",
    "CPR Part 3 - Case Management": "CPR 3",
    "CPR Part 6 - Service": "CPR 6",
    "CPR Part 7 - Starting Proceedings": "CPR 7",
    "CPR Part 12 - Default Judgment": "CPR 12",
    "CPR Part 14 - Admissions": "CPR 14",
    "CPR Part 15 - Defence and Reply": "CPR 15",
    "CPR Part 16 - Statements of Case": "CPR 16",
    "CPR Part 17 - Amendments": "CPR 17",
    "CPR Part 18 - Further Information": "CPR 18",
    "CPR Part 19 - Parties": "CPR 19",
    "CPR Part 20 - Counterclaims": "CPR 20",
    "CPR Part 21 - Protected Parties": "CPR 21",
    "CPR Part 22 - Statements of Truth": "CPR 22",
    "CPR Part 23 - Applications": "CPR 23",
    "CPR Part 24 - Summary Judgment": "CPR 24",
    "CPR Part 25 - Interim Remedies": "CPR 25",
    "CPR Part 26 - Case Management": "CPR 26",
    "CPR Part 27 - Small Claims Track": "CPR 27",
    "CPR Part 28 - Fast Track": "CPR 28",
    "CPR Part 29 - Multi-Track": "CPR 29",
    "CPR Part 31 - Disclosure": "CPR 31",
    "CPR Part 32 - Evidence": "CPR 32",
    "CPR Part 33 - Miscellaneous Evidence": "CPR 33",
    "CPR Part 34 - Depositions": "CPR 34",
    "CPR Part 35 - Experts": "CPR 35",
    "CPR Part 36 - Offers to Settle": "CPR 36",
    "CPR Part 38 - Discontinuance": "CPR 38",
    "CPR Part 39 - Hearings": "CPR 39",
    "CPR Part 40 - Judgments and Orders": "CPR 40",
    "CPR Part 44 - Costs": "CPR 44",
    "CPR Part 45 - Fixed Costs": "CPR 45",
    "CPR Part 46 - Costs Assessment": "CPR 46",
    "CPR Part 52 - Appeals": "CPR 52",
    // Practice Directions
    "Practice Direction 31B - Disclosure of Electronic Documents": "PD 31B",
    "Practice Direction 35 - Experts and Assessors": "PD 35",
    "Practice Direction 57AC - Trial Witness Statements": "PD 57AC",
    // Court Guides (with new display names)
    "Commercial Court Guide": "Commercial",
    "Circuit Commercial Court Guide": "Circuit Comm",
    "Technology and Construction Court Guide": "TCC",
    "King's Bench Division Guide": "KB",
    "Chancery Guide": "Chancery",
    "Patents Court Guide": "Patents",
    "CPR & Practice Directions": "CPR & PD",
    // Other documents
    "Disclosure Pilot Scheme": "DPS",
    "Pre-Action Protocols": "PAP",
    "All Sources": "All"
};

/**
 * Get abbreviated source name for mobile display
 */
export function getAbbreviatedCategory(fullName: string): string {
    // Check exact match first
    if (CATEGORY_ABBREVIATIONS[fullName]) {
        return CATEGORY_ABBREVIATIONS[fullName];
    }

    // Try to extract CPR Part number
    const cprMatch = fullName.match(/CPR\s*Part\s*(\d+)/i);
    if (cprMatch) {
        return `CPR ${cprMatch[1]}`;
    }

    // Try to extract Practice Direction number
    const pdMatch = fullName.match(/Practice\s*Direction\s*(\d+\w*)/i);
    if (pdMatch) {
        return `PD ${pdMatch[1]}`;
    }

    // Fallback: truncate to first 8 chars
    if (fullName.length > 10) {
        return fullName.substring(0, 8) + "…";
    }

    return fullName;
}

/**
 * Search Depth Abbreviations for Mobile
 */
export const DEPTH_OPTIONS = {
    minimal: { full: "Quick", short: "Q", icon: "⚡" },
    low: { full: "Standard", short: "S", icon: "📋" },
    medium: { full: "Thorough", short: "T", icon: "🔍" }
};

export function getDepthLabel(key: string, isMobile: boolean): string {
    const option = DEPTH_OPTIONS[key as keyof typeof DEPTH_OPTIONS];
    if (!option) return key;
    return isMobile ? option.short : option.full;
}

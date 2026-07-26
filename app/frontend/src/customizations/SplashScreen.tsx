import React, { useState, useEffect, useCallback } from "react";
import styles from "./SplashScreen.module.css";

// CUSTOM: Default logo using the same SVG as the main app
const DefaultLogo = () => (
    <svg className={styles.logoIcon} viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path
            d="M7.4 12.8a1.04 1.04 0 0 0 1.59-.51l.45-1.37a2.34 2.34 0 0 1 1.47-1.48l1.4-.45A1.04 1.04 0 0 0 12.25 7l-1.37-.45A2.34 2.34 0 0 1 9.4 5.08L8.95 3.7a1.03 1.03 0 0 0-.82-.68 1.04 1.04 0 0 0-1.15.7l-.46 1.4a2.34 2.34 0 0 1-1.44 1.45L3.7 7a1.04 1.04 0 0 0 .02 1.97l1.37.45a2.33 2.33 0 0 1 1.48 1.48l.46 1.4c.07.2.2.37.38.5Zm6.14 4.05a.8.8 0 0 0 1.22-.4l.25-.76a1.09 1.09 0 0 1 .68-.68l.77-.25a.8.8 0 0 0-.02-1.52l-.77-.25a1.08 1.08 0 0 1-.68-.68l-.25-.77a.8.8 0 0 0-1.52.01l-.24.76a1.1 1.1 0 0 1-.67.68l-.77.25a.8.8 0 0 0 0 1.52l.77.25a1.09 1.09 0 0 1 .68.68l.25.77c.06.16.16.3.3.4Z"
            fill="currentColor"
        />
    </svg>
);

interface SplashScreenProps {
    title?: string;
    subtitle?: string;
    duration?: number;
    onComplete?: () => void;
    logo?: React.ReactNode;
    skipOnRevisit?: boolean;
    storageKey?: string;
}

/**
 * SplashScreen — animation-only intro overlay.
 *
 * With Easy Auth (Azure Built-in Authentication) enabled, unauthenticated
 * users are redirected to the Microsoft login page by the platform *before*
 * the app loads. This splash screen is therefore purely cosmetic: it shows
 * a brief branded animation and then morphs into the header bar.
 */
export const SplashScreen: React.FC<SplashScreenProps> = ({
    title = "Civil Procedure Copilot",
    subtitle = "Search the Civil Procedure Rules, Practice Directions, and Court Guides",
    duration = 1800,
    onComplete,
    logo,
    skipOnRevisit = true,
    storageKey = "cpr-splash-shown"
}) => {
    const [phase, setPhase] = useState<"appearing" | "visible" | "morphing" | "done">("appearing");

    const prefersReducedMotion = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const wasShownThisSession = skipOnRevisit && typeof sessionStorage !== "undefined" && sessionStorage.getItem(storageKey) === "true";

    const finishSplash = useCallback(() => {
        if (skipOnRevisit && typeof sessionStorage !== "undefined") {
            sessionStorage.setItem(storageKey, "true");
        }
        setPhase("morphing");
        const morphDuration = prefersReducedMotion ? 0 : 700;
        setTimeout(() => {
            setPhase("done");
            onComplete?.();
        }, morphDuration);
    }, [onComplete, prefersReducedMotion, skipOnRevisit, storageKey]);

    const handleDismiss = useCallback(() => {
        if (phase === "morphing" || phase === "done") return;
        finishSplash();
    }, [phase, finishSplash]);

    const handleKeyDown = useCallback(
        (e: KeyboardEvent) => {
            if (e.key === "Escape" || e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                handleDismiss();
            }
        },
        [handleDismiss]
    );

    useEffect(() => {
        if (wasShownThisSession) {
            setPhase("done");
            onComplete?.();
            return;
        }
        if (prefersReducedMotion) {
            if (skipOnRevisit && typeof sessionStorage !== "undefined") sessionStorage.setItem(storageKey, "true");
            setPhase("done");
            onComplete?.();
            return;
        }
        const t1 = setTimeout(() => setPhase("visible"), 800);
        const t2 = setTimeout(() => handleDismiss(), duration);
        document.addEventListener("keydown", handleKeyDown);
        return () => {
            clearTimeout(t1);
            clearTimeout(t2);
            document.removeEventListener("keydown", handleKeyDown);
        };
    }, [wasShownThisSession, prefersReducedMotion, duration, handleDismiss, handleKeyDown, onComplete, skipOnRevisit, storageKey]);

    if (phase === "done") return null;

    const isMorphing = phase === "morphing";

    return (
        <div
            className={`${styles.splashOverlay} ${isMorphing ? styles.morphing : styles.fadeIn}`}
            onClick={handleDismiss}
            role="dialog"
            aria-modal="true"
            aria-label={`${title} - ${subtitle}`}
            tabIndex={0}
        >
            <div className={`${styles.splashContent} ${isMorphing ? styles.contentMorphing : ""}`}>
                <div className={`${styles.logoContainer} ${isMorphing ? styles.logoMorphing : ""}`}>{logo || <DefaultLogo />}</div>
                <h1 className={`${styles.title} ${isMorphing ? styles.titleMorphing : ""}`}>{title}</h1>
                <p className={`${styles.subtitle} ${isMorphing ? styles.subtitleMorphing : ""}`}>{subtitle}</p>
            </div>
        </div>
    );
};

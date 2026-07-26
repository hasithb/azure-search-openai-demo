import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SplashScreen } from "../SplashScreen";

describe("SplashScreen", () => {
    beforeEach(() => {
        // Clear sessionStorage before each test
        sessionStorage.clear();

        // Mock matchMedia for reduced motion tests
        Object.defineProperty(window, "matchMedia", {
            writable: true,
            value: vi.fn().mockImplementation(query => ({
                matches: false,
                media: query,
                onchange: null,
                addListener: vi.fn(),
                removeListener: vi.fn(),
                addEventListener: vi.fn(),
                removeEventListener: vi.fn(),
                dispatchEvent: vi.fn()
            }))
        });
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    it("renders with default props", () => {
        render(<SplashScreen />);

        expect(screen.getByText("Civil Procedure Copilot")).toBeInTheDocument();
        expect(screen.getByText(/Search the Civil Procedure Rules/)).toBeInTheDocument();
    });

    it("renders with custom title and subtitle", () => {
        render(<SplashScreen title="Custom Title" subtitle="Custom Subtitle" />);

        expect(screen.getByText("Custom Title")).toBeInTheDocument();
        expect(screen.getByText("Custom Subtitle")).toBeInTheDocument();
    });

    it("dismisses on click", async () => {
        const onComplete = vi.fn();
        render(<SplashScreen onComplete={onComplete} duration={10000} />);

        const overlay = screen.getByRole("dialog");
        fireEvent.click(overlay);

        // Wait for fade-out animation
        await waitFor(
            () => {
                expect(onComplete).toHaveBeenCalled();
            },
            { timeout: 1000 }
        );
    });

    it("dismisses on Escape key", async () => {
        const onComplete = vi.fn();
        render(<SplashScreen onComplete={onComplete} duration={10000} />);

        fireEvent.keyDown(document, { key: "Escape" });

        await waitFor(
            () => {
                expect(onComplete).toHaveBeenCalled();
            },
            { timeout: 1000 }
        );
    });

    it("dismisses on Enter key", async () => {
        const onComplete = vi.fn();
        render(<SplashScreen onComplete={onComplete} duration={10000} />);

        fireEvent.keyDown(document, { key: "Enter" });

        await waitFor(
            () => {
                expect(onComplete).toHaveBeenCalled();
            },
            { timeout: 1000 }
        );
    });

    it("dismisses on Space key", async () => {
        const onComplete = vi.fn();
        render(<SplashScreen onComplete={onComplete} duration={10000} />);

        fireEvent.keyDown(document, { key: " " });

        await waitFor(
            () => {
                expect(onComplete).toHaveBeenCalled();
            },
            { timeout: 1000 }
        );
    });

    it("auto-dismisses after duration", async () => {
        const onComplete = vi.fn();
        render(<SplashScreen onComplete={onComplete} duration={100} />);

        await waitFor(
            () => {
                expect(onComplete).toHaveBeenCalled();
            },
            { timeout: 1000 }
        );
    });

    it("stores visit in sessionStorage when dismissed", async () => {
        render(<SplashScreen duration={100} storageKey="test-splash" />);

        await waitFor(
            () => {
                expect(sessionStorage.getItem("test-splash")).toBe("true");
            },
            { timeout: 1000 }
        );
    });

    it("skips splash if already shown this session", () => {
        sessionStorage.setItem("test-splash", "true");

        const onComplete = vi.fn();
        render(<SplashScreen onComplete={onComplete} storageKey="test-splash" skipOnRevisit={true} />);

        // Should call onComplete immediately and not render
        expect(onComplete).toHaveBeenCalled();
        expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("does not skip if skipOnRevisit is false", () => {
        sessionStorage.setItem("test-splash", "true");

        render(<SplashScreen storageKey="test-splash" skipOnRevisit={false} />);

        expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    it("skips immediately for users preferring reduced motion", () => {
        // Mock prefers-reduced-motion
        Object.defineProperty(window, "matchMedia", {
            writable: true,
            value: vi.fn().mockImplementation(query => ({
                matches: query === "(prefers-reduced-motion: reduce)",
                media: query,
                onchange: null,
                addListener: vi.fn(),
                removeListener: vi.fn(),
                addEventListener: vi.fn(),
                removeEventListener: vi.fn(),
                dispatchEvent: vi.fn()
            }))
        });

        const onComplete = vi.fn();
        render(<SplashScreen onComplete={onComplete} />);

        expect(onComplete).toHaveBeenCalled();
    });

    it("has correct accessibility attributes", () => {
        render(<SplashScreen />);

        const overlay = screen.getByRole("dialog");
        expect(overlay).toHaveAttribute("aria-modal", "true");
        expect(overlay).toHaveAttribute("aria-label");
        expect(overlay).toHaveAttribute("tabIndex", "0");
    });

    it("renders custom logo when provided", () => {
        const CustomLogo = () => <div data-testid="custom-logo">Custom Logo</div>;
        render(<SplashScreen logo={<CustomLogo />} />);

        expect(screen.getByTestId("custom-logo")).toBeInTheDocument();
    });

    it("ignores clicks after morphing starts", async () => {
        const onComplete = vi.fn();
        render(<SplashScreen duration={10000} onComplete={onComplete} skipOnRevisit={false} />);

        // Get the dialog - it should be visible initially
        const overlay = screen.getByRole("dialog");
        expect(overlay).toBeInTheDocument();

        // Click to start morphing
        fireEvent.click(overlay);

        // Click again - should be ignored since morphing started
        fireEvent.click(overlay);
        fireEvent.click(overlay);

        // After morph animation completes, dialog should be gone
        // and onComplete should only be called once
        await waitFor(
            () => {
                expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
            },
            { timeout: 1000 }
        );

        // onComplete should only be called once despite multiple clicks
        expect(onComplete).toHaveBeenCalledTimes(1);
    });
});

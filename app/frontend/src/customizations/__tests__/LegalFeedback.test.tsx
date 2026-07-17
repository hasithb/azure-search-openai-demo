import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { LegalFeedback } from "../LegalFeedback";

describe("LegalFeedback", () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockResolvedValue({ ok: true });
        // @ts-expect-error - override for tests
        global.fetch = fetchMock;
    });

    afterEach(() => {
        vi.resetAllMocks();
    });

    it("submits helpful feedback immediately without sharing context", async () => {
        render(
            <LegalFeedback messageId="m1" userPrompt="What is CPR?" aiResponse="Answer" conversationHistory={[{ role: "user", content: "Hi" }]} thoughts={[]} />
        );

        fireEvent.click(screen.getByRole("button", { name: "Accurate & Helpful" }));

        await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
        const [, init] = fetchMock.mock.calls[0];
        const body = JSON.parse(String(init.body));
        expect(body).toMatchObject({
            message_id: "m1",
            rating: "helpful",
            issues: [],
            comment: "",
            context_shared: false
        });
        expect(body.user_prompt).toBeUndefined();
    });

    it("requires context choice before submitting unhelpful feedback", async () => {
        render(
            <LegalFeedback
                messageId="m2"
                userPrompt="Question"
                aiResponse="Answer"
                conversationHistory={[{ role: "user", content: "Question" }]}
                thoughts={[]}
            />
        );

        fireEvent.click(screen.getByRole("button", { name: "Report Issue" }));

        const submit = await screen.findByRole("button", { name: /Submit Report/i });
        expect(submit).toBeDisabled();

        // Choose "No" (no context)
        fireEvent.click(screen.getByText(/No, submit feedback without my prompt/i));
        expect(submit).not.toBeDisabled();

        fireEvent.click(submit);

        await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
        const [, init] = fetchMock.mock.calls[0];
        const body = JSON.parse(String(init.body));
        expect(body).toMatchObject({
            message_id: "m2",
            rating: "unhelpful",
            context_shared: false
        });
        expect(body.user_prompt).toBeUndefined();
    });
});

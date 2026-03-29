# Deployed frontend comparison report

Generated on 2026-03-06.

## Recoverability

- Deployed frontend assets were recovered from production source maps.
- Recovered source location: `archive/deployed_snapshot_2026-03-06/reconstructed_frontend`
- Recoverable surface: frontend only.
- Backend server source is not downloadable from the app URL unless separately exposed by the deployment.

## Summary

- Same local files: 21
- Changed local files: 32
- Missing locally: 1

## Changed files

- app/frontend/src/assets/applogo.svg
- app/frontend/src/api/api.ts
- app/frontend/src/api/models.ts
- app/frontend/src/customizations/citationSanitizer.ts
- app/frontend/src/customizations/LegalFeedback.tsx
- app/frontend/src/customizations/HelpAboutPanel.tsx
- app/frontend/src/components/Answer/AnswerParser.tsx
- app/frontend/src/components/Answer/Answer.tsx
- app/frontend/src/components/Answer/AnswerLoading.tsx
- app/frontend/src/components/Answer/AnswerError.tsx
- app/frontend/src/components/Answer/SpeechOutputBrowser.tsx
- app/frontend/src/components/QuestionInput/QuestionInput.tsx
- app/frontend/src/components/SupportingContent/SupportingContentParser.ts
- app/frontend/src/components/SupportingContent/SupportingContent.tsx
- app/frontend/src/components/AnalysisPanel/TokenUsageGraph.tsx
- app/frontend/src/components/AnalysisPanel/AgentPlan.tsx
- app/frontend/src/components/AnalysisPanel/ThoughtProcess.tsx
- app/frontend/src/components/MarkdownViewer/MarkdownViewer.tsx
- app/frontend/src/components/AnalysisPanel/AnalysisPanel.tsx
- app/frontend/src/components/HistoryItem/HistoryItem.tsx
- app/frontend/src/components/HistoryPanel/HistoryPanel.tsx
- app/frontend/src/components/UploadFile/UploadFile.tsx
- app/frontend/src/components/TokenClaimsDisplay/TokenClaimsDisplay.tsx
- app/frontend/src/i18n/LanguagePicker.tsx
- app/frontend/src/components/HelpCallout/HelpCallout.tsx
- app/frontend/src/components/VectorSettings/VectorSettings.tsx
- app/frontend/src/components/Settings/Settings.tsx
- app/frontend/src/pages/chat/Chat.tsx
- app/frontend/src/components/LoginButton/LoginButton.tsx
- app/frontend/src/pages/layout/Layout.tsx
- app/frontend/src/layoutWrapper.tsx
- app/frontend/src/index.tsx

## Missing locally

- app/frontend/src/components/GPT4VSettings/GPT4VSettings.tsx

## Notes

To preserve upstream updates, the safest alignment strategy is to compare the recovered deployed frontend files against the current local files and port only UI/feature deltas rather than overwriting local sources wholesale.

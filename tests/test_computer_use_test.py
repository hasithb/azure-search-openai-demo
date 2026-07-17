from scripts.computer_use_test import Scenario, build_scenario_task, resolve_model


def test_resolve_model_prefers_non_mini_deployment(monkeypatch):
    monkeypatch.delenv("COMPUTER_USE_MODEL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-5.4-mini")
    monkeypatch.setenv("AZURE_OPENAI_CHATGPT_MODEL", "gpt-5.4")
    monkeypatch.setenv("AZURE_OPENAI_EVAL_DEPLOYMENT", "gpt-5.4")
    monkeypatch.setenv("AZURE_OPENAI_EVAL_MODEL", "gpt-5.4")
    monkeypatch.delenv("AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_SEARCHAGENT_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_GPT4V_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_GPT4V_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_KNOWLEDGEBASE_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_KNOWLEDGEBASE_MODEL", raising=False)

    assert resolve_model() == "gpt-5.4"


def test_resolve_model_falls_back_to_mini_when_only_capable_option(monkeypatch):
    monkeypatch.delenv("COMPUTER_USE_MODEL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-5.4-mini")
    monkeypatch.setenv("AZURE_OPENAI_CHATGPT_MODEL", "gpt-5.4")
    monkeypatch.delenv("AZURE_OPENAI_EVAL_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_EVAL_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_SEARCHAGENT_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_GPT4V_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_GPT4V_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_KNOWLEDGEBASE_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_KNOWLEDGEBASE_MODEL", raising=False)

    assert resolve_model() == "gpt-5.4-mini"


def test_build_scenario_task_can_assume_preconfigured_source_filter():
    scenario = Scenario(
        key="demo-scenario",
        title="Demo Scenario",
        source_filter="Patents Court Guide",
        questions=["What does the guide say about trial windows?"],
    )

    task = build_scenario_task(scenario, source_filter_preconfigured=True)

    assert "already set exactly as 'Patents Court Guide'" in task
    assert "Select the source filter exactly as 'Patents Court Guide'" not in task
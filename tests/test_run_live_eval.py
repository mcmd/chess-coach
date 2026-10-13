import json

import pytest

from scripts.run_live_eval import _PlayerRelay, _Script, _agent_model, _resolved_model, _reviewer_event


SCRIPT = """### S0 — Load
**Send:**
```text
load game
```
### S1 — Candidate
**Send:**
```text
candidate
```
**Allowed branch S1a:** explain once
```text
reason
```
### S2 — Detour
**Send:**
```text
detour
```
### S3 — Stop
No more answers.
"""


class Session:
    def __init__(self, failure=None):
        self.failure = failure
        self.submissions = []

    def send(self, message):
        self.submissions.append(message)

    def until_ready(self, timeout):
        if self.failure:
            raise self.failure
        return {"coach_visible": "A complete coach question", "errors": []}


def test_model_cannot_skip_a_student_turn_or_finish_early(tmp_path):
    session = Session()
    relay = _PlayerRelay(session, _Script(SCRIPT), tmp_path, 60, 30)
    result = relay.exchange("S2")
    assert result["execution_status"] == "STOPPED_UNEXPECTED"
    assert result["last_sent_step"] is None
    assert "incompatible" in result["error"]
    assert session.submissions == []
    assert relay.exchange("S0")["execution_status"] == "STOPPED_UNEXPECTED"


def test_explanation_branch_cannot_repeat_or_reopen_a_completed_turn(tmp_path):
    relay = _PlayerRelay(Session(), _Script(SCRIPT), tmp_path, 60, 30)
    for step in ("S0", "S1", "S1a"):
        assert relay.exchange(step)["execution_status"] == "RUNNING"
    result = relay.exchange("S1a")
    assert result["execution_status"] == "STOPPED_UNEXPECTED"
    assert result["last_sent_step"] == "S1a"
    assert [entry["step"] for entry in relay.sent_steps] == ["S0", "S1", "S1a"]


def test_timeout_preserves_pending_turn_and_does_not_retry(tmp_path):
    relay = _PlayerRelay(Session(TimeoutError("model deadline")), _Script(SCRIPT), tmp_path, 60, 30)
    result = relay.exchange("S0")
    assert result["execution_status"] == "TIMED_OUT"
    assert relay.busy is True
    assert result["last_sent_step"] == "S0"
    assert relay.exchange("S0")["execution_status"] == "TIMED_OUT"
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert len(execution["sent_steps"]) == 1
    assert "completed_at" not in execution["sent_steps"][0]
    assert execution["error"] == "model deadline"


def test_player_that_sent_every_step_completes_without_the_stop_step(tmp_path):
    # Runs 1l_zs5z1 and e054prsf: the player sent S0-S29, reported SCRIPT_COMPLETE, but
    # never sent the stop step, so the run was recorded as STOPPED_UNEXPECTED.
    (tmp_path / "done").mkdir()
    (tmp_path / "early").mkdir()
    done = _PlayerRelay(Session(), _Script(SCRIPT), tmp_path / "done", 60, 30)
    for step in ("S0", "S1", "S2"):
        done.exchange(step)
    done.finish("SCRIPT_COMPLETE", "all steps sent")
    assert done.result()["execution_status"] == "SCRIPT_COMPLETE"

    early = _PlayerRelay(Session(), _Script(SCRIPT), tmp_path / "early", 60, 30)
    for step in ("S0", "S1"):
        early.exchange(step)
    early.finish("SCRIPT_COMPLETE", "claims done")
    assert early.result()["execution_status"] == "STOPPED_UNEXPECTED"
    assert early.result()["error"] == "claims done"


def test_role_model_comes_from_the_agent_definition_and_missing_model_fails(tmp_path):
    agents = tmp_path / ".omp/agents"
    agents.mkdir(parents=True)
    (agents / "player.md").write_text('---\nname: player\nmodel: "openai-codex/gpt-6-luna:high"\n---\nBody\n')
    (agents / "bare.md").write_text("---\nname: bare\n---\nBody\n")

    assert _agent_model(tmp_path, "player") == "openai-codex/gpt-6-luna:high"
    with pytest.raises(ValueError):
        _agent_model(tmp_path, "bare")


def test_resolved_model_reports_the_last_model_the_subagent_ran_on(tmp_path):
    session = tmp_path / "cwd-slug" / "parent-session"
    session.mkdir(parents=True)
    events = [{"type": "session_init"},
              {"type": "model_change", "model": "openai-codex/gpt-6-luna"},
              {"type": "thinking_level_change", "thinkingLevel": "high"},
              {"type": "model_change", "model": "anthropic/claude-opus-5-5"}]
    (session / "Player-1.jsonl").write_text("\n".join(json.dumps(e) for e in events))

    assert _resolved_model("Player-1", tmp_path) == "anthropic/claude-opus-5-5:high"
    assert _resolved_model("Missing", tmp_path) is None


def test_reviewer_events_keep_answers_and_calls_but_not_raw_model_internals():
    request = {"id": 1, "kind": "model_request", "turn": 1,
               "data": {"payload": {"contents": ["history"], "systemInstruction": "prompt"}}}
    response = {"id": 2, "kind": "model_response", "turn": 1, "data": {"response": {
        "candidates": [{"content": {"parts": [
            {"text": "private thought", "thought": True},
            {"text": "s Thinking Process:\n\n1. **Analyze** best is Rd4"},
            {"text": "Which move defends g4?"},
            {"function_call": {"name": "evaluate_explanation", "args": {"fen": "x"}},
             "thought_signature": "c2lnbmF0dXJl"}]}}],
        "usage_metadata": {"thoughts_token_count": 300}}}}
    tool = {"id": 3, "kind": "tool_result", "turn": 1, "data": {"name": "t", "result": {"ok": True}}}

    assert "history" not in json.dumps(_reviewer_event(request))
    reduced = _reviewer_event(response)
    assert reduced["id"] == 2 and reduced["data"]["response"] == {
        "text": "Which move defends g4?",
        "function_calls": [{"name": "evaluate_explanation", "args": {"fen": "x"}}],
        "leaked_reasoning_parts_withheld": 1}
    assert _reviewer_event(tool) == tool

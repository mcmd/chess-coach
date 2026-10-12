import json

from scripts.run_live_eval import _PlayerRelay, _Script, _spawn_role


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


class _Handle:
    id = "agent-1"
    handle = "agent://agent-1"


def test_unpinned_spawn_records_default_model_instead_of_selector():
    calls = []

    def spawn(prompt, *, label=None, schema=None):
        calls.append({"label": label})
        return _Handle()

    _, identity = _spawn_role(spawn, "prompt", "@smol", label="Player")
    assert calls == [{"label": "Player"}]
    assert identity["model_pinned"] is False
    assert identity["requested_model"] == "@smol"
    assert identity["resolved_model"] != "@smol"


def test_pinned_spawn_passes_selector():
    calls = []

    def spawn(prompt, *, model=None, label=None):
        calls.append({"model": model, "label": label})
        return _Handle()

    _, identity = _spawn_role(spawn, "prompt", "@default", label="Reviewer")
    assert calls == [{"model": "@default", "label": "Reviewer"}]
    assert identity["model_pinned"] is True and identity["resolved_model"] == "@default"

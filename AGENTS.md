# Agent Instructions

## Commit Attribution

When Codex creates a commit in this repository or the nested `design` repository,
append this trailer to the commit message:

`Co-Authored-By: OpenAI Codex <noreply@openai.com>`

Keep the configured human Git identity as the commit author.

## Live Coaching Evaluation

After making changes in this repository, follow `tests/live_coach_run.md` for the
two-pass live eval, in addition to relevant deterministic checks:

1. In omp Eval, import `scripts.run_live_eval.run_eval` and call
   `await run_eval(agent_fn=agent, tool_fn=tool)` from the repository root.
   The callable reads `tests/live_coach_agents.json`, starts a disposable real
   coach, and spawns the separate configured player with only student-visible I/O.
2. The callable shuts down the coach, saves structured evidence, runs
   `scripts/check_live_eval.py`, then spawns a fresh configured reviewer.
   Follow `tests/live_coach_run.md` for artifacts, limits, and subagent invocation.

Use a fresh isolated profile and the real CLI/model/engines; do not reuse the
user's active `mcmd` session or silently fall back to mocks. Follow the player's
gates/stop rule and the run protocol's recording requirements/timeouts.

Report player execution status separately from reviewer verdicts, with failures,
blockers, unobserved checks, evidence paths, requested/resolved models, and agent
IDs. Do not do both roles in the parent. If the harness cannot pin models, run
unpinned and report `model_pinned: false` rather than claiming the configured
selectors. If a
known API quota prevents a new live trial, report it without retrying; independent
archived-run checks are allowed but must not be described as a new live run.
Do not claim a pass from plausible prose or pytest alone. This is a harness-agent
workflow requirement, not a background watcher or an unattended CI job.


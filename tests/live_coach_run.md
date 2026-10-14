# Live coaching evaluation: run protocol

This is the operator/runner protocol, not a student prompt or grading rubric.
`scripts/run_live_eval.py::run_eval` implements this harness-orchestrated two-pass
workflow with separate player/reviewer agents and a structured event recorder.
It is callable inside omp Eval, not an independent shell runner.

## Entry point

From the repository root, in an omp Eval cell:

```python
from scripts.run_live_eval import run_eval
result = await run_eval(agent_fn=agent, tool_fn=tool)
display(result)
```

Alternatively, `%load scripts/run_live_eval.py` in one cell, then
`await run_eval()` in the next; loading injects the harness interfaces.
The callable returns the run directory, execution/review statuses, artifact
paths, agent IDs, requested selectors, and reported model IDs.

Defaults: source profile `mcmd`, startup deadline 90 seconds, loading deadline
600 seconds, later-turn deadline 300 seconds, reviewer deadline 3600 seconds (the
reviewer runs depth-30 engine checks, minutes each).
These are keyword arguments (`source_profile`, `startup_timeout`, `load_timeout`,
`turn_timeout`, `reviewer_timeout`); `root` can supply an explicit repository path.
`config_overrides` sets per-run profile settings without changing the source profile
(e.g. `{"router_model": "gemini-3.8-flash"}` for a router run); the manifest records it.
`await review_run(run_dir, agent_fn=agent)` re-runs pass 2 (checker and a fresh reviewer)
on a frozen run directory whose review failed; the player's evidence is not touched.

Any omp development agent with Eval access can invoke the same callable.
To delegate the whole eval, spawn a runner subagent instructed to call it and
return the artifact paths/statuses, without retries or substituting either role.
The runner will spawn its own separate player and reviewer, each on the model its
project agent definition pins, independent of the runner's own model. There is no
background watcher; `AGENTS.md`
requires the development assistant to invoke the flow after changes.


## Role-specific inputs

| Role | Receives | Does not receive |
|---|---|---|
| Pass-1 player | `tests/2dtDqQgR_script.md` + student-visible conversation | Reviewer rubric, hidden coach prompts/tool results, judge findings |
| Runner/operator | This protocol + setup and execution status | Authority to invent student answers or grade coaching quality during play |
| Pass-2 reviewer | `tests/2dtDqQgR_review.md` + matching player script + frozen run evidence + deterministic checks | Ability to influence or repair the completed conversation |
| Human reviewer | Full conversation, evidence, automated findings | A requirement to trust the model's verdict |

The player can be a cheaper model, but fixed script messages need no model to
generate them. A model is only needed to match semantic gates/allowed branches.
Keep the coach's configured model and real tools unchanged. The runner may
select messages on the player's behalf; it must preserve the same role boundary.

## Harness orchestration and models

The parent coding assistant is the runner/operator. `run_eval` reads
`tests/live_coach_agents.json`, which maps each role to a project agent, and
spawns that agent by name (`agent(prompt, agent="live-coach-player")`). omp
resolves the model from the agent definition's `model:` frontmatter.

- `.omp/agents/live-coach-player.md` and `.omp/agents/live-coach-reviewer.md`
  both pin `openai-codex/gpt-6-luna:high`. These are harness models, not Chess
  Coach/Gemini model settings. The reviewer is not on an Anthropic model: two of
  its Opus reviews were refused (`reasoning_extraction`), and a same-model review is
  less independent, so report that limitation with the verdict.
- Use explicit `provider/model[:level]` selectors. In a subagent, `@default`
  follows the calling session's live model, not `modelRoles.default`, so it does
  not pin anything. Do not change global harness settings or the coach's
  configured model to route these two roles.
- Each run snapshots both agent definitions and records, per role, the agent ID,
  `requested_model` (from the definition), `resolved_model` (the `model_change`
  and thinking level in the subagent's omp session file, or `unavailable`), and
  the agent's self-reported model ID. Report `resolved_model`; never collapse both
  roles into the parent.

For a live run, use this true dependency order:

1. The parent prepares the disposable coach session and recording.
2. Spawn a fresh player agent with the player model and only the player script
   plus student-visible conversation. The parent handles coach I/O, trace
   capture, deadlines, and shutdown. Relay complete visible replies to the
   player; send only its selected scripted messages to the coach. Do not expose
   hidden tool results or reviewer criteria to help it choose answers.
3. The player reports execution facts only. The parent closes the coach and
   freezes the evidence, including partial/error runs.
4. After deterministic checks, spawn a different, fresh reviewer agent with the
   reviewer model, full frozen evidence, matching script, and reviewer rubric.
   Do not pass it the player's private reasoning or the parent's proposed grade.
5. Preserve the reviewer's findings as a separate artifact for human review.
   The reviewer never steers the player or interacts with the finished coach.

Use separate agents even if both selectors resolve to the same model; disclose
that fact rather than claiming model diversity. Never run the live player and
its reviewer concurrently: the reviewer depends on frozen evidence. Independent
archived-run checks may run concurrently, but label them replay checks, not a new
live trial. Known quota blockers should be reported without retrying the coach.

Role visibility is a harness prompting/I/O boundary, not a filesystem security
sandbox. The parent must not hand the player the rubric or unfiltered trace.

## Setup

1. Create a unique disposable profile inside the configured `profiles_dir`.
   Copy only `profiles/mcmd/config.json`. Do not copy conversation, lesson state,
   dossier, or CLI history. Do not modify or stop the user's active session.
   `--wipe_memory` alone does not reset all persisted state.
2. Prerequisites: English, Elo 1200, Lichess handle `mcmd`, self-assessment enabled.
   Record incompatible settings rather than silently changing user configuration.
3. Record run ID, scenario ID and exact copies of the player script/reviewer
   rubric used, code version, timestamp, model/engine settings, and available
   game PGN/network payloads or hashes. Never capture API keys.
4. The callable launches the instrumented session from the repository root:
   `.venv/bin/python scripts/live_coach_session.py --events <run-dir>/events.jsonl --event-fd <fd> --profile=<fresh-profile> --trace`.
   It runs real `coach.main()` with a newline stdin adapter, retaining the actual
   model, commands, tools, engines, renderer, and persistence. Terminal editing,
   interactive prompt-toolkit behavior, and terminal-specific styling are not
   evaluated by this adapter.

## Pass 1: execute and record, do not judge

- Wait for a new flushed `ready` event after each completed response. Startup
  readiness is not turn completion. Do not wait for the long-lived process to
  exit as a proxy for a reply. The recorder uses an inherited event pipe; waits
  are blocking queue reads with monotonic deadlines, not spinner/output polling.
- Allow 10 minutes for loading/review and 5 minutes for later turns. Timestamp
  submissions and actual completion events; do not infer completion from a
  stale `You:` prompt or the appearance of a trace line.
- Keep original student-visible output, including diagrams and warning messages.
  Do not summarize, correct, or filter out coach errors in the recording.
- Log each sent step/branch and any unexpected gate. Execution statuses are
  `SCRIPT_COMPLETE`, `STOPPED_UNEXPECTED`, `TIMED_OUT`, or
  `INFRASTRUCTURE_BLOCKED`, not coaching PASS/FAIL judgments.
- Do not silently retry. Record any explicitly requested retry as a new trial.
- On normal completion or an incompatible gate, send `exit`. On a busy-turn
  timeout, interrupt the disposable session so it can save history; verify
  shutdown and record any forced termination or missing final artifacts.

## Evidence bundle

Keep each recording in its unique disposable profile directory:

| Artifact | Contents |
|---|---|
| `events.jsonl` | Original flushed events with IDs, UTC timestamps, and turn IDs |
| `review-events.jsonl` | Reviewer copy of the events: same IDs; raw model requests withheld; model responses reduced to answer text and function calls |
| `terminal.log` | Actual coach stdout/stderr, including trace, boards, warnings, shutdown |
| `conversation.md` | Derived complete student-visible inputs, rendered replies, and errors |
| `event-index.json` | Event IDs/kinds/turns for locating complete payloads |
| `execution.json` | Submitted step IDs/messages, timing, status, final reply/error, shutdown |
| `manifest.json` | Source revision, recording boundaries, agent IDs/selectors/reported models |
| `checks.json`, `checker.log` | Offline deterministic findings and checker execution output |
| `player-result.json` | Player's execution handoff; not a coaching grade |
| `review.json`, `report.md` | Independent model findings and readable human review |
| Scenario/config snapshots | Exact player/rubric/protocol/model routing and credential-free profile settings |

Also retain the coach's actual history, memory, lesson, and dossier files when
created. Missing evidence is explicit; `review-error.json` preserves reviewer
failures instead of inventing findings.

Structured events contain student/control inputs, logical and transformed API
requests, full exposed API responses, rendered coach replies, matching tool and
engine calls/results, actual HTTP bodies/PGNs, board diagrams, FENs, complete
review agenda, lesson metadata, diagnoses/attempts, errors, and session lifecycle.
Hooks live only in `scripts/live_coach_session.py`; production coach behavior
and callable signatures remain unchanged. The installed SDK's request boundary
is instrumented, so an SDK upgrade may require adjusting that hook.

Flush both the durable file and event pipe immediately. Partial events, errors,
deadlines, and forced shutdown remain evidence, never synthetic success.
Credentials and unavailable private model internals are excluded. An unconsumed
streaming HTTP response is marked unavailable rather than consuming its stream.
Older text-only runs have truncated traces and no structured provenance; do not
treat those archived recordings as equivalent to new event bundles.

## Freeze and hand off

Close the session before review. Preserve the original evidence unchanged;
separate derived reports from originals. The handoff contains:

- Run/scenario IDs, settings and evidence paths.
- Step IDs sent, branch use, and execution status.
- Exact final completed coach message, last pending turn if any, and timestamps.
- Recording gaps and shutdown outcome, without grading the coach.

## Pass 2: check and review

The callable runs `scripts/check_live_eval.py <run-dir>` against actual recorded
FENs, PGN, tool/engine lines, diagrams, and lesson transitions. This checker is
also runnable offline without model/API/engine calls. It checks legal moves/PVs,
diagram provenance, lesson/physical-timeline consistency, detour restoration,
duplicate diagnoses, and recording completeness. Semantic claims and numerical
interpretation remain the reviewer's job.

The physical timeline and displayed lesson board are separate contexts; the
checker must not require them to have identical FENs. Prose without an explicit,
contextualized move sequence is not silently treated as a checked continuation.
The reviewer must mark unexercised legality checks NOT OBSERVED rather than
substituting chess intuition.

Chess claims that the recorded evidence does not settle (evaluations, best or
refuting moves, lines, threats, what players at a rating play) are checked with
the local engines, never by calculating lines by hand:
`.venv/bin/python scripts/verify_chess.py --fen FEN [--move SAN ...] [--line 'SAN ...'] [--elo N]`
runs Stockfish from `config.json` at depth 30 from a cleared hash (time cap 600 s,
only a safety net) and Maia3. Do not lower the depth: shallow searches missed the
31. Kh2 refutation, which appears only at depth 16+. A result with
`"complete": false` stopped short of the requested depth and is inconclusive.
Cite the command with the reported depth and nodes.

Give a fresh reviewer agent the frozen evidence, deterministic results, and
`tests/2dtDqQgR_review.md`, using the configured reviewer model. It can also read
the player script to understand what was asked, but it never supplies messages
for the finished session. Do not substitute the parent for this separate agent;
if orchestration is unavailable, report the prerequisite as blocked.

Preserve the reviewer findings and a human-readable report with evidence links.
Allow human review of the whole conversation and disagreements with the judge.
Mark missing evidence NOT OBSERVED and timeouts BLOCKED; do not silently fill
gaps. Report comparisons only against a named, compatible saved baseline.

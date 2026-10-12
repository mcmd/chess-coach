# AI Assistant Context: Chess Coach CLI

## What this project is

A terminal-based AI chess coach. A Gemini LLM runs the **dialogue**; a Python
backend runs the **analysis** and owns all chess truth. There is no web app —
`coach.py` is the only entry point.

## Architecture (hybrid: backend-led analysis, LLM-led pedagogy)

```
coach.py             thin wiring: config -> state -> engines -> tools -> chat -> REPL
app/commands.py      slash-command / chat-input processing (CommandProcessor, testable)
app/agent_loop.py    Gemini tool-execution loop + response rendering
app/state.py         SessionState (single source of truth) + TimelineManager (board mutations)
app/engines.py       EngineManager: Stockfish subprocess + in-process Maia-3 (tiered
                     deep/scan) AND all engine policy (tiers, limits, eval source)
app/review.py        game review pipeline: per-move classification (win%-space eval swings),
                     lesson agenda selection, digest_for_llm
app/concepts.py      pure chess features: SEE-based piece safety, pawn structure, outposts,
                     king perimeter, phase, forcing moves, endgame features
app/motifs.py        tactical motif detection (fork/pin/skewer/back-rank/...) on engine PVs
app/tools.py         CoachTools: the Gemini-callable tools (analyze_position packet,
                     evaluate_student_answer, play_sparring_reply, submit_diagnosis enum, ...)
app/prompts.py       system instruction (modes, Socratic protocol) + chat session factory
app/memory.py        atomic persistence: chat history, player dossier (taxonomy aggregation)
app/render.py        board ASCII rendering bound to [[BOARD_xxxx]] tokens
app/config.py        config.json + profiles + CLI args
```

## Key invariants — do not break these

- **The LLM never mutates or computes chess state.** `SessionState.board` is the
  only truth; all mutations go through `TimelineManager`. The LLM's only state
  mutation paths are the `play_sparring_reply` / `execute_coach_move` tools.
- **Engine policy lives in `EngineManager` only.** Never call
  `engines.stockfish.analyse` from other modules; add a method instead. This
  includes Maia tiering: `maia_predict` picks the deep/scan Maia-3 model by
  `tier=`; callers never choose a model. If Maia-3 fails to load, the coach runs
  engine-only (no human-move lens) behind a warning — no silent fallback.
- **The win% stream prefers the PGN's `[%eval]`.** `review_game`'s scan uses the
  game's own deep annotations for eval/drop/slide detection when present (the ~17%
  slide baseline was calibrated on Lichess evals — `scripts/scan_eval_slides.py`);
  the local Stockfish search is the fallback and mainly supplies best-move/punisher.
  `GameReview.eval_source` records `"pgn"` vs `"engine"`; slides read off the
  fallback are flagged in the batch digest. Bump `batch.SCAN_VERSION` if scan-level
  eval/record fields change incompatibly.
- **Socratic withholding is backend-enforced**: `evaluate_student_answer` counts
  attempts per FEN and only includes the engine answer when `reveal_allowed`.
- **Verified-line moves must be named by exact SAN.** If a tool or digest line says
  `Rad8`, prose must say `Rad8`, not "rook to d8" or `Rd8`; ambiguous piece prose is
  shadow-linted at render time.
- **Student clarification questions take priority.** If the latest student turn asks
  a question, answer it from verified facts before probing, calling `submit_diagnosis`,
  or advancing to the next moment.
- **WHY probes must be backend-authored.** If `evaluate_explanation` returns
  `probe_question`, ask that question verbatim or near-verbatim and use only
  `probe_facts`; do not invent the missing detail or a new conceptual distinction.
  If the tool returns no `probe_question`, teach directly from the verified packet.
- **Grader verdicts must be narrated.** After any grading tool returns, state the
  verdict plainly before probing, closing a moment, or advancing. Do not bury or skip
  `not_a_real_threat`, `best`, `wrong`, `illegal`, or similar verdicts.
- **Stale grading errors are hard stops.** If a grading tool returns
  `stale_or_fabricated_student_input`, do not affirm the submitted move/threat.
  Tell the student it was not their latest answer and ask them to restate it or
  answer directly without grading.
- **No raw deliberation or fake verification.** Never render phrases like "but wait"
  or "that's confusing", and never claim to check/re-check an engine line unless a
  tool was actually called in that turn.
- **`submit_diagnosis` accepts only the closed taxonomy** in
  `app/tools.py::ERROR_TAXONOMY` — that is what makes the dossier aggregate.
- **`TimelineManager.reset()` rewinds but keeps the loaded game**;
  `clear()` is the full wipe. Don't conflate them.
- **`app/concepts.py` and `app/motifs.py` stay pure** (board in, data out,
  no engine/network/UI) — that's why they're exhaustively unit-testable.
- **Aspirational instinct is reveal-gated and divergence-guarded.**
  `review._aspirational_instinct` shows what a ~+350-Elo player would choose only
  when that move *differs* from the student's played move (never dressing the
  student's own move as the target), only after the moment resolves (inlined in the
  digest's WITHHELD block), and self-suppresses near the rating cap.

## Development

- Windows-first dev environment; venv at `.venv` (`.venv/Scripts/python.exe`).
- Run tests: `python -m pytest tests/ -q` — fast, no engines or network needed
  (fakes live in `tests/conftest.py`).
- Concept-extractor changes require regenerating the snapshot baseline
  deliberately: `python -m tests.test_concepts_snapshot --save`.
- Manual model-loading smoke scripts belong in `scripts/`, never named `test_*.py`
  (pytest would load the model at collection time).
- Secrets: `GEMINI_API_KEY` and the optional `LICHESS_TOKEN` (scope-less; sent only to Lichess hosts,
  required by the opening explorer) come from `.env` (gitignored). Never commit
  `config.json` (machine-local paths) — `config.example.json` is the template.

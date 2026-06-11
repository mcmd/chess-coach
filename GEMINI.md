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
app/engines.py       EngineManager: Stockfish + Maia-2 processes AND all engine policy
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
  `engines.stockfish.analyse` from other modules; add a method instead.
- **Socratic withholding is backend-enforced**: `evaluate_student_answer` counts
  attempts per FEN and only includes the engine answer when `reveal_allowed`.
- **`submit_diagnosis` accepts only the closed taxonomy** in
  `app/tools.py::ERROR_TAXONOMY` — that is what makes the dossier aggregate.
- **`TimelineManager.reset()` rewinds but keeps the loaded game**;
  `clear()` is the full wipe. Don't conflate them.
- **`app/concepts.py` and `app/motifs.py` stay pure** (board in, data out,
  no engine/network/UI) — that's why they're exhaustively unit-testable.

## Development

- Windows-first dev environment; venv at `.venv` (`.venv/Scripts/python.exe`).
- Run tests: `python -m pytest tests/ -q` — fast, no engines or network needed
  (fakes live in `tests/conftest.py`).
- Concept-extractor changes require regenerating the snapshot baseline
  deliberately: `python -m tests.test_concepts_snapshot --save`.
- `scripts/smoke_maia2.py` is a manual smoke test that downloads/loads the
  Maia-2 model — never name it `test_*.py` (pytest would load the model at
  collection time).
- Secrets: `GEMINI_API_KEY` comes from `.env` (gitignored). Never commit
  `config.json` (machine-local paths) — `config.example.json` is the template.

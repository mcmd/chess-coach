# Chess Coach (Engine-Grounded AI Coaching)

## Overview

An AI chess coach that runs in your terminal. A Gemini LLM conducts the lesson — asks questions, explains ideas, adapts to you — while a Python backend computes every chess fact it teaches from: Stockfish evaluations, Maia-2 human-move predictions, exact exchange evaluation, tactical motif detection, Lichess opening statistics, and Syzygy tablebases.

The architecture is deliberately **hybrid**:

- **Backend-led analysis.** When you load a game, the backend reviews every move (evaluation swings measured in win-percentage space), classifies each move (`blunder` / `mistake` / `inaccuracy` / `ok` / `good` / `best` / `only-move` / `forced`), selects the most instructive moments as a **lesson agenda**, and hands the coach a verified digest — including the engine's answers, marked *withheld*.
- **LLM-led pedagogy.** The coach works through the agenda one moment at a time: it sets the scene, asks you ONE question, and grades your answer through a backend tool that counts your attempts — the engine line is only unlocked when you get it right or after two tries. Verified good moves get genuine credit; mistakes that Maia says most players at your rating also make are normalized before they're dissected.

The LLM is never the source of chess truth. It cannot judge your answers itself, cannot pick its own sparring moves, and cannot reveal withheld analysis early — those are all backend-enforced.

## Key Features

- **Full-game review with a lesson agenda.** `/load_pgn` (or pasting a PGN) runs an engine review of every move and produces critical moments with: win% swing, the better move and its line, the punishment line, detected tactical motifs (forks, pins, skewers, back-rank mates, removal of the defender, discovered checks), Maia's probability that a player at your rating makes the same mistake, and clock-based time flags (`hope_chess`, `time_scramble`, `freezing`).
- **Socratic loop with graded answers.** When the coach asks "find the better move", your answer is engine-graded (`best` / `excellent_alternative` / `reasonable_but_imprecise` / `inaccurate` / `wrong`, with rank and win% cost). Near-misses get partial credit; after two failed attempts the answer unlocks and is taught step by step.
- **Exact piece-safety (SEE).** Hanging and weakly-defended pieces are computed by legal-move static exchange evaluation — pinned defenders, x-rays, and capture sequences are handled correctly, so "your knight was actually lost there" is mathematically true.
- **A rich, verified concept library.** Loose pieces (LPDO), overloaded defenders, trapped pieces, pawn tension, available pawn breaks, backward pawns, back-rank/luft vulnerability, rooks on open files, opposite-colored bishops, mobility counts, and exact endgame rules (rule of the square, wrong-colored-bishop draws) — all pure-Python, all emitted only when present, all fed into the lesson agenda's "Verified position facts."
- **Threat detection.** A null-move engine probe answers "what was the opponent threatening if you'd done nothing?" — the most important question at club level.
- **Human-realistic sparring.** In `/spar` mode the coach's moves are picked by the backend by sampling Maia-2 at your Elo (not by the LLM). If your move loses significant win%, the coach pauses the game and turns it into a mini-lesson.
- **A dossier that actually accumulates.** Diagnoses use a closed mistake taxonomy (`hanging_piece`, `ignored_threat`, `missed_fork`, ...) so your recurring weaknesses aggregate across sessions. `/review` quizzes you on positions from your own past mistakes.
- **Anti-hallucination plumbing.** The backend owns the board; the model is fed the current FEN, legal moves, and capture-annotated history with every message; board diagrams are rendered by the backend and inserted via opaque tokens the model can only paste, never draw.
- **Elo-aware everywhere.** Maia predictions, opening statistics (master *and* your-rating amateur databases), and the coaching register all key off your configured rating.
- **Profiles, persistent memory, clock analysis, multi-language coaching** — as before, with history persistence now atomic, capped, and scrubbed of internal scratchpads.

## The Coach's Tools

| Tool | What it does |
|---|---|
| `analyze_position` | The full coaching packet: top-3 engine lines, SEE piece safety, threats-if-you-pass, Maia tendencies, tactical motifs, phase, structural features, before/after board tokens. Supports hypothetical lines via `moves_to_apply`. |
| `evaluate_student_answer` | Grades the student's proposed move; counts attempts; withholds the answer until earned. |
| `play_sparring_reply` | Backend plays the coach's sparring move (Maia-sampled at student Elo). |
| `execute_coach_move` | Plays a user-dictated move, guarded by the Logic Tutor legality layer. |
| `submit_diagnosis` | Logs a mistake to the dossier using the closed taxonomy. |
| `get_game_review_summary` | Re-fetches the loaded game's review digest. |
| `get_opening_repertoire` | Lichess master + amateur (your Elo bracket) opening statistics. |
| `query_endgame_tablebase` | Syzygy perfect play (≤7 pieces, backend-enforced). |
| `fetch_lichess_game_by_id` / `fetch_chessgames_com_game` | Download a game, review it, load it. |

## Commands

```
/load_pgn <path>   Load + review a PGN file (engine review, lesson agenda)
/move <san>        Play a move (e.g. /move Nf3)
/next  /prev       Step forward / back through the loaded game
/goto <n>[b]       Jump to before White's (or Black's) move n
/reset             Rewind to the start of the loaded game (keeps the game)
/spar  /nospar     Sparring mode on / off
/review            Quiz yourself on a past mistake from your dossier
/memorize          Save this session's diagnoses to your long-term dossier
/clear             Wipe board, review, and conversation memory
/help              Show commands
```

You can also paste a PGN directly, or just talk.

## Setup

1. Create `.env` in the project root:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
2. Copy `config.example.json` to `config.json` and edit it:
   ```json
   {
       "user_elo": 1200,
       "language": "English",
       "stockfish_path": "/path/to/your/stockfish",
       "gemini_model": "gemini-3.1-pro-preview",
       "profiles_dir": "./profiles",
       "player_name": "your-lichess-username"
   }
   ```
   `player_name` is matched against PGN headers so the coach knows which side you played. Download [Stockfish](https://stockfishchess.org/download/) for your OS; Maia-2 installs via pip and downloads its model on first run.
3. Install and run:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   python coach.py                # or: python coach.py --profile=name
   ```

CLI flags: `--profile=<name>` (isolated config/memory/dossier per user), `--wipe_memory`, `--dark_terminal`.

## Development & Testing

```bash
python -m pytest tests/ -q
```

The suite (66 tests) runs in ~2 seconds with no engines or network — engine doubles live in `tests/conftest.py`. It includes adversarial SEE positions (pinned defenders), motif detection, classification thresholds, the full review pipeline against scripted evaluations, command-layer regressions, the Socratic attempt/reveal state machine, and dossier aggregation.

If you intentionally change the concept extractor, regenerate the snapshot baseline: `python -m tests.test_concepts_snapshot --save`.

`scripts/smoke_maia2.py` is a manual Maia-2 smoke test (loads the real model).

See `GEMINI.md` for the module map and architectural invariants.

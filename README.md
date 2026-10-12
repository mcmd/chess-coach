# Chess Coach (Engine-Grounded AI Coaching)

## Overview

An AI chess coach that runs in your terminal. A Gemini LLM conducts the lesson (asking questions, explaining ideas, adapting to you) while a Python backend computes every chess fact it teaches from: Stockfish evaluations, Maia human-move predictions (history- and outcome-aware **Maia-3**), exact exchange evaluation, tactical motif detection, Lichess opening/master databases, and Syzygy tablebases.

The architecture is deliberately **hybrid: backend-led analysis, LLM-led pedagogy**.

- **The backend finds the mistakes and assembles the truth.** When you load a game, it reviews every move (evaluation swings measured in win-percentage space, with Maia consulted on every position), classifies each move, selects the most instructive moments as a **lesson agenda**, and hands the coach a verified digest: the better move and its line (marked *withheld*), the punishment line with named tactical motifs, verified positional facts, a graded **question ladder**, a recommended diagnosis, and concrete explanation primitives for the reveal phase.
- **The LLM teaches.** It works the agenda one moment at a time: opens with a Level-0/1 question, grades your answers through a backend tool that counts attempts (the answer unlocks only when you earn it or after two tries), asks you *why* the right move works, then explains from backend-verified facts.

The LLM is never the source of chess truth. It cannot judge your answers itself, cannot pick its own sparring moves, cannot reveal withheld analysis early, cannot log a diagnosis category the backend disagrees with without justifying the override, and cannot quietly grade one move out of a hedged candidate list: those are all backend-enforced.

## Key Features

- **Full-game review with a lesson agenda.** `/load_pgn` (or pasting a PGN, or fetching by Lichess/chessgames ID) runs an engine review of every move and produces critical moments with: win% swing, the better move and line, the punishment line, detected motifs, time flags from PGN clocks (`hope_chess`, `time_scramble`, `freezing`), and verified position facts (open files, hanging/loose/trapped pieces, king exposure, pawn tension…).
- **Maia on every move: the typicality lens.** Each mistake is classified as *typical for your level* (a knowledge gap: normalize, then teach the pattern) or *atypical* (a process slip: coach the habit, skip the lecture). The same data finds **brilliancies** (only-moves few peers find), **avoided traps** (you sidestepped the move most peers play), **findability** of the right move (calibrates hints and praise), **practical difficulty** (engine eval vs. how often humans actually convert), your **most likely wrong answer** (pre-refuted), the **aspirational instinct** (what players ~350 Elo above you choose), a per-phase **human-alignment** stat, and (from Maia-3's win/draw/loss head) **risk and drawing-chance** verdicts for the side to move.
- **Maia-3: history- and outcome-aware, tiered.** The human-move model is loaded in-process and reads the **moves leading into a position** (not just the static board), and exposes a **WDL** (win/draw/loss) head, so predictions and risk reads are sharper than a positional-only model. Two tiers share a loaded checkpoint (a *deep* tier for agenda moments, the aspirational instinct, `analyze_position`, and batch representatives; a *scan* tier for per-move review). If the model fails to load, the coach continues engine-only (no human-move lens) behind a clear warning, rather than silently degrading. Set `maia3_offline: true` once the checkpoint is cached to load from the local Hugging Face cache with no startup phone-home.
- **A Socratic loop that is enforced, not requested.** Question ladder (open question → conceptual nudge → directed hint → near-reveal), answer grading with engine verdicts and partial credit, attempt counting with reveal-on-earn, verified `hint_facts` so hints can't be invented, and a command-layer guard that detects hedged multi-candidate answers and forces a commitment before grading.
- **Online database grounding (network-enhanced, never network-dependent).** Reviews walk the opening through the Lichess masters database: book moves are tagged (theory is never scolded), and the exact move where the game **left established theory** is reported; when the student's deviation is costly, it becomes a full agenda moment with the theory moves *and their scoring* (for masters and for your rating bracket) as the withheld answer, diagnosed as `opening_theory`. In tablebase range (≤7 pieces) every ply is probed: **category flips** (a proven WIN becoming a proven DRAW) are forced onto the agenda even when the engine's eval swing is too small to notice, the digest marks where the game *entered* tablebase territory ("from move 38 this was a proven draw, everything after is technique"), and answer grading switches from centipawns to game-theoretic categories. All of it degrades silently to engine-only behavior when offline.
- **Exact piece-safety (SEE).** Hanging, weakly-defended, and trapped pieces are computed by legal-move static exchange evaluation: pinned defenders, x-rays, and capture sequences are handled correctly, so "your knight was actually lost there" is mathematically true.
- **A rich, verified concept library.** Loose pieces (LPDO), overloaded defenders, pawn tension, available pawn breaks, backward pawns, back-rank/luft vulnerability, rooks on open files, opposite-colored bishops, mobility counts, king castled-status, and exact endgame rules (rule of the square, wrong-colored-bishop draws), all pure Python, emitted only when present.
- **Tactical motif detection on engine lines.** Forks, pins, skewers, discovered attacks and checks, double checks, zwischenzugs, removal of the defender, back-rank and smothered mates, promotions, detected on forcing engine PVs for precision, and used to name what you missed, what punished you, and what diagnosis fits.
- **A dossier that actually accumulates.** Diagnoses use a closed 16-category mistake taxonomy with backend-recommended categories and override governance, so recurring weaknesses aggregate across sessions, including a **typicality trend** (improvement shows as fewer peer-common errors). `/review` quizzes you on positions from your own past mistakes.
- **Cross-game pattern mining (`/batch`).** Scan 10–50 of your games to cluster recurring mistakes deterministically, ranked by **both frequency and impact** (average severity in plain blunder/mistake/inaccuracy terms, not a raw win% sum), then deep-verify a few representatives per pattern as teaching positions (with the Lichess link, move number, and opponent's last move). It also surfaces an eval-only **cross-game slide signature**, a habit of gradual one-sided collapses with no single blunder, scored against a ~17% corpus baseline (calibrated on Lichess `[%eval]`, and read straight from those annotations when the PGN carries them), and tags each pattern on a **knowledge-gap vs. execution** axis. The findings evolve a dossier trend across batches with an honest active/improving/stale lifecycle.
- **Lesson resume.** Lessons persist across sessions: re-fetching a Lichess game offers continue-or-restart; reloading a local or pasted game (identified by movetext hash) resumes at the saved moment automatically (`/reset` restarts). Completed lessons load fresh.
- **Grounded storytelling.** `explain_pawn_structure_evolution` turns the game's structural history into a backend-verified narrative with vivid/concrete language pairs and explicit avoid-claims derived from the actual position, so color commentary never contradicts the board.
- **Human-realistic sparring.** In `/spar` mode the coach's moves are sampled from the Maia human-move model at your Elo (not chosen by the LLM); blunders pause the game for a mini-lesson.
- **Anti-hallucination plumbing.** The backend owns the board; every message carries FEN, legal moves, and capture-annotated history; board diagrams are backend-rendered and inserted via opaque pre-minted tokens (invented tokens are scrubbed at render time); positional prose may only cite backend facts.
- **Execution tracing.** `--trace` (or `"trace": true`) shows every tool call with arguments and timing, and every Stockfish/Maia/Lichess invocation, the coach's reasoning made visible.
- **Profiles, persistent memory, multi-language coaching**, atomic file writes, capped/scrubbed history.

## The Science Behind the Coach

Every mechanism below exists in the code today; references are foundations, not decoration.

### 1. LLMs cannot be trusted with board state, so they never hold it
Language models are sequence predictors without a persistent board representation; in chess they reliably drift into illegal moves, phantom pieces, and invented geometry ([Westerdale 2024][ref-westerdale]). The backend therefore owns the only board (`SessionState` + `TimelineManager`); the model's spatial claims are restricted to backend-computed data (verified position facts, SEE-based safety, motif details), diagrams are opaque backend-minted tokens, and every hypothetical line must be verified through `analyze_position` before it may be asserted.

### 2. Computation is delegated to solvers; language stays with the language model
Strategic calculation plateaus far below expert level when attempted in text, while tool-delegating architectures let each component do what it is actually good at ([Hwang et al. 2025][ref-sirl]; [Chen et al. 2025][ref-vtool]). Stockfish provides evaluation and refutation, Maia provides human behavior, Syzygy provides proven endgame truth, pure-Python chess logic provides concepts and motifs, and the LLM's job narrows to pedagogy and prose. Engine lines are truncated (≤8 plies) and multipv-bounded to keep the model grounded rather than flooded.

### 3. Human-aligned modeling: teaching the player you are, not the engine you aren't
Maia was built to predict *human* moves at specific skill levels, motivated explicitly by chess teaching ([McIlroy-Young et al., KDD 2020][ref-maia1]; [Maia-2, NeurIPS 2024][ref-maia2]). This coach uses it on **every move** of a review, now via **Maia-3**, which conditions on the moves leading into a position (not just the static board) and predicts a win/draw/loss distribution as well as a move: mistakes most peers share are taught as knowledge gaps; mistakes peers rarely make are coached as process slips; only-moves few peers find are celebrated as brilliancies; the findability of the right answer calibrates hints; the WDL head turns into honest risk and drawing-chance verdicts; and the "aspirational instinct" shows what players a few hundred points stronger choose: a concrete image of the next level. Engine-only coaching can do none of this.

### 4. Move quality in win-percentage space
Centipawn swings exaggerate irrelevant differences in decided positions (−2 → −5 is not a blunder in any human sense). Classification therefore converts evaluations to expected win percentage and measures the *drop*, following the approach popularized by Lichess's accuracy metric ([Lichess][ref-lichess-acc]). Thresholds: ≥30 blunder, ≥20 mistake, ≥10 inaccuracy, applied to the student's side of the board.

### 5. Retrieval before instruction: the question ladder and the earned reveal
Being made to retrieve (or honestly attempt) an answer before receiving it produces dramatically better retention than being told ([Roediger & Karpicke 2006][ref-testing]), errors corrected after a confident attempt are remembered *better* than passive instruction ([Butterfield & Metcalfe 2001][ref-hypercorrection]), and struggling before instruction beats instruction-first ([Kapur 2008][ref-kapur]). Hence: the backend withholds the answer until the student attempts (or exhausts two attempts), the question ladder escalates scaffolding gradually in the spirit of the zone of proximal development ([Vygotsky][ref-vygotsky]), hedged candidate lists must be committed to one move before grading, and after the reveal the student is asked to explain *why* the move works. The sharpest facts are deliberately saved for the post-reveal explanation ("ask less before, explain more after").

### 6. Deliberate practice on critical moments, not move-by-move commentary
Expertise grows from focused effort on specific weaknesses at the edge of competence, with immediate feedback ([Ericsson et al. 1993][ref-ericsson]). The review compresses a game into 3–6 agenda moments chosen by win% damage, each teaching ONE concept, each ending in a taxonomy-classified diagnosis, and the dossier turns those diagnoses into a longitudinal picture (recurring categories, assigned habits, typicality trend) that `/review` converts into spaced re-testing on your own positions.

### 7. Concept-grounded commentary
Free-form chess commentary by generative models is fluent but strategically vacuous or wrong; grounding the commentary in explicitly computed concepts measurably improves it ([Kim et al. 2024][ref-ccc]). The concept extractor (SEE safety, pawn-play, king safety, activity, endgame rules) and the motif detector feed every lesson; the pawn-structure narrator goes further and supplies vivid/concrete language *pairs* plus position-derived avoid-claims, so style can never outrun truth.

### 8. Self-reported confidence is not calibration
Models behave like good test-takers: they guess confidently rather than admit uncertainty ([Kalai et al. 2025][ref-kalai]). An earlier design asked the model to self-score diagnosis confidence and filtered on it; that mechanism was removed. Instead, diagnosis categories are **recommended by the backend from motif and clock evidence**, and the model may only deviate by stating the conversational evidence for a different root cause: justification, not self-rated confidence.

[ref-westerdale]: https://www.nicowesterdale.com/blog/why-llms-cant-play-chess
[ref-sirl]: https://arxiv.org/abs/2507.00726
[ref-vtool]: https://arxiv.org/abs/2505.11792
[ref-maia1]: https://arxiv.org/abs/2006.01855
[ref-maia2]: https://arxiv.org/abs/2409.20553
[ref-lichess-acc]: https://lichess.org/page/accuracy
[ref-testing]: https://doi.org/10.1111/j.1467-9280.2006.01693.x
[ref-hypercorrection]: https://doi.org/10.1037/0278-7393.27.6.1491
[ref-kapur]: https://doi.org/10.1080/07370000802212669
[ref-vygotsky]: https://en.wikipedia.org/wiki/Zone_of_proximal_development
[ref-ericsson]: https://doi.org/10.1037/0033-295X.100.3.363
[ref-ccc]: https://arxiv.org/abs/2410.20811
[ref-kalai]: https://arxiv.org/abs/2509.04664

## The Coach's Tools

| Tool | What it does |
|---|---|
| `analyze_position` | The full coaching packet: top-3 engine lines, SEE piece safety, threats-if-you-pass (null-move probe), Maia tendencies, tactical motifs, phase, structural features, before/after board tokens. Supports hypothetical lines via `moves_to_apply`. |
| `evaluate_student_answer` | Grades the student's move (tablebase categories override centipawns in ≤7-piece endings); counts attempts; withholds the answer until earned; returns verified `hint_facts` while withholding and explanation primitives on reveal; refuses to grade uncommitted candidate lists. |
| `evaluate_threat_answer` | Grades "what is the opponent threatening?" answers against the null-move threat probe (correct / real-but-secondary / not-a-real-threat) with the same attempt counting and withholding as move answers. |
| `submit_diagnosis` | Logs a mistake using the closed 16-category taxonomy; deviating from the backend's recommended category requires a substantive override reason. |
| `explain_pawn_structure_evolution` | Backend-verified pawn-structure story of the loaded game with safe/avoid language guardrails. |
| `get_game_review_summary` | Re-fetches the loaded game's review digest. |
| `play_sparring_reply` | Backend plays the coach's sparring move (Maia-sampled at student Elo). |
| `execute_coach_move` | Plays a user-dictated move, guarded by the Logic Tutor legality layer. |
| `get_opening_repertoire` | Lichess master + your-Elo amateur opening statistics (cached). |
| `query_endgame_tablebase` | Syzygy proven result, best move, and margin for error ("2 of 14 legal moves preserve the win"). ≤7 pieces, backend-enforced. |
| `fetch_lichess_game_by_id` / `fetch_chessgames_com_game` | Download, review, and load a game. The Lichess fetch supports `mode=auto/continue/restart` for lesson resume. |

## Commands

```
/load_pgn <path>   Load + review a PGN file (engine review, lesson agenda, auto-resume)
/batch <path>      Scan a multi-game PGN (10-50 games) for recurring mistake patterns
                   (/batch alone reloads the last report)
/move <san>        Play a move (e.g. /move Nf3)
/next  /prev       Step forward / back through the loaded game
/goto <n>[b]       Jump to before White's (or Black's) move n
/reset             Rewind to the start of the loaded game (keeps the game)
/spar  /nospar     Sparring mode on / off
/review            Quiz yourself on a past mistake from your dossier
/memorize          Save this session's diagnoses to your long-term dossier
/clear             Wipe board, review, lesson state, and conversation memory
/help              Show commands
```

You can also paste a PGN directly, or just talk.

## Talking to Your Coach

Beyond the slash commands, you drive the coach in plain language; it has backend tools for review, position analysis, openings, endgames, sparring, and your dossier, and it decides which to call. There's no special syntax; ask the way you would ask a human coach. Every example below is backed by a real backend tool, so the answers are engine-grounded, not invented.

**Review a game**
- `Review this Lichess game: 2dtDqQgRy2OD` to fetch, engine-review, and start the lesson
- `Here's my game:` then paste the PGN (or `/load_pgn game.pgn`)
- `Pull this one from chessgames.com: <id or url>`
- `What were my critical mistakes?` / `Give me your insights`

**Answer the coach during a review.** It teaches one critical moment at a time and asks you first; reply naturally:
- `Black is threatening exd4` (a threat read)
- `I'd play Rad1` / `Ne4` (a candidate move)
- `Because it brings my last rook into play and defends e4` (the *why*, after you've found the move)

You earn the answer by attempting: it unlocks when you are right, or after two tries.

**Ask about risk & drawing chances** (from the human-move model, for the side to move)
- `How risky is this position, am I playing for a win or a draw?`
- `What are my practical drawing chances here?`
- `Is this drawish, or should I keep pushing?`

**Understand a position or test an idea**
- `Is anything hanging here?` / `Why is my knight on a5 bad?`
- `What is the threat if I do nothing?`
- `What if I had played Rfe1 instead?` (the coach verifies the line on the engine before answering)

**Pawn-structure story**
- `How did my pawn structure evolve during the game?`

**Openings & endgames**
- `What is the main theory in this position?` / `What do strong players play here?`
- `Is this endgame winning, or a theoretical draw?` (Syzygy tablebase, ≤7 pieces)

**Spar against a human-like opponent**
- `/spar`, then play your moves (`/move e4`, or just say your move); `/nospar` to stop
- The coach's replies are sampled from the human-move model at your Elo; a blunder pauses the game for a mini-lesson

**Work on your recurring weaknesses (dossier)**
- `What do I keep getting wrong?`
- `/review` quizzes you on a position from a past mistake
- `/memorize` saves this session's diagnoses to your long-term dossier
- `/batch games.pgn` scans 10–50 of your games for recurring patterns and trains on them

Tip: run with `--trace` to watch exactly which tools the coach calls for each answer.

## Setup

1. Create `.env` in the project root:
   ```env
   GEMINI_API_KEY=your_api_key_here
   LICHESS_TOKEN=lip_your_token_here
   ```
   `LICHESS_TOKEN` is optional. Lichess's opening explorer (masters and rating-band databases used for
   theory grounding) rejects anonymous requests; without a token those lookups are skipped and
   everything else still works. Create one at <https://lichess.org/account/oauth/token/create> with
   **no scopes ticked**: the explorer only needs a valid token, and a scope-less token can read only
   public data. The coach sends it to Lichess hosts only.
2. Copy `config.example.json` to `config.json` and edit it:
   ```json
   {
       "user_elo": 1200,
       "language": "English",
       "stockfish_path": "/path/to/your/stockfish",
       "gemini_model": "gemini-3.1-pro-preview",
       "profiles_dir": "./profiles",
       "player_name": "your-lichess-username",
       "usernames": {
           "lichess": ["your-lichess-username"],
           "chesscom": ["your-chesscom-username"]
       },
       "dark_terminal": false,
       "trace": false,
       "review_eval_time": 0.5,
       "use_online_databases": true,
       "maia3_offline": false
   }
   ```
   `usernames` is matched case-insensitively against PGN `White`/`Black` headers so the coach knows which side you played; list every handle you use. `player_name` still works as a legacy single-handle alias. `review_eval_time` is Stockfish seconds per position during a game review (higher = more reliable classification, slower scan). `use_online_databases` toggles the Lichess explorer/tablebase enrichment. `maia3_offline` loads the Maia-3 checkpoint from the local Hugging Face cache only (set it `true` after the first run to silence the startup phone-home). Download [Stockfish](https://stockfishchess.org/download/) for your OS; Maia-3 installs via pip and downloads its checkpoint from Hugging Face on first run (cached thereafter).
3. Install and run:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   python coach.py                # or: python coach.py --profile=name --trace
   ```

CLI flags: `--profile=<name>` (isolated config/memory/dossier/lesson state per user), `--wipe_memory`, `--dark_terminal`, `--trace`.

Runtime files (per profile): `chess_coach_memory.json` (conversation), `player_dossier.json` (long-term weaknesses), `lesson_state.json` (lesson resume), `chess_cli_history.txt` (CLI history). All gitignored.

## Development & Testing

```bash
python -m pytest tests/ -q
```

The suite (260+ tests) runs in a few seconds with no engines and no network; engine doubles live in `tests/conftest.py`, and an autouse fixture blocks all real HTTP. It covers adversarial SEE positions (pinned defenders), motif detection, classification thresholds, the review pipeline against scripted evaluations, opening/tablebase grounding with injected API payloads, the Socratic attempt/reveal state machine, candidate-commitment guards, lesson resume, diagnosis override governance, and dossier aggregation.

If you intentionally change the concept extractor, regenerate the snapshot baseline: `python -m tests.test_concepts_snapshot --save`.

### Live conversation evaluation

Live evals use role-specific documents and a project model configuration,
separate from the mocked pytest suite and backend-only `scripts/eval_coaching.py`:

| Document | Role |
|---|---|
| `tests/2dtDqQgR_script.md` | Pass-1 player: messages, semantic gates, allowed branches, stop rule |
| `tests/live_coach_run.md` | Runner/operator: isolation, capture, waiting, shutdown, evidence handoff |
| `tests/2dtDqQgR_review.md` | Pass-2 reviewer: correctness, grounding, lesson state, coaching quality |
| `tests/live_coach_agents.json` | Role → project agent: `live-coach-player`, `live-coach-reviewer` (`.omp/agents/*.md` set the models) |

The scenario (`lichess-2dtDqQgR-full-v1`) is a complete game review: loading,
self-assessment, all four agenda moments (threat grading, a legal but inferior
answer, a hint, an illegal move, candidate commitment, explanations, an in-check
moment), questions about earlier positions, and post-review questions (pawn
structure, risk, takeaway). Script steps declare offline checks with
`**Checks:** detour` (lesson state must not change) and `**Checks:** return N`
(the agenda moment at move N is restored). The player does
not check legality or grade responses. After shutdown, the reviewer uses frozen
evidence and deterministic checks, rather than influencing the live session.

The executable entry point is `scripts/run_live_eval.py::run_eval`. In an omp
Eval cell, from the repository root:

```python
from scripts.run_live_eval import run_eval
result = await run_eval(agent_fn=agent, tool_fn=tool)
display(result)
```

It needs the harness's `agent`/`tool` interfaces, the project `.venv`, and the
normal live coach prerequisites. It is not a standalone shell/CI command.
Another coding agent can call the same entry point in its own Eval session;
`tests/live_coach_run.md` describes that handoff.

After repository changes, `AGENTS.md` requires this flow in addition to relevant
deterministic checks. Each run copies only `mcmd`'s configuration into
`profiles/live-eval-2dtDqQgR-*`, then uses the real Gemini/Stockfish/Maia/Lichess.
The configured player receives exact scripted messages and rendered replies.
The independent reviewer starts only after shutdown and deterministic checks.

Each run retains `conversation.md`, `terminal.log`, flushed `events.jsonl`,
`event-index.json`, `execution.json`, `checks.json`, `manifest.json`, script/config
snapshots, `review.json`, and `report.md`. Structured events contain API requests
and responses, complete tool/engine inputs and results, actual PGN/network
responses, diagrams, and lesson-state snapshots. Credentials and unavailable
private model internals are excluded. Newline input replaces terminal editing;
this exercises the actual coach but does not evaluate prompt-toolkit interaction.

Each role runs as a project agent whose definition sets its model:
`.omp/agents/live-coach-player.md` (`openai-codex/gpt-6-luna:high`) and
`.omp/agents/live-coach-reviewer.md` (`anthropic/claude-opus-5-5:high`).
`tests/live_coach_agents.json` maps each role to its agent. Use explicit
`provider/model[:level]` selectors: in a subagent, `@default` follows the calling
session's live model rather than `modelRoles.default`. Each run snapshots both
definitions and records the requested selector, agent IDs, the model each agent
actually ran on (from its omp session file), and the model ID it reported. No coach retry.

Script completion is not an overall coaching pass. Exact wording/percentages are
not snapshots; missing evidence is not a pass. Real API calls can take several
minutes. Known quota blockers are reported without retrying; archived-run checks
are not new live trials.

### Manual scripts (`scripts/`)

These are developer/research tools, **not** part of the pytest suite (they load real models or hit the network). Run them from the repo root with the project venv, e.g. `python scripts/<name>.py`.

| Script | What it does |
|---|---|
| `scan_eval_slides.py` | Eval-only "within-game slide" corpus scanner (no model, no tokens). Downloads a player's analysed Lichess games (`--user <name>`, cached locally) and classifies each game's archetype (gradual slide / single cliff / chaos / clean) purely from the `[%eval]` annotations, reporting the base rate of sustained one-sided slides and dumping candidate games. Used to validate the `/review` slide detector (`app/review.py::_detect_slides`). Example: `python scripts/scan_eval_slides.py --user msnid --max 3000`. |
| `validate_batch_slides.py` | End-to-end `/batch` validator: runs the real pipeline (Stockfish + Maia-3) on a multi-game PGN exactly as the command does, writing all output to a throwaway temp dir, then prints the cross-game slide signature, per-pattern axis tags, the relevant digest blocks, and the dossier `slide_trend` the run produces, so the batch output can be eyeballed against ground truth without touching a real profile. Example: `python scripts/validate_batch_slides.py games/your.pgn <player>`. |

See `GEMINI.md` for the module map and architectural invariants.

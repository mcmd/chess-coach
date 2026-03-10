# Gemini Chess Coach (Agent + Tools Architecture)

## Overview
This project is an advanced AI chess coach that combines the reasoning and conversational capabilities of a Large Language Model (Gemini) with the domain-specific accuracy of dedicated chess engines and databases. 

The core philosophy of this project is its **Agent + Tools Architecture**. While LLMs are excellent at orchestrating complex tasks and translating dense, technical chess concepts into easily understandable, plain-language explanations, they are prone to hallucinations when calculating deep chess lines or evaluating positions. 

To solve this, we give the Gemini agent access to a suite of highly specialized tools. The LLM orchestrates the learning experience, but it grounds its advice in absolute chess truth. By querying Stockfish for objective evaluations, the Maia chess engine for human psychology, Lichess opening explorers for real-world statistics, and Syzygy tablebases for perfect endgame play, the agent provides a personalized, accurate, and deeply insightful coaching experience.

Currently operating as a rich Command Line Interface (CLI) tool, it allows you to analyze specific positions, load PGNs, play moves interactively, and get human-like feedback tailored to your exact Elo rating.

*(Note: While the project structure contains remnants of a Flask web template, such as `main.py` and `src/index.html`, the current primary functionality is executed via the CLI. Future updates may introduce a web-based API.)*

## Key Features
* **Modular Architecture:** The codebase is fully modularized, with isolated components for configuration, state management, engine interactions, memory handling, tool definitions, conceptual extractions, and prompting logic. This ensures a clean, maintainable, and scalable foundation for future updates.
* **Localization:** The coach communicates natively in the language of your choice, defined by the `language` variable in `config.json`.
* **Neuro-Symbolic Architecture (Partial KL):** The agent utilizes a `<think>` block to perform internal monologue reasoning based on the 5 MATE Dimensions (Material, Activity, Pawn Structure, Space, King Safety) before synthesizing a response. This monologue is rendered as a beautifully formatted UI panel in the terminal. The agent is strictly grounded, required to cite the JSON tool returns for its internal logic.
* **The Logic Tutor:** When the Coach attempts to make a move during sparring, a deterministic Logic Tutor acts as an intervention layer. If the LLM hallucinates an illegal move, the Logic Tutor intercepts it, explains *why* it's illegal (e.g., pinned piece, in check), and mathematically proves the correct legal moves, curing LLM spatial blindness.
* **Context Poisoning Prevention:** The system meticulously controls the amount of data fed to the LLM during game loads. Instead of dumping entire 60-move PGNs into the prompt, it provides only the current FEN, the 5 most recent moves, and mathematically generated legal moves, eliminating token bloat and preventing "Context Poisoning".
* **State Encapsulation:** The internal architecture relies on a strict `SessionState` object and a `TimelineManager` to ensure zero state-desync bugs during branching or timeline stepping.
* **Structured Chain-of-Thought (The Diagnostician):** The coach uses a strict taxonomy to explicitly diagnose decision failure modes before it speaks, ensuring highly focused, pedagogical output. This diagnostic reasoning is displayed as a sleek UI panel before the main conversational response.
* **Long-Term Memory Manager (The Dossier):** Every time the Diagnostician identifies a mistake, it is temporarily logged. You can use the `/memorize` command to save these weaknesses to your long-term `player_dossier.json`. On startup, the Coach reads your dossier so it can reference your historical blunders and assigned preventative habits across sessions. Duplicate diagnoses for the same position are smartly rejected.
* **Interactive Live Sparring:** The coach features full interactive sparring. Use the `/move` command to play on the global board, ask the Coach to play a response, and use the new `/undo` command to take back moves during practice. The Coach uses the `execute_coach_move` tool to physically update the board state in real-time.
* **Game Stepping (VCR):** Loaded games can be stepped through chronologically. Use `/reset` to jump to the starting position of a loaded PGN, and `/next` or `/prev` to step through the moves one by one, allowing you to ask the Coach questions at specific, exact moments in the game without fast-forwarding to the end. You can even use `/clear` to hard-reset the board and wipe the AI's memory entirely.
* **Elo-Parametrized Coaching:** The coach adapts its advice, Maia predictions, and Lichess database queries to your exact Elo rating configured in `config.json`.
* **Global Engine Initialization:** Maia and Stockfish engines are initialized globally via an `EngineManager`, reducing latency when loading weights for move predictions and tactical evaluations. Stockfish is configured to use maximum available threads and Hash memory for superior depth-per-second performance.
* **Brilliancy & Trap Scanner (2-Pass Algorithm):** The system uses a sophisticated 2-pass engine scan on all loaded games. 
    1. **Fast Scan:** Rapidly filters positions to find candidates. 
    2. **Deep Confirmation:** Re-analyzes candidates with higher depth to rigorously verify `[⭐ ONLY MOVE FOUND!]` and `[🛡️ DODGED HUMAN TRAP!]` moments, ensuring GM-level accuracy before tagging them.
* **Engine Evaluation Caching:** To improve performance and prevent memory leaks, engine evaluations are managed by a bounded LRU Cache (`lru_cache`), ensuring rapid re-evaluation of previously seen FENs without overwhelming system memory.
* **Token-Based Visual Rendering System (UUID Integration):** To prevent the AI from hallucinating piece locations or mangling ASCII formatting, visual boards are tracked securely on the backend using unique UUIDs. The AI now has the capability to render distinct "Before Move" and "After Move" boards based on the tools it executes.
* **Deep Positional & Safety Analysis (`app/concepts.py`):** The engine tools automatically extract and provide deep structural heuristics to the AI via a dedicated conceptual pipeline. When evaluating a position, the AI instantly knows about:
    * **Material Imbalances** and Bishop pair status.
    * **Central Control dominance** (how many attackers are fighting for D4, E4, D5, E5).
    * **Pawn Structure defects** (isolated pawns, explicit island spans, passed pawns, doubled files, pawn majorities on the Queenside or Kingside).
    * **Piece Safety** (hanging pieces, pinned pieces).
    * **King Safety Perimeter** (exact pawn shield coordinates, enemy pieces actively attacking the squares around the King).
    * **Piece Activity and Coordination**: True Knight Outposts, Bad Bishops (structurally blocked by central pawns), connected rooks, and rooks stationed on the 2nd/7th rank.
    * **Development Phase**: Castling rights status and a strict count of undeveloped minor pieces on the back rank to penalize premature attacks.
* **Context Enrichment:** User inputs are secretly enriched with the current board's FEN and full algebraic move history (e.g., "1. e4 c5..."), ensuring the LLM always has perfect situational awareness of the ongoing game.
* **Clock & Time Analysis:** When loading games via PGN, the coach analyzes clock timestamps (`[%clk]`). It can identify "Hope Chess" (playing too fast on blunders) or "Freezing" (overthinking forced moves), providing feedback on the student's psychological state during the game.
* **Profiles:** Create isolated learning environments using the `--profile` flag. This separates your configuration (Elo, model), chat memory, and CLI history into independent subdirectories, perfect for studying different openings or having multiple users.
* **Wipe Memory Option:** Quickly reset your conversation history for a fresh start using the `--wipe_memory` flag.
* **Socratic Pedagogy:** The agent is prompted to teach concepts (King Safety, Piece Activity) rather than just spoon-feeding Stockfish lines.
* **Persistent Memory:** The conversation context is saved across sessions (`chess_coach_memory.json`), allowing you to resume your lessons later.
* **Singleton UI Console:** Ensures a smooth, flicker-free terminal experience by routing all outputs, spinners, and tool logs through a single shared `rich` console object.
* **Interactive CLI:** Built with `rich` and `prompt_toolkit`, featuring beautiful markdown rendering, color-coded chess notation, and cross-session up-arrow command history.
* **Built-in Commands:** Use `/load_pgn <filename>` to instantly import games for analysis, `/move <san>` to play moves on the internal board, or `/undo` to take back a move. Or use `/reset`, `/clear`, `/next`, `/prev`, and `/memorize` to manage your session and timeline.

## The Agent's Tools
The orchestrating Gemini LLM has autonomous access to the following tools to ground its knowledge:

- **`analyze_position` (Stockfish)**: Interrogates the local Stockfish engine to find the mathematically perfect, objective best moves and position evaluations. It calculates the first 5 moves of the Principal Variation (PV) for the top 2 engine lines. Furthermore, it supports analyzing hypothetical move sequences by passing in `moves_to_apply`, returning dual tokens to visualize the board *before* and *after* those moves are applied, empowering the AI to clearly answer "What if I play..." questions without spoiling the current board state. This tool now returns a comprehensive conceptual breakdown of the board's positional geometry, control dynamics, and safety features.
- **`predict_human_move` (Maia Chess + Stockfish)**: Uses pre-trained Maia neural network models to predict the most likely human move. Then, it uses Stockfish to analyze the resulting position if that predicted move is played, returning the objective evaluation and a deep refutation line (depth=25).
- **`get_opening_repertoire`**: Queries BOTH the Lichess Masters database and the Amateur database (filtered specifically to the user's Elo bracket). This allows the coach to compare Grandmaster theory against what amateur opponents actually play.
- **`query_endgame_tablebase`**: Accesses the Syzygy 7-piece endgame tablebase. When 7 or fewer pieces are left, the agent uses this tool to get mathematically perfect endgame play.
- **`fetch_lichess_game_by_id`**: Pulls specific games from Lichess using their game ID, fetching PGNs, clock times, and computer evaluations. Smartly handles full IDs with player perspective characters.
- **`fetch_chessgames_com_game`**: Imports Master-level games (using their numeric ID) from chessgames.com to teach concepts using examples from legends.
- **`execute_coach_move`**: A highly privileged tool that allows the Coach to push a chosen move onto the global board during interactive live sparring.
- **`submit_diagnosis`**: Used by the AI to formally log pedagogical weaknesses and insights into the student's dossier for long-term tracking.

## Setup Instructions

### Prerequisites & Configuration
Before running the application, you **must** configure your environment and engines:

1. Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

2. Create a `config.json` file in the root directory to define your local engine paths and Elo. **This will act as your default configuration and the template for new profiles:**
   ```json
   {
       "user_elo": 1200,
       "language": "English",
       "stockfish_path": "/path/to/your/stockfish",
       "lc0_path": "/path/to/your/lc0",
       "maia_weights_dir": "./maia_weights",
       "gemini_model": "gemini-2.5-pro",
       "profiles_dir": "./profiles"
   }
   ```
   *Note: Ensure you download and install [Stockfish](https://stockfishchess.org/download/) and [Lc0](https://lczero.org/play/download/) for your specific operating system.*

### Option 1: Project IDX / Firebase Studio (Recommended)

This project is pre-configured to run seamlessly in a Nix-based environment.

1. **Automatic Configuration**: The Nix environment (`.idx/dev.nix`) will automatically provision the workspace, creating a Python 3 virtual environment (`.venv`) and installing all dependencies from `requirements.txt`. It even installs Stockfish and Lc0.
2. **Terminal Usage**: To run the CLI tool, open the integrated terminal. Before running any `python` scripts or `pip` commands, activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. **Run the Coach**: Execute the script:
   ```bash
   python coach.py
   ```
   *To use a profile, add the flag:*
   ```bash
   python coach.py --profile=mcmd
   ```

### Option 2: Generic Local Setup (From Scratch)

To set up and run this project locally on any standard machine (Linux, macOS, or Windows):

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd <your-project-directory>
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   * On Linux/macOS: `source .venv/bin/activate`
   * On Windows: `.venv\Scripts\activate`

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**:
   ```bash
   python coach.py
   ```

## Known Limitations / Future Work
- The Flask template structure (`main.py`) is currently unused and serves as a placeholder for a future web UI or API layer.
- Ensure Lc0 is configured correctly on your OS if running locally outside the provided Nix environment, as it handles the Maia weights.

## 🔬 The Science of Pedagogical Grounding

Building an AI chess coach requires solving the fundamental tension between probabilistic neural reasoning and deterministic symbolic rules. Standard large language models (LLMs) are plagued by structural hallucinations because they lack an internal representation of the game state, leading to "line-of-sight" errors, temporal confusion, and illegal move suggestions.

The Gemini Chess Coach implements a **Type 6 Neuro-Symbolic Architecture**, where a neural "pedagogical brain" orchestrates external symbolic tools to ensure advice is anchored in absolute truth.

### 1. State Tracking Failure and Pointer Misbinding
* **The Challenge:** This is the primary driver of hallucinations. LLMs are probabilistic sequence predictors that lack a persistent board representation. As games progress, the model's internal "mental" board diverges from reality, leading it to recommend moves for captured pieces or ignore physical blockers[cite: 4].
* **Our Solution:** We implement **Symbolic State Grounding and Dual-Token Rendering**. The Python backend (specifically `app/state.py`) maintains the canonical game state using the `python-chess` library, which serves as the sole "source of truth"[cite: 17, 18, 22]. To prevent spatial hallucination, the LLM is never permitted to update the state autonomously. Furthermore, our `analyze_position` tool captures deterministic ASCII snapshots *before* and *after* hypothetical moves, binds them to secure UUIDs (e.g., `[[BOARD_a1b2c3d4]]`), and forces the LLM to simply output the correct token, stripping it of spatial rendering responsibilities.
* **Reference:** [Westerdale (2024)][ref1]

### 2. Historical State Integrity (Temporal Hallucinations)
* **The Challenge:** When reviewing past mistakes, models struggle with "context compression" and "reasoning degradation," often confusing the current board state with a historical one[cite: 11, 13]. This "Time Travel" paradox leads the model to hallucinate pieces in past or future positions.
* **Our Solution:** We implement **Rationale-Guided Retrieval (RAG+)** through our **Eternal FEN Rule**. Our `scanner.py` script performs a 2-pass scan and injects a unique `[%fen "..."]` tag into every move's PGN comment[cite: 35]. The model is governed by a Hard Search Constraint: it is physically forbidden from calculating a past position mentally and must instead retrieve the exact FEN key from the move history to re-ground its tools[cite: 34].
* **Reference:** [Chen et al. (2025a)][ref2]

### 3. Logic Hallucinations and the Need for a "Logic Tutor"
* **The Challenge:** LLMs often favor "correlation completion"—predicting move strings that look linguistically plausible in PGN—over formal correctness[cite: 11, 13]. This results in "line-of-sight hallucinations" where the model attempts illegal moves (like castling through check)[cite: 4]. Standard agentic systems simply reject these moves, causing the LLM to get stuck in an infinite failure loop[cite: 24, 25].
* **Our Solution:** We utilize the **Logic Tutor Paradigm**. Our `execute_coach_move` tool acts as a deterministic intervention layer. When the model proposes a move during sparring, the Python layer validates it against a `Board.is_legal()` verifier[cite: 22]. If a violation is found, the system provides a "causal diagnostic report" derived from a formal logic trace (e.g., explaining why a piece is pinned), which triggers an iterative, self-correcting pedagogical refinement of the AI's plan[cite: 24, 25].
* **Reference:** [OpenReview (2024)][ref3]

### 4. The "Good Test-Taker" Bias vs. Epistemic Uncertainty
* **The Challenge:** Models are typically optimized to act as "good test-takers"—guessing confidently whenever the probability of correctness is non-zero rather than admitting uncertainty[cite: 10, 30, 867]. This results from standard reinforcement learning with verifiable rewards (RLVR) using binary correctness signals, which fail to penalize overconfident errors[cite: 10, 27, 59].
* **Our Solution:** We implement **Behavioral Calibration and Risk Thresholding**. Using the `submit_diagnosis` tool, the model is forced to isolate its "weakest reasoning claim" and score its confidence[cite: 13, 86]. If confidence falls below a strict 0.90 threshold, the diagnosis is rejected as "none," transitioning the model from an overconfident "test-taker" to an "honest communicator"[cite: 11, 13, 16, 33, 955].
* **References:** [Wu et al. (2026)][ref4]; [Kalai et al. (2025)][ref5]

### 5. Computational Burdens in Sequential Reasoning
* **The Challenge:** Strategic reasoning in complex domains introduces unique challenges that are difficult to manage through purely textual, step-by-step reasoning alone[cite: 3535]. Pure LLM reasoning often plateaus far below expert levels due to a deficit in internal understanding of domain-specific rules and global computational dependencies[cite: 1769, 1770, 3533].
* **Our Solution:** We implement **Solver-Informed Reinforcement Learning (SIRL)**. Our framework delegates computationally intensive depth calculations to specialized external solvers (Stockfish, Maia, and Syzygy) through generated tool calls[cite: 1782, 3538]. This division of labor ensures the LLM handles problem comprehension and pedagogy, while the symbolic "Action" component guarantees functional correctness and tactical accuracy[cite: 1, 9, 2697, 2703, 3539].
* **References:** [Hwang et al. (2025)][ref6]; [Chen et al. (2025b)][ref7]

### 6. Semantic Distortion and Concept Extraction
* **The Challenge:** Purely generative commentary often suffers from "semantic distortion," where the model offers superficial advice that is linguistically fluent but strategically vacuous (e.g., hallucinating that a Knight is a permanent outpost when it can be immediately kicked by an enemy pawn)[cite: 1, 35, 37].
* **Our Solution:** We adopt the **Concept-guided Chess Commentary (CCC)** framework. Our `app/concepts.py` engine extracts high-fidelity strategic data, such as mathematically rigorous "Soviet Standard" Knight Outposts and precise King Safety perimeters[cite: 1, 31]. We then enforce a Hard Grounding Constraint in `app/prompts.py`, requiring the AI to explicitly cite this JSON data in its `<think>` block before it is allowed to speak[cite: 1, 21, 43].
* **Reference:** [Kim et al. (2025)][ref8]


[ref1]: https://www.nicowesterdale.com/blog/why-llms-cant-play-chess
[ref2]: https://arxiv.org/html/2506.11555v4
[ref3]: https://openreview.net/pdf/e954744b4cf122cfc82e282e4ac8f33668f1ac5e.pdf
[ref4]: https://arxiv.org/abs/2512.19920
[ref5]: https://arxiv.org/abs/2509.04664
[ref6]: https://arxiv.org/abs/2507.00726
[ref7]: https://arxiv.org/abs/2505.11792
[ref8]: https://arxiv.org/abs/2410.20811
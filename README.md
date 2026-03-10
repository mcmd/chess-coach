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
    * **Pawn Structure defects** (isolated pawns, explicit island spans, passed pawns, doubled files).
    * **Piece Safety** (hanging pieces, pinned pieces).
    * **King Safety Perimeter** (exact pawn shield coordinates, enemy pieces actively attacking the squares around the King).
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
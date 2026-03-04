# Gemini Chess Coach (Agent + Tools Architecture)

## Overview
This project is an advanced AI chess coach that combines the reasoning and conversational capabilities of a Large Language Model (Gemini) with the domain-specific accuracy of dedicated chess engines and databases. 

The core philosophy of this project is its **Agent + Tools Architecture**. While LLMs are excellent at orchestrating complex tasks and translating dense, technical chess concepts into easily understandable, plain-language explanations, they are prone to hallucinations when calculating deep chess lines or evaluating positions. 

To solve this, we give the Gemini agent access to a suite of highly specialized tools. The LLM orchestrates the learning experience, but it grounds its advice in absolute chess truth. By querying Stockfish for objective evaluations, the Maia chess engine for human psychology, Lichess opening explorers for real-world statistics, and Syzygy tablebases for perfect endgame play, the agent provides a personalized, accurate, and deeply insightful coaching experience.

Currently operating as a rich Command Line Interface (CLI) tool, it allows you to analyze specific positions, load PGNs, play moves interactively, and get human-like feedback tailored to your exact Elo rating.

*(Note: While the project structure contains remnants of a Flask web template, such as `main.py` and `src/index.html`, the current primary functionality is executed via the CLI. Future updates may introduce a web-based API.)*

## Key Features
* **Elo-Parametrized Coaching:** The coach adapts its advice, Maia predictions, and Lichess database queries to your exact Elo rating configured in `config.json`.
* **Global Engine Initialization:** Maia and Stockfish engines are initialized globally, reducing latency when loading weights for move predictions and tactical evaluations. Stockfish is now configured to use maximum available threads and Hash memory for superior depth-per-second performance.
* **Brilliancy & Trap Scanner (2-Pass Algorithm):** The system uses a sophisticated 2-pass engine scan on all loaded games. 
    1. **Fast Scan:** Rapidly filters positions to find candidates. 
    2. **Deep Confirmation:** Re-analyzes candidates with higher depth to rigorously verify `[⭐ ONLY MOVE FOUND!]` and `[🛡️ DODGED HUMAN TRAP!]` moments, ensuring GM-level accuracy before tagging them.
* **Token-Based Visual Rendering System:** To prevent the AI from hallucinating piece locations or mangling ASCII formatting, visual boards are now tracked securely on the backend. The AI places simple tokens (e.g., `[[BOARD_1]]`) in its text, which are seamlessly swapped with visually stunning, POV-corrected chess boards right before rendering in the terminal.
* **Context Enrichment:** User inputs are secretly enriched with the current board's FEN and full algebraic move history (e.g., "1. e4 c5..."), ensuring the LLM always has perfect situational awareness of the ongoing game.
* **Clock & Time Analysis:** When loading games via PGN, the coach now analyzes clock timestamps (`[%clk]`). It can identify "Hope Chess" (playing too fast on blunders) or "Freezing" (overthinking forced moves), providing feedback on the student's psychological state during the game.
* **Profiles:** Create isolated learning environments using the `--profile` flag. This separates your configuration (Elo, model), chat memory, and CLI history into independent subdirectories, perfect for studying different openings or having multiple users.
* **Wipe Memory Option:** Quickly reset your conversation history for a fresh start using the `--wipe_memory` flag.
* **Socratic Pedagogy:** The agent is prompted to teach concepts (King Safety, Piece Activity) rather than just spoon-feeding Stockfish lines.
* **Persistent Memory:** The conversation context is saved across sessions (`chess_coach_memory.json`), allowing you to resume your lessons later.
* **Interactive CLI:** Built with `rich` and `prompt_toolkit`, featuring beautiful markdown rendering, color-coded chess notation, and cross-session up-arrow command history.
* **Built-in Commands:** Use `/load_pgn <filename>` to instantly import games for analysis, or `/move <san>` to play moves on the internal board.

## The Agent's Tools
The orchestrating Gemini LLM has autonomous access to the following tools to ground its knowledge:

- **`analyze_position` (Stockfish)**: Interrogates the local Stockfish engine to find the mathematically perfect, objective best moves and position evaluations. It also calculates and returns the first 5 moves of the Principal Variation (PV) for the top 2 engine lines. Furthermore, it supports analyzing hypothetical move sequences by passing in `moves_to_apply`, returning a token to visualize the board after those moves are applied, empowering the AI to clearly answer "What if I play..." questions.
- **`predict_human_move` (Maia Chess + Stockfish)**: Uses pre-trained Maia neural network models (dynamically routing to `maia-1100`, `maia-1500`, or `maia-1900` based on the user's Elo) to predict the most likely human move. Then, it uses Stockfish to analyze the resulting position if that predicted move is played, returning the objective evaluation and a refutation line (punishment).
- **`get_opening_repertoire`**: Queries BOTH the Lichess Masters database and the Amateur database (filtered specifically to the user's Elo bracket). This allows the coach to compare Grandmaster theory against what amateur opponents actually play.
- **`query_endgame_tablebase`**: Accesses the Syzygy 7-piece endgame tablebase. When 7 or fewer pieces are left, the agent uses this tool to get mathematically perfect endgame play (win/draw/loss statuses and exact distance to mate).
- **`fetch_lichess_game_by_id`**: Pulls specific games from Lichess using their game ID, fetching PGNs, clock times, and computer evaluations.
- **`fetch_chessgames_com_game`**: Imports Master-level games (using their numeric ID) from chessgames.com to teach concepts using examples from legends like Fischer or Kasparov.

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

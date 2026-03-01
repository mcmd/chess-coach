# Maia Chess Coach (Agent + Tools Architecture)

## Overview
This project is an advanced AI chess coach that combines the reasoning and conversational capabilities of a Large Language Model (LLM) with the domain-specific accuracy of dedicated chess engines and databases. 

The core philosophy of this project is its **Agent + Tools Architecture**. While LLMs are excellent at orchestrating complex tasks and translating dense, technical chess concepts into easily understandable, plain-language explanations, they are prone to hallucinations when calculating deep chess lines or evaluating positions. 

To solve this, we give the LLM agent access to a suite of highly specialized tools. The LLM orchestrates the learning experience, but it grounds its advice in absolute chess truth. By querying Stockfish for objective evaluations, the Maia chess engine for human psychology and typical mistakes at specific Elo levels, Lichess opening explorers for real-world amateur statistics, and Syzygy tablebases for perfect endgame play, the agent provides a personalized, accurate, and deeply insightful coaching experience.

In short, it provides the "best of all worlds": the super-smart conversational abilities of Gemini, firmly grounded in rigorous chess theory and engine analysis to prevent hallucinations.

Currently operating as a Command Line Interface (CLI) tool, it allows you to analyze specific positions, load PGNs, and get human-like feedback tailored to your exact Elo rating.

*(Note: While the project structure contains remnants of a Flask web template, such as `main.py` and `src/index.html`, the current primary functionality is executed via CLI. Future updates may introduce a web-based API.)*

## The Agent's Tools
The orchestrating LLM (Gemini) has access to the following tools to ground its knowledge and provide accurate coaching:

- **`analyze_position` (Stockfish)**: Interrogates the Stockfish engine to find the mathematically perfect, objective best moves and position evaluations.
- **`predict_human_move` (Maia Chess)**: Uses pre-trained Maia neural network models (`maia-1100`, `maia-1500`, and `maia-1900`) dynamically selected based on the user's Elo. This predicts what a human player at that specific level is actually likely to do, rather than what a computer would do.
- **`get_amateur_opening_data`**: Queries the Lichess explorer database, filtering for the user's specific Elo bracket, to identify typical amateur moves, common traps, and win percentages in the early game.
- **`query_endgame_tablebase`**: Accesses the Syzygy 7-piece endgame tablebase. When 7 or fewer pieces are left on the board, the agent uses this tool to get mathematically perfect endgame play (win/draw/loss statuses and exact distance to mate).
- **`fetch_lichess_game_by_id`**: Allows the agent to pull specific games from Lichess using their game ID, fetching PGNs, clock times, and computer evaluations.
- **`get_fen_from_moves`**: A utility tool for the agent to precisely calculate board states (FEN strings) when evaluating hypothetical move sequences suggested by the user.
- **`print_board`**: Generates a rich text visualization of the current chess board to show the user the position upon request.

## Setup Instructions

### Option 1: Firebase Studio / Project IDX (Recommended)
This project is pre-configured to run seamlessly in a Nix-based environment managed by Firebase Studio.

1. **Open the workspace**: Load the repository into Firebase Studio.
2. **Automatic Configuration**: The Nix environment (`.idx/dev.nix`) will automatically provision the workspace, creating a Python 3 virtual environment (`.venv`) and installing all dependencies from `requirements.txt`.
3. **Terminal Usage**: To run the CLI tool, open the integrated terminal. Before running any `python` scripts or `pip` commands, activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. **Run the Coach**: Execute the coach script:
   ```bash
   python coach.py
   ```

### Option 2: Generic Local Setup (From Scratch)
To set up and run this project locally on any standard machine (Linux, macOS, or Windows), follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd <your-project-directory>
   ```

2. **Create a Virtual Environment**:
   It is highly recommended to isolate the project dependencies.
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. **Install Dependencies**:
   With the virtual environment active, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run Setup Scripts (Optional)**:
   If there are additional environment variables or assets to configure, run the setup script:
   ```bash
   chmod +x setupEnv.sh
   ./setupEnv.sh
   ```

6. **Run the Application**:
   You can now use the CLI application by running the main coaching script:
   ```bash
   python coach.py
   ```
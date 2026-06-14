import argparse
import os
import json
import shutil
from dataclasses import dataclass

@dataclass
class AppConfig:
    user_elo: int
    language: str
    stockfish_path: str
    gemini_model: str
    api_key: str
    history_file: str
    dossier_file: str
    cli_history_file: str
    lesson_file: str
    profile: str = None
    dark_terminal: bool = False
    player_name: str = None  # matched against PGN White/Black headers to find the student's side
    trace: bool = False      # show tool calls and engine invocations
    use_online_databases: bool = True  # opening explorer + tablebase enrichment of reviews

    # Board diagram theme. Square colours are hex strings; board_scale sets how
    # many character cells each square spans (2*scale wide, scale tall).
    board_light_color: str = "#eeeed2"
    board_dark_color: str = "#769656"
    board_scale: int = 2

    # Human-move model (maia2 / maia3). `maia_version`: "auto" prefers maia3 with
    # a maia2 fallback; "3" is maia3-only; "2" forces maia2. Both tiers default to
    # maia3-79M (loaded once, shared); point the scan tier at maia3-23m if the
    # batch trajectory pass ever needs more throughput.
    maia_version: str = "auto"
    maia3_deep_model: str = "maia3-79m"
    maia3_scan_model: str = "maia3-79m"
    maia3_cache_dir: str = None
    maia3_device: str = "cpu"
    maia_history_plies: int = 12      # max game-history plies fed to maia3
    spar_temperature: float = 0.0     # maia3 sparring move-sampling temperature (0 = argmax)

    # Engine quality knobs. Broad scans are cheaper; agenda/threat/practicality
    # analysis is where extra time most directly improves coaching quality.
    review_eval_time: float = 1.0
    agenda_best_depth: int = 20
    agenda_best_time: float = 6.0
    agenda_punishment_depth: int = 18
    agenda_punishment_time: float = 4.0
    threat_depth: int = 18
    threat_time: float = 4.0
    threat_multipv: int = 3
    practicality_depth: int = 16
    practicality_time: float = 3.0
    batch_eval_time: float = 0.6   # Stockfish seconds per position in the /batch fast scan
                                   # (0.6 not 0.4: steadier win% evals reduce borderline
                                   # slide-detection flips run-to-run; see issue #4)
    batch_max_games: int = 50      # cap on games per /batch run
    batch_top_patterns: int = 3    # patterns presented in the training plan
    batch_reps: int = 3            # deep-verified teaching positions per pattern


def _is_dark_terminal() -> bool:
    colorfgbg = os.environ.get("COLORFGBG")
    if colorfgbg:
        try:
            bg = colorfgbg.split(";")[-1]
            if int(bg) < 8: return True
        except: pass
    return False

def load_config() -> AppConfig:

    parser = argparse.ArgumentParser(
        description="Gemini Chess Coach: An AI-powered, tool-augmented chess coach.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python coach.py                  # Runs the coach using the default root configuration and memory.
  python coach.py --profile=mcmd   # Loads (or creates) an isolated configuration and memory for 'mcmd'.
"""
    )
    parser.add_argument("--wipe_memory", action="store_true", help="Clear the chat memory (chess_coach_memory.json) before starting the coach.")
    parser.add_argument(
        "--profile", 
        type=str, 
        help="Name of the profile to load.\n"
             "This isolates your Elo settings, chat memory, and command history.", 
        default=None
    )
    parser.add_argument("--dark_terminal", action="store_true", help="Invert chess Unicode pieces for dark backgrounds.")
    parser.add_argument("--trace", action="store_true", help="Show all tool calls and engine (Stockfish/Maia) invocations.")

    args = parser.parse_args()

    ROOT_CONFIG_FILE = "config.json"
    CONFIG_FILE = ROOT_CONFIG_FILE
    HISTORY_FILE = "chess_coach_memory.json"
    DOSSIER_FILE = "player_dossier.json"
    CLI_HISTORY_FILE = "chess_cli_history.txt"
    LESSON_FILE = "lesson_state.json"

    if not os.path.exists(ROOT_CONFIG_FILE):
        print(f"Error: {ROOT_CONFIG_FILE} not found in the root directory. Please create it first.")
        exit(1)

    with open(ROOT_CONFIG_FILE, "r") as f:
        root_config = json.load(f)

    BASE_PROFILES_DIR = root_config.get("profiles_dir", "./profiles")

    if args.profile:
        profile_dir = os.path.join(BASE_PROFILES_DIR, args.profile)
        os.makedirs(profile_dir, exist_ok=True)
        
        CONFIG_FILE = os.path.join(profile_dir, "config.json")
        HISTORY_FILE = os.path.join(profile_dir, "chess_coach_memory.json")
        DOSSIER_FILE = os.path.join(profile_dir, "player_dossier.json")
        CLI_HISTORY_FILE = os.path.join(profile_dir, "chess_cli_history.txt")
        LESSON_FILE = os.path.join(profile_dir, "lesson_state.json")
        
        if not os.path.exists(CONFIG_FILE):
            print(f"Creating new config for profile '{args.profile}'.")
            shutil.copy(ROOT_CONFIG_FILE, CONFIG_FILE)

    if args.wipe_memory and os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        print(f"Memory wiped: {HISTORY_FILE} has been cleared.")

    with open(CONFIG_FILE, "r") as f:
        app_config = json.load(f)
    engine_config = app_config.get("engine", {})

    def engine_value(name, default):
        return engine_config.get(name, app_config.get(name, default))

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        exit(1)

    return AppConfig(
        user_elo=app_config.get("user_elo", 1100),
        language=app_config.get("language", "English"),
        stockfish_path=app_config.get("stockfish_path"),
        gemini_model=app_config.get("gemini_model", "gemini-3.1-pro-preview"),
        api_key=api_key,
        history_file=HISTORY_FILE,
        dossier_file=DOSSIER_FILE,
        cli_history_file=CLI_HISTORY_FILE,
        lesson_file=LESSON_FILE,
        profile=args.profile,
        dark_terminal=args.dark_terminal or _is_dark_terminal() or app_config.get("dark_terminal", False),
        player_name=app_config.get("player_name"),
        trace=args.trace or app_config.get("trace", False),
        use_online_databases=bool(app_config.get("use_online_databases", True)),
        board_light_color=app_config.get("board_light_color", "#eeeed2"),
        board_dark_color=app_config.get("board_dark_color", "#769656"),
        board_scale=int(app_config.get("board_scale", 2)),
        maia_version=str(app_config.get("maia_version", "auto")),
        maia3_deep_model=app_config.get("maia3_deep_model", "maia3-79m"),
        maia3_scan_model=app_config.get("maia3_scan_model", "maia3-79m"),
        maia3_cache_dir=app_config.get("maia3_cache_dir"),
        maia3_device=app_config.get("maia3_device", "cpu"),
        maia_history_plies=int(engine_value("maia_history_plies", 12)),
        spar_temperature=float(app_config.get("spar_temperature", 0.0)),
        review_eval_time=float(engine_value("review_eval_time", 1.0)),
        agenda_best_depth=int(engine_value("agenda_best_depth", 20)),
        agenda_best_time=float(engine_value("agenda_best_time", 6.0)),
        agenda_punishment_depth=int(engine_value("agenda_punishment_depth", 18)),
        agenda_punishment_time=float(engine_value("agenda_punishment_time", 4.0)),
        threat_depth=int(engine_value("threat_depth", 18)),
        threat_time=float(engine_value("threat_time", 4.0)),
        threat_multipv=int(engine_value("threat_multipv", 3)),
        practicality_depth=int(engine_value("practicality_depth", 16)),
        practicality_time=float(engine_value("practicality_time", 3.0)),
        batch_eval_time=float(engine_value("batch_eval_time", 0.4)),
        batch_max_games=int(app_config.get("batch_max_games", 50)),
        batch_top_patterns=int(app_config.get("batch_top_patterns", 3)),
        batch_reps=int(app_config.get("batch_reps", 3)),
    )

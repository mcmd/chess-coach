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
    lc0_path: str
    maia_weights_dir: str
    gemini_model: str
    api_key: str
    history_file: str
    dossier_file: str
    cli_history_file: str
    profile: str = None

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
    args = parser.parse_args()

    ROOT_CONFIG_FILE = "config.json"
    CONFIG_FILE = ROOT_CONFIG_FILE
    HISTORY_FILE = "chess_coach_memory.json"
    DOSSIER_FILE = "player_dossier.json"
    CLI_HISTORY_FILE = "chess_cli_history.txt"

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
        
        if not os.path.exists(CONFIG_FILE):
            print(f"Creating new config for profile '{args.profile}'.")
            shutil.copy(ROOT_CONFIG_FILE, CONFIG_FILE)

    if args.wipe_memory and os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        print(f"Memory wiped: {HISTORY_FILE} has been cleared.")

    with open(CONFIG_FILE, "r") as f:
        app_config = json.load(f)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        exit(1)

    return AppConfig(
        user_elo=app_config.get("user_elo", 1100),
        language=app_config.get("language", "English"),
        stockfish_path=app_config.get("stockfish_path"),
        lc0_path=app_config.get("lc0_path"),
        maia_weights_dir=app_config.get("maia_weights_dir", "./maia_weights"),
        gemini_model=app_config.get("gemini_model", "gemini-3.1-pro-preview"),
        api_key=api_key,
        history_file=HISTORY_FILE,
        dossier_file=DOSSIER_FILE,
        cli_history_file=CLI_HISTORY_FILE,
        profile=args.profile
    )
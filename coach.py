import logging
import os

from dotenv import load_dotenv
from google import genai
from prompt_toolkit import prompt as cli_prompt, HTML
from prompt_toolkit.history import FileHistory
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from app.agent_loop import run_agent_turn
from app.commands import CommandProcessor
from app.config import load_config
from app.engines import EngineManager, EngineStartError
from app.memory import load_history, save_history
from app.prompts import create_chat_session
from app.state import SessionState, TimelineManager
from app.tools import CoachTools
from app.ui import console
from app import trace

os.environ["PROMPT_TOOLKIT_NO_CPR"] = "1"

# python-chess warns (harmlessly) about null-move history during threat probes.
logging.getLogger("chess.engine").setLevel(logging.ERROR)

load_dotenv()


class _ReviewProgress:
    """Adapter: review_game(on_progress=...) -> rich progress bar."""

    def __init__(self):
        self._progress = None
        self._task = None

    def __call__(self, done, total):
        if self._progress is None:
            self._progress = Progress(
                TextColumn("[bold yellow]⏳ Reviewing moves...[/bold yellow]"),
                BarColumn(bar_width=None), TaskProgressColumn(), TimeRemainingColumn(),
                console=console, transient=True,
            )
            self._progress.start()
            self._task = self._progress.add_task("", total=total)
        self._progress.update(self._task, completed=done)
        if done >= total:
            self._progress.stop()
            self._progress = None


def main():
    config = load_config()
    trace.set_enabled(config.trace)
    client = genai.Client(api_key=config.api_key)

    app_state = SessionState()
    timeline = TimelineManager(app_state)

    engines = EngineManager(config)
    try:
        engines.start()
    except EngineStartError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1
    if engines.maia is None and engines.maia_error:
        console.print(f"[yellow]Warning: Maia-2 unavailable ({engines.maia_error}). "
                      "Human-move prediction disabled.[/yellow]")

    review_progress = _ReviewProgress()
    coach_tools = CoachTools(app_state, timeline, engines, config, on_progress=review_progress)
    chat = create_chat_session(client, config, coach_tools, load_history(config.history_file))
    commands = CommandProcessor(app_state, timeline, engines, config, on_progress=review_progress)

    profile_text = f" (Profile: {config.profile})" if config.profile else ""
    trace_text = "  [dim cyan](trace ON)[/dim cyan]" if config.trace else ""
    console.print(Panel.fit(
        f"[bold green]Chess Coach (Elo: {config.user_elo}){profile_text}[/bold green]{trace_text}\n"
        f"Type [bold cyan]/help[/bold cyan] for commands, [bold red]exit[/bold red] to save and close.",
        border_style="green"))

    try:
        while True:
            console.print()
            user_input = cli_prompt(HTML("<b><ansiblue>You:</ansiblue></b> "),
                                    history=FileHistory(config.cli_history_file))
            if user_input.strip().lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue

            result = commands.process(user_input)
            if result.clear_screen:
                console.clear()
            if result.reset_chat:
                chat = create_chat_session(client, config, coach_tools, [])
            if result.message:
                console.print(result.message)
            if not result.prompt:
                continue

            try:
                # The spinner lives inside the agent loop now: it covers model
                # calls only, so tools (e.g. a full game review) can render
                # their own progress bars without fighting Rich's single-live rule.
                run_agent_turn(chat, coach_tools, result.prompt)
            except Exception as e:
                console.print(f"\n[bold red]An API error occurred:[/bold red] {e}")

    except KeyboardInterrupt:
        pass
    finally:
        save_history(chat, config.history_file)
        engines.quit()
        console.print(f"\n[bold dim]Conversation saved to {config.history_file}. Goodbye![/bold dim]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

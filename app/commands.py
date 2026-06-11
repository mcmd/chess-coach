"""Slash-command and chat-input processing.

Extracted from coach.py so the command surface is importable and testable.
Returns CommandResult objects instead of mutating the chat session directly.
"""
import io
import os
import random
from dataclasses import dataclass

import chess
import chess.pgn

from app.memory import load_dossier, update_dossier
from app.review import review_game, attach_board_tokens

HELP_TEXT = """[bold]Commands[/bold]
  /load_pgn <path>   Load + review a PGN file (full engine review, lesson agenda)
  /move <san>        Play a move on the board (e.g. /move Nf3)
  /next  /prev       Step forward / back through the loaded game
  /goto <n>[b]       Jump to before White's (or Black's, with 'b') move n
  /reset             Rewind to the start of the loaded game (keeps the game)
  /spar  /nospar     Enable / disable sparring mode
  /review            Quiz yourself on a past mistake from your dossier
  /memorize          Save this session's diagnoses to your long-term dossier
  /clear             Wipe board, review, and the coach's conversation memory
  /help              Show this help
You can also paste a full PGN directly, or just talk to the coach."""


@dataclass
class CommandResult:
    prompt: str = None        # text to send to the model (None = nothing to send)
    message: str = None       # rich-markup status message to show the user
    reset_chat: bool = False  # caller should recreate the chat session
    clear_screen: bool = False


def looks_like_pgn(text: str) -> bool:
    """Strict PGN sniffing: a header tag, or 3+ parseable mainline moves.
    Prevents chat like 'why is 1. e4 best?' from clobbering the board."""
    if "[Event " in text:
        return True
    if "1." not in text:
        return False
    try:
        game = chess.pgn.read_game(io.StringIO(text))
    except Exception:
        return False
    return game is not None and len(list(game.mainline_moves())) >= 3


class CommandProcessor:
    def __init__(self, state, timeline, engines, config, on_progress=None):
        self.state = state
        self.timeline = timeline
        self.engines = engines
        self.config = config
        self.on_progress = on_progress  # optional callable(done, total) for review progress

    # ------------------------------------------------------------------

    def process(self, user_input: str) -> CommandResult:
        text = user_input.strip()

        if text.startswith("/help"):
            return CommandResult(message=HELP_TEXT)
        if text.startswith("/load_pgn"):
            return self._load_pgn(text)
        if text.startswith("/move"):
            return self._move(text)
        if text.startswith("/undo") or text.startswith("/prev"):
            return self._step_back()
        if text.startswith("/next"):
            return self._step_forward()
        if text.startswith("/goto"):
            return self._goto(text)
        if text.startswith("/reset"):
            return self._reset()
        if text.startswith("/spar"):
            self.state.sparring_mode = True
            return CommandResult(prompt="I am ready to spar! Play a move.",
                                 message="[bold green]🥊 Sparring mode ENABLED.[/bold green]")
        if text.startswith("/nospar"):
            self.state.sparring_mode = False
            return CommandResult(message="[bold yellow]🛑 Sparring mode DISABLED.[/bold yellow]")
        if text.startswith("/review"):
            return self._dossier_quiz()
        if text.startswith("/clear"):
            return self._clear()
        if text.startswith("/memorize"):
            return self._memorize()

        if looks_like_pgn(text):
            return self._load_pasted_pgn(text)

        return CommandResult(prompt=self._enrich(text))

    # ------------------------------------------------------------------
    # GAME LOADING & REVIEW
    # ------------------------------------------------------------------

    def _load_pgn(self, text: str) -> CommandResult:
        parts = text.split(" ", 1)
        if len(parts) < 2:
            return CommandResult(message="Please provide a file path: /load_pgn game.pgn")
        file_path = parts[1].strip().strip('"')
        if not os.path.exists(file_path):
            return CommandResult(message=f"[red]Could not find the file: {file_path}[/red]")
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                pgn_text = f.read()
        except OSError as e:
            return CommandResult(message=f"[red]Error reading PGN: {e}[/red]")
        return self._review_and_load(pgn_text, source=file_path)

    def _load_pasted_pgn(self, text: str) -> CommandResult:
        return self._review_and_load(text, source="pasted game")

    def _review_and_load(self, pgn_text: str, source: str) -> CommandResult:
        try:
            review = review_game(pgn_text, self.engines, self.config,
                                 on_progress=self.on_progress)
        except Exception as e:
            return CommandResult(message=f"[red]Game review failed: {e}[/red]")
        if review is None:
            return CommandResult(message=f"[red]Could not parse a game with moves from {source}.[/red]")

        self.timeline.load_game(review.game)
        self.state.review = review
        self.state.socratic_attempts.clear()
        attach_board_tokens(review, self.state, invert=getattr(self.config, "dark_terminal", False))

        n = len(review.agenda_plies)
        prompt = (review.digest()
                  + "\n\n[System: The game is loaded with the board at the STARTING position. "
                    "The student steps through moves with /next. Greet the student with a one-paragraph "
                    "summary of the game and the agenda, then suggest starting with Moment 1. "
                    "Do not analyze anything yet.]")
        message = (f"[green]Reviewed {source}: {len(review.records)} moves analyzed, "
                   f"{n} critical moment(s) found. Board set to the start.[/green]")
        return CommandResult(prompt=self._enrich(prompt), message=message)

    # ------------------------------------------------------------------
    # BOARD NAVIGATION
    # ------------------------------------------------------------------

    def _move(self, text: str) -> CommandResult:
        parts = text.split(" ", 1)
        if len(parts) < 2:
            return CommandResult(message="[red]Please provide a move: /move e4[/red]")
        move_san = parts[1].strip()
        fen_before = self.state.board.fen()
        try:
            clean_san = self.timeline.play_san(move_san)
        except ValueError:
            return CommandResult(message="[red]Invalid or illegal move.[/red]")

        swing_note = self._eval_swing_note(fen_before, self.state.board.fen())
        if self.state.sparring_mode:
            prompt = (f"I played {clean_san}.{swing_note} "
                      "Reply by calling play_sparring_reply (the backend picks your move), "
                      "then comment on both moves in 1-3 sentences. If my move lost significant "
                      "win%, pause the game commentary and coach me on it first.")
        else:
            prompt = f"I have played {clean_san}.{swing_note} The new position FEN is: {self.state.board.fen()}. Please evaluate."
        return CommandResult(prompt=self._enrich(prompt),
                             message=f"[green]Played {clean_san}. Board updated.[/green]")

    def _step_back(self) -> CommandResult:
        try:
            self.timeline.step_backward()
        except IndexError:
            return CommandResult(message="[red]No moves to undo![/red]")
        prompt = f"I took back the last move. The FEN is now: {self.state.board.fen()}. What should be played here instead?"
        return CommandResult(prompt=self._enrich(prompt), message="[green]Move taken back. Board updated.[/green]")

    def _step_forward(self) -> CommandResult:
        try:
            move_san, time_str = self.timeline.step_forward()
        except IndexError:
            return CommandResult(message="[red]You are at the end of the loaded game (or no game is loaded)![/red]")

        prompt = (f"I stepped forward: the game move {move_san} was played.{time_str} "
                  f"The position FEN is now: {self.state.board.fen()}.")
        prompt += self._record_note(self.state.move_index - 1)
        return CommandResult(prompt=self._enrich(prompt),
                             message=f"[green]Stepped forward: played {move_san}. Board updated.[/green]")

    def _goto(self, text: str) -> CommandResult:
        parts = text.split(" ", 1)
        if len(parts) < 2:
            return CommandResult(message="[red]Usage: /goto 12 (before White's 12th) or /goto 12b (before Black's).[/red]")
        arg = parts[1].strip().lower()
        black = arg.endswith("b")
        num_str = arg.rstrip("wb")
        if not num_str.isdigit():
            return CommandResult(message="[red]Usage: /goto 12 or /goto 12b[/red]")
        ply = (int(num_str) - 1) * 2 + (1 if black else 0)
        try:
            self.timeline.go_to_ply(ply)
        except IndexError:
            return CommandResult(message="[red]That move number is outside the loaded game.[/red]")
        side = "Black" if black else "White"
        prompt = (f"I jumped to the position BEFORE {side}'s move {num_str}. "
                  f"The FEN is: {self.state.board.fen()}.")
        record = self.state.review.record_at(ply) if self.state.review else None
        if record and record.is_agenda:
            prompt += self._record_note(ply, before_move=True)
        return CommandResult(prompt=self._enrich(prompt),
                             message=f"[green]Jumped to before {side}'s move {num_str}.[/green]")

    def _reset(self) -> CommandResult:
        self.timeline.reset()
        if self.state.loaded_moves:
            prompt = ("I rewound the loaded game to its starting position. "
                      "I'll step through it with /next.")
            message = "[green]Rewound to the start of the loaded game (game kept — use /next).[/green]"
        else:
            prompt = "I have reset the board to the starting position. What opening should we look at?"
            message = "[green]Board reset to the starting position.[/green]"
        return CommandResult(prompt=self._enrich(prompt), message=message)

    # ------------------------------------------------------------------
    # SESSION MANAGEMENT
    # ------------------------------------------------------------------

    def _clear(self) -> CommandResult:
        self.timeline.clear()
        if self.engines:
            try:
                self.engines.clear_cache()
            except Exception:
                pass
        if os.path.exists(self.config.history_file):
            try:
                os.remove(self.config.history_file)
            except OSError:
                pass
        return CommandResult(message="[bold green]✨ Session memory, board state, and timeline wiped clean![/bold green]",
                             reset_chat=True)

    def _memorize(self) -> CommandResult:
        if not self.state.diagnostics:
            return CommandResult(message="[yellow]No diagnostics generated this session to memorize.[/yellow]")
        # Enrich diagnoses with backend typicality before persisting, so the
        # dossier can track the typical->atypical progress trend.
        if self.state.review:
            by_fen = {r.fen_before: r for r in self.state.review.records}
            for diag in self.state.diagnostics:
                record = by_fen.get(diag.get("fen"))
                if record and record.error_typicality and not diag.get("error_typicality"):
                    diag["error_typicality"] = record.error_typicality
        update_dossier(self.state.diagnostics, self.config.dossier_file)
        self._clear()  # wipes timeline, cache, and history file
        return CommandResult(message="[bold green]✨ Session memorized! Dossier updated. Memory wiped and screen cleared.[/bold green]",
                             reset_chat=True, clear_screen=True)

    def _dossier_quiz(self) -> CommandResult:
        dossier = load_dossier(self.config.dossier_file)
        positions = dossier.get("recent_positions", [])
        if not positions:
            return CommandResult(message="[yellow]No stored mistakes to review yet. Use /memorize after a session to build your dossier.[/yellow]")
        pick = random.choice(positions)
        prompt = (f"DOSSIER REVIEW DRILL. I previously made a '{pick.get('error_category', 'unknown')}' mistake "
                  f"in this position: {pick['fen']}. Set the scene with analyze_position (do not reveal the answer), "
                  "remind me what KIND of mistake I tend to make here, and quiz me: ask me to find the best move. "
                  "Grade my answer with evaluate_student_answer.")
        return CommandResult(prompt=self._enrich(prompt),
                             message="[green]Pulled a position from your dossier. Quiz time![/green]")

    # ------------------------------------------------------------------
    # CONTEXT ENRICHMENT
    # ------------------------------------------------------------------

    def _record_note(self, ply: int, before_move: bool = False) -> str:
        """Backend review data for the move at `ply`, appended to /next and /goto prompts."""
        review = self.state.review
        if not review:
            return ""
        record = review.record_at(ply)
        if not record:
            return ""
        if before_move:
            note = (f"\n[Backend review: the move about to be played here is {record.move_label} "
                    f"({record.quality}, loses {record.drop_pct} win%). This is an AGENDA moment — "
                    "coach it per the protocol: ask the student ONE question before any analysis is revealed. "
                    f"FEN before the move: {record.fen_before}]")
            return note
        note = (f"\n[Backend review of this move: {record.quality}"
                + (f", lost {record.drop_pct} win%" if record.drop_pct > 0 else "")
                + (f", time flag: {record.time_flag}" if record.time_flag else "")
                + ".")
        if record.is_agenda:
            note += (" This is an AGENDA moment from the digest. Coach it now: state what changed, "
                     "ask the student ONE focused question (the threat, or the better move), and use "
                     f"evaluate_student_answer on FEN {record.fen_before} to grade their answer. "
                     "Do NOT reveal the engine line until reveal_allowed is true.")
        elif record.quality in ("good", "best", "only_move"):
            note += " The move was objectively fine — if the student played it, say so honestly."
        if record.brilliancy_note:
            note += (f" BRILLIANCY: {record.brilliancy_note}. Celebrate this specifically — "
                     "it is the strongest praise moment a review can produce.")
        if record.avoided_natural_mistake:
            note += (f" PRAISE MOMENT: {record.avoided_natural_mistake}. "
                     "Give genuine, specific credit for sidestepping it.")
        if record.error_typicality == "typical_for_level":
            note += " This mistake is common at the student's level: normalize first, then teach the pattern."
        elif record.error_typicality == "atypical_for_level":
            note += (" Players at this level rarely make this mistake: coach the process "
                     "(time, attention), not the pattern.")
        note += "]"
        return note

    def _eval_swing_note(self, fen_before: str, fen_after: str) -> str:
        if not (self.engines and getattr(self.engines, "stockfish", None)):
            return ""
        try:
            from app.review import cp_to_win_pct, MATE_CP
            before_score, _ = self.engines.quick_eval(fen_before)
            after_score, _ = self.engines.quick_eval(fen_after)
            mover = chess.Board(fen_before).turn
            before_cp = before_score.pov(mover).score(mate_score=MATE_CP)
            after_cp = after_score.pov(mover).score(mate_score=MATE_CP)
            drop = cp_to_win_pct(before_cp) - cp_to_win_pct(after_cp)
            b_w = before_score.white().score(mate_score=MATE_CP) / 100.0
            a_w = after_score.white().score(mate_score=MATE_CP) / 100.0
            note = f" [Backend quick eval: {b_w:+.2f} -> {a_w:+.2f}"
            if drop >= 10:
                note += f"; the move lost ~{drop:.0f} win% — treat it as a mistake and coach it"
            note += ".]"
            return note
        except Exception:
            return ""

    def _enrich(self, prompt_text: str) -> str:
        board = self.state.board
        try:
            if board.move_stack:
                temp_board = board.copy()
                num_moves = min(5, len(temp_board.move_stack))
                show_moves = temp_board.move_stack[-num_moves:]
                for _ in range(num_moves):
                    temp_board.pop()

                history_parts = []
                for m in show_moves:
                    san = temp_board.san(m)
                    if temp_board.is_capture(m):
                        if temp_board.is_en_passant(m):
                            san += " (captures pawn)"
                        else:
                            cap_piece = temp_board.piece_at(m.to_square)
                            if cap_piece:
                                san += f" (captures {chess.piece_name(cap_piece.piece_type)})"
                    if temp_board.turn == chess.WHITE:
                        history_parts.append(f"{temp_board.fullmove_number}. {san}")
                    elif not history_parts:
                        history_parts.append(f"{temp_board.fullmove_number}... {san}")
                    else:
                        history_parts.append(san)
                    temp_board.push(m)
                history_str = " ".join(history_parts)
            else:
                history_str = "Starting Position"
        except Exception:
            history_str = "Unknown"

        legal_moves_str = ", ".join(board.san(m) for m in board.legal_moves)
        return (f"{prompt_text}\n\n[System Context: Current FEN: {board.fen()} | "
                f"Legal Moves: {legal_moves_str} | Move History: {history_str}]")

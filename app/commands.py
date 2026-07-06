"""Slash-command and chat-input processing.

Extracted from coach.py so the command surface is importable and testable.
Returns CommandResult objects instead of mutating the chat session directly.
"""
import contextlib
import hashlib
import io
import os
import random
import re
from dataclasses import dataclass

import chess
import chess.pgn

from app.memory import (load_dossier, load_lesson_metadata, save_lesson_metadata,
                        record_session_takeaway, update_dossier, update_dossier_with_batch)
from app.notation import legal_san_mentions
from app.planner import TeachingPlan, plan_free_text_turn
from app.review import adopt_review_pov, review_game, attach_board_tokens
from app.trace import trace
from app.ui import console

HELP_TEXT = """[bold]Commands[/bold]
  /load_pgn <path>   Load + review a PGN file (full engine review, lesson agenda)
  /batch <path>      Scan a multi-game PGN (10-50 games) for recurring mistake patterns
                     and train on them (/batch alone reloads the last report)
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
    move_numbers = re.findall(r"\b\d+\.(?:\.\.)?\s*[A-Za-z0-9O+#=x-]+", text)
    if len(move_numbers) < 3:
        return False
    try:
        with contextlib.redirect_stderr(io.StringIO()):
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
        # Match the EXACT first token, not a prefix: startswith made "/review-old",
        # "/movefoo", "/sparring" trigger commands they only resemble.
        cmd = text.split(maxsplit=1)[0].lower() if text else ""

        if cmd == "/help":
            return CommandResult(message=HELP_TEXT)
        if cmd == "/load_pgn":
            return self._load_pgn(text)
        if cmd == "/batch":
            return self._batch(text)
        if cmd == "/move":
            return self._move(text)
        if cmd in ("/undo", "/prev"):
            return self._step_back()
        if cmd == "/next":
            return self._step_forward()
        if cmd == "/goto":
            return self._goto(text)
        if cmd == "/reset":
            return self._reset()
        if cmd == "/spar":
            self.state.sparring_mode = True
            return CommandResult(prompt="I am ready to spar! Play a move.",
                                 message="[bold green]🥊 Sparring mode ENABLED.[/bold green]")
        if cmd == "/nospar":
            self.state.sparring_mode = False
            return CommandResult(message="[bold yellow]🛑 Sparring mode DISABLED.[/bold yellow]")
        if cmd == "/review":
            return self._dossier_quiz()
        if cmd == "/clear":
            return self._clear()
        if cmd == "/memorize":
            return self._memorize()

        if self.state.pending_unmatched_review:
            return self._resolve_unmatched_review(text)

        if looks_like_pgn(text):
            return self._load_pasted_pgn(text)

        plan = plan_free_text_turn(text, default_language=getattr(self.config, "language", "English"))
        if self._completion_ack_requests_close(text, plan):
            plan = TeachingPlan(
                intent="close_with_takeaway",
                language=getattr(self.config, "language", "English"),
                density="brief",
                no_tail_redirect=True,
                reasons=["intent:close_after_lesson_complete"],
            )
        self.state.teaching_ledger["last_plan"] = {
            "intent": plan.intent,
            "language": plan.language,
            "density": plan.density,
            "withhold_answer": plan.withhold_answer,
            "socratic_redirect": plan.socratic_redirect,
            "no_tail_redirect": plan.no_tail_redirect,
            "decline": plan.decline,
            "reasons": list(plan.reasons),
        }
        trace(f"teaching plan: {self.state.teaching_ledger['last_plan']}")
        if plan.decline:
            return CommandResult(message=plan.decline_message)

        planned = f"{plan.prompt_block()}\n\nStudent request: {text}"
        planned += self._self_assessment_response_note(text, plan)
        planned += self._redundancy_note(plan)
        if plan.intent == "close_with_takeaway":
            planned += self._takeaway_note()
        return CommandResult(prompt=self._enrich(planned) + self._candidate_list_note(text)
                             + self._give_up_note(text) + self._pending_diagnosis_note())

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

    # ------------------------------------------------------------------
    # BATCH PATTERN REVIEW
    # ------------------------------------------------------------------

    def _batch(self, text: str) -> CommandResult:
        from app.batch import batch_dossier_summary, load_latest_batch, run_batch_review
        parts = text.split(" ", 1)
        if len(parts) < 2 or not parts[1].strip():
            batch = load_latest_batch(self.config)
            if batch is None:
                return CommandResult(message="No saved batch report yet. Usage: /batch <path-to.pgn>")
            return self._load_batch(batch, f"[green]Reloaded batch report {batch.batch_id} "
                                           f"({batch.games_analyzed} games).[/green]")
        path = parts[1].strip().strip('"')
        if not os.path.exists(path):
            return CommandResult(message=f"[red]Could not find the file: {path}[/red]")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                pgn_text = f.read()
        except OSError as e:
            return CommandResult(message=f"[red]Error reading PGN: {e}[/red]")
        console.print("[dim yellow]↳ Batch scan starting (cached games are reused; "
                      "interrupting and rerunning is safe)...[/dim yellow]")
        try:
            batch = run_batch_review(
                pgn_text, self.engines, self.config, on_progress=self.on_progress,
                on_status=lambda s: console.print(f"[dim]· batch | {s}[/dim]"))
        except Exception as e:
            return CommandResult(message=f"[red]Batch review failed: {e}[/red]")
        if isinstance(batch, dict):
            return CommandResult(message=f"[red]{batch['error']}[/red]")
        update_dossier_with_batch(self.config.dossier_file, batch_dossier_summary(batch))
        return self._load_batch(
            batch, f"[green]Batch report ready: {batch.games_analyzed} games, "
                   f"{batch.moments_found} mistake moments, "
                   f"{len(batch.patterns)} recurring pattern(s).[/green]")

    def _load_batch(self, batch, message: str) -> CommandResult:
        from app.batch import attach_batch_board_tokens
        meta = dict(self.state.lesson_metadata or {})
        resume_idx = None
        if (meta.get("batch_id") == batch.batch_id and batch.records
                and meta.get("batch_lesson_status") not in (None, "", "unavailable",
                                                            "lesson_complete", "fresh_review")):
            resume_idx = max(0, min(int(meta.get("batch_agenda_index") or 0),
                                    len(batch.records) - 1))
        self.state.review = batch
        self.state.current_game_id = f"batch:{batch.batch_id}"
        self.timeline.initial_fen = chess.STARTING_FEN
        self.state.board = chess.Board()
        self.state.loaded_moves = []
        self.state.move_index = 0
        self.timeline.pgn_nodes = []
        self.state.socratic_attempts.clear()
        attach_batch_board_tokens(batch, self.state,
                                  invert=getattr(self.config, "dark_terminal", False))
        meta["batch_id"] = batch.batch_id
        if resume_idx is not None:
            meta["batch_agenda_index"] = resume_idx
            meta["batch_lesson_status"] = "awaiting_answer"
        else:
            meta["batch_agenda_index"] = 0
            meta["batch_lesson_status"] = "fresh_review"
        self.state.lesson_metadata = meta
        if getattr(self.config, "lesson_file", None):
            save_lesson_metadata(self.config.lesson_file, meta)
        prompt = batch.digest()
        if resume_idx is not None:
            record = batch.records[resume_idx]
            prompt += (f"\n\n(The student is RESUMING this batch lesson at teaching position "
                       f"{resume_idx + 1} ({record.move_label}). Welcome them back briefly and "
                       "continue that position — do not re-present the whole training plan.)")
        else:
            prompt += ("\n\n(The student just ran /batch. Present the training plan now: "
                       "strengths first, then the recurring patterns with their counts, then "
                       "recommend Pattern 1 and ask which pattern they want to work on. "
                       "Do not start teaching a position yet.)")
        return CommandResult(prompt=prompt, message=message)

    @staticmethod
    def _local_game_id(review) -> str:
        """Stable identity for local/pasted games so lessons can resume."""
        movetext = " ".join(r.uci for r in review.records)
        return "local-" + hashlib.sha1(movetext.encode()).hexdigest()[:12]

    def _review_and_load(self, pgn_text: str, source: str) -> CommandResult:
        try:
            review = review_game(pgn_text, self.engines, self.config,
                                 on_progress=self.on_progress)
        except Exception as e:
            return CommandResult(message=f"[red]Game review failed: {e}[/red]")
        if review is None:
            return CommandResult(message=f"[red]Could not parse a game with moves from {source}.[/red]")

        if review.student_color is None:
            self.timeline.clear()
            self.state.pending_unmatched_review = {
                "review": review,
                "source": source,
                "source_game_id": None,
            }
            self.state.review_context = {
                "ownership": "unmatched_pending",
                "dossier_consent": False,
            }
            white = review.headers.get("White", "?")
            black = review.headers.get("Black", "?")
            return CommandResult(message=(
                "[yellow]I don't recognize either player as one of your configured usernames.[/yellow]\n"
                f"White: {white} | Black: {black}\n"
                "Reply with `white` or `black` to choose the POV for this review. "
                "By default, nothing from this game will be saved to your personal dossier. "
                "Add `save` if you explicitly want this borrowed game logged, e.g. `black save`."
            ))

        return self._activate_review(review, source, ownership="own",
                                     dossier_consent=True)

    def _activate_review(self, review, source: str, *, ownership: str,
                         dossier_consent: bool, source_game_id: str = None) -> CommandResult:

        self.state.pending_unmatched_review.clear()
        self.timeline.load_game(review.game)
        self.state.review = review
        self.state.socratic_attempts.clear()
        attach_board_tokens(review, self.state, invert=getattr(self.config, "dark_terminal", False))

        game_id = source_game_id or self._local_game_id(review)
        self.state.current_game_id = game_id
        self.state.review_context = {
            "ownership": ownership,
            "pov": review.student_color,
            "dossier_consent": bool(dossier_consent),
        }
        lesson_file = getattr(self.config, "lesson_file", None)
        meta = dict(self.state.lesson_metadata or load_lesson_metadata(lesson_file))
        resumable = (meta.get("last_game_id") == game_id
                     and meta.get("lesson_status") not in (None, "", "unavailable",
                                                           "lesson_complete", "fresh_review")
                     and review.agenda_plies)

        n = len(review.agenda_plies)
        if resumable:
            idx = max(0, min(int(meta.get("last_agenda_index") or 0), n - 1))
            self.timeline.go_to_ply(review.agenda_plies[idx])
            record = review.agenda[idx]
            meta.update({"last_game_id": game_id, "last_agenda_index": idx,
                         "lesson_status": "awaiting_answer"})
            prompt = (review.digest()
                      + f"\n\n[System: This game was recognized from a previous session. The lesson "
                        f"RESUMES at Moment {idx + 1} ({record.move_label}); the board is already at "
                        "that position. Greet the student briefly, remind them where the lesson left "
                        "off, and continue that moment. The student can /reset to restart from move 1. "
                        "Do not re-analyze earlier moments.]")
            message = (f"[green]Recognized {source} from a previous session — resumed at "
                       f"Moment {idx + 1} ({record.move_label}). Use /reset to start over.[/green]")
        else:
            meta.update({"last_game_id": game_id, "last_agenda_index": 0,
                         "lesson_status": "fresh_review"})
            self._start_self_assessment(review)
            prompt = (review.digest()
                      + "\n\n[System: The game is loaded with the board at the STARTING position. "
                        "The student steps through moves with /next. Greet the student with a one-paragraph "
                        "summary of the game. If the backend self-assessment note is present, ask that "
                        "single question before revealing the lesson agenda or starting Moment 1. "
                        "Do not analyze anything yet.]")
            prompt += self._borrowed_game_note()
            prompt += self._self_assessment_open_note(review)
            message = (f"[green]Reviewed {source}: {len(review.records)} moves analyzed, "
                       f"{n} critical moment(s) found. Board set to the start.[/green]")

        self.state.lesson_metadata = meta
        if lesson_file:
            save_lesson_metadata(lesson_file, meta)
        return CommandResult(prompt=self._enrich(prompt), message=message)

    _POV_WHITE_RE = re.compile(r"\b(?:white|w)\b", re.IGNORECASE)
    _POV_BLACK_RE = re.compile(r"\b(?:black|b)\b", re.IGNORECASE)
    _CONSENT_SAVE_RE = re.compile(r"\b(?:save|log|memorize|remember|dossier|yes)\b", re.IGNORECASE)
    _CONSENT_NO_RE = re.compile(r"\b(?:don'?t\s+save|do\s+not\s+save|no\s+save|no\s+dossier|borrowed only)\b",
                                re.IGNORECASE)

    def _resolve_unmatched_review(self, text: str) -> CommandResult:
        pending = self.state.pending_unmatched_review or {}
        review = pending.get("review")
        if review is None:
            self.state.pending_unmatched_review.clear()
            return CommandResult(message="[red]No pending unmatched review is available. Load the game again.[/red]")

        color = None
        if self._POV_WHITE_RE.search(text or ""):
            color = "white"
        if self._POV_BLACK_RE.search(text or ""):
            if color:
                return CommandResult(message="Please choose one POV only: `white` or `black`.")
            color = "black"
        if color is None:
            white = review.headers.get("White", "?")
            black = review.headers.get("Black", "?")
            return CommandResult(message=(
                f"Please choose the POV for this unmatched game: `white` ({white}) or `black` ({black}). "
                "Add `save` only if you want this borrowed game written to your personal dossier."
            ))

        consent = bool(self._CONSENT_SAVE_RE.search(text or ""))
        if self._CONSENT_NO_RE.search(text or ""):
            consent = False

        try:
            review = adopt_review_pov(review, color, self.engines, self.config)
        except Exception as e:
            self.state.pending_unmatched_review.clear()
            return CommandResult(message=f"[red]Could not adopt that POV: {e}[/red]")

        source = pending.get("source") or "borrowed game"
        source_game_id = pending.get("source_game_id")
        self.state.pending_unmatched_review.clear()
        result = self._activate_review(
            review, source,
            ownership="borrowed",
            dossier_consent=consent,
            source_game_id=source_game_id,
        )
        consent_msg = ("Dossier saving is ON for this borrowed game because you opted in."
                       if consent else
                       "Dossier saving is OFF for this borrowed game. Use `save` when choosing POV next time if you want opt-in logging.")
        result.message = (result.message or "") + f"\n[cyan]Using {color.upper()} POV. {consent_msg}[/cyan]"
        return result

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
        lesson_file = getattr(self.config, "lesson_file", None)
        if lesson_file and os.path.exists(lesson_file):
            try:
                os.remove(lesson_file)
            except OSError:
                pass
        self.state.lesson_metadata.clear()
        self.state.pending_unmatched_review.clear()
        self.state.review_context.clear()
        return CommandResult(message="[bold green]✨ Session memory, board state, and timeline wiped clean![/bold green]",
                             reset_chat=True)

    def _memorize(self) -> CommandResult:
        if not self.state.diagnostics:
            return CommandResult(message="[yellow]No diagnostics generated this session to memorize.[/yellow]")
        if not self._dossier_write_allowed():
            return CommandResult(message=(
                "[yellow]This is a borrowed/unmatched game and dossier saving was not opted in. "
                "Nothing was written to your personal dossier.[/yellow]"
            ))
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

    def _candidate_list_note(self, text: str) -> str:
        """The model can cherry-pick one move out of a multi-candidate answer
        and grade it (observed live: 'd5, Rad1 or Qd3' -> graded Rad1, praised).
        The command layer sees the student's FULL text, so it injects the
        verified candidate list — making the hedge impossible to ignore."""
        if not (self.state.review or self.state.sparring_mode):
            return ""
        boards = [self.state.board.copy(stack=False)]
        if self.state.review:
            boards.extend(chess.Board(r.fen_before) for r in self.state.review.agenda)
        best = []
        for board in boards:
            mentions = legal_san_mentions(board, text)
            if len(mentions) > len(best):
                best = mentions
        if len(best) < 2:
            return ""
        return ("\n[Backend: the student's message mentions multiple legal candidate moves: "
                + ", ".join(best[:6]) +
                ". If you asked them to find a move: grade ONLY a move they clearly committed to "
                "(e.g. 'X is better'); if they did not commit to one, do NOT grade — ask which ONE "
                "they choose. NEVER pick the best candidate for them. If you did not ask a "
                "find-the-move question, ignore this note.]")

    _GIVE_UP_RE = re.compile(
        r"\b(?:i\s*(?:don'?t|do\s*not)\s*know|no\s+idea|i\s*give\s*up|idk|not\s+a\s+clue)\b",
        re.IGNORECASE)

    def _give_up_note(self, text: str) -> str:
        """'I don't know' had no legitimate path through the attempt counters, so
        the model was observed INVENTING a guess on the student's behalf to call
        the grading tools. Surrender is now a first-class input: the backend
        spends the attempt and tells the model the honest next step."""
        if not self._GIVE_UP_RE.search(text):
            return ""
        meta = self.state.lesson_metadata or {}
        if not self.state.review or meta.get("lesson_status") in ("lesson_complete", "awaiting_diagnosis"):
            return ""
        agenda = self.state.review.agenda
        idx = int(meta.get("last_agenda_index") or 0)
        if not 0 <= idx < len(agenda):
            return ""
        record = agenda[idx]
        if any(d.get("fen") == record.fen_before for d in self.state.diagnostics):
            return ""
        attempts = self.state.socratic_attempts
        for key in (record.fen_before, f"threat:{record.fen_before}"):
            attempts[key] = attempts.get(key, 0) + 1
        spent = max(attempts[record.fen_before], attempts[f"threat:{record.fen_before}"])
        if spent >= 2:
            return ("\n[Backend: the student GAVE UP on the current question; a surrendered attempt was "
                    f"recorded (now {spent}). Reveal is allowed: kindly reveal this moment's withheld "
                    "answer from the digest, walk through WHY it works step by step, then log the "
                    "diagnosis and continue — all in this turn. NEVER invent a guess on the student's "
                    "behalf to call the grading tools.]")
        return ("\n[Backend: the student GAVE UP on the current question; a surrendered attempt was "
                f"recorded ({spent} of 2 before reveal). Give the NEXT ladder hint only and ask them to "
                "try once more. NEVER invent a guess on the student's behalf to call the grading tools.]")

    def _pending_diagnosis_note(self) -> str:
        """Observed live (3x in one session): the model ends its turn ANNOUNCING
        it will log the diagnosis and move on, forcing the student to say 'ok'
        to advance. The lesson state machine knows a diagnosis is pending, so
        every student message carries the reminder until it's logged."""
        meta = self.state.lesson_metadata or {}
        if meta.get("lesson_status") != "awaiting_diagnosis" or not self.state.review:
            return ""
        agenda = self.state.review.agenda
        idx = int(meta.get("last_agenda_index") or 0)
        if not 0 <= idx < len(agenda):
            return ""
        record = agenda[idx]
        if any(d.get("fen") == record.fen_before for d in self.state.diagnostics):
            return ""
        return (f"\n[Backend: the diagnosis for the current moment ({record.move_label}) has NOT been "
                "logged yet. In THIS turn: finish the explanation if needed, call submit_diagnosis, and "
                "move straight on to the next moment (or the game synthesis if this was the last one). "
                "Do not end your turn by announcing future actions.]")

    _COMPLETE_ACK_RE = re.compile(
        r"^\s*(?:ok(?:ay)?|thanks?|thank you|great|perfect|nice|cool|done|that'?s all)\s*[.!]*\s*$",
        re.IGNORECASE)

    def _completion_ack_requests_close(self, text: str, plan: TeachingPlan) -> bool:
        if plan.intent == "close_with_takeaway":
            return False
        meta = self.state.lesson_metadata or {}
        if meta.get("lesson_status") != "lesson_complete":
            return False
        if not self.state.review:
            return False
        return bool(self._COMPLETE_ACK_RE.match(text or ""))

    def _dossier_write_allowed(self) -> bool:
        return bool((self.state.review_context or {}).get("dossier_consent", True))

    def _borrowed_game_note(self) -> str:
        context = self.state.review_context or {}
        if context.get("ownership") != "borrowed":
            return ""
        if context.get("dossier_consent"):
            return ("\n\n[Backend borrowed-game policy - P0-4c: this was not recognized as the student's "
                    "own game, but the student explicitly opted into dossier logging. Coach the chosen POV "
                    "only and log diagnoses normally.]")
        return ("\n\n[Backend borrowed-game policy - P0-4c: this was not recognized as the student's own "
                "game. Coach the chosen POV only. Dossier saving is OFF by default: submit_diagnosis may "
                "be used for session progression, but do NOT claim anything has been saved permanently to "
                "the student's personal dossier, and do not tell the student to /memorize this session.]")

    def _start_self_assessment(self, review) -> None:
        if not getattr(self.config, "review_self_assessment", True):
            return
        agenda = list(getattr(review, "agenda", []) or [])
        if not agenda:
            return
        self.state.teaching_ledger["self_assessment"] = {
            "status": "pending",
            "agenda": [
                {
                    "ply": r.ply,
                    "move_label": r.move_label,
                    "san": r.san,
                    "diagnosis": r.recommended_diagnosis,
                    "drop_pct": r.drop_pct,
                }
                for r in agenda
            ],
        }

    def _self_assessment_open_note(self, review) -> str:
        sa = self.state.teaching_ledger.get("self_assessment") or {}
        if sa.get("status") != "pending":
            return ""
        agenda = list(getattr(review, "agenda", []) or [])
        if not agenda:
            return ""
        labels = ", ".join(r.move_label for r in agenda[:5])
        return (
            "\n\n[Backend self-assessment opener - P0-8: before revealing the agenda, ask exactly one "
            "question: \"Before I show you my agenda, which move do you think was your biggest mistake, "
            "and why?\" Do not grade it yet and do not let the answer change the agenda. Verified agenda "
            f"labels for later deterministic comparison: {labels}.]")

    @staticmethod
    def _normalize_label_token(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").casefold())

    def _self_assessment_response_note(self, text: str, plan: TeachingPlan) -> str:
        sa = self.state.teaching_ledger.get("self_assessment") or {}
        if sa.get("status") != "pending":
            return ""
        if plan.intent in {"answer_directly", "explain_phase_summary"} and re.search(
                r"\b(?:just show me|skip|show me the agenda|don'?t quiz me|do not quiz me)\b",
                text or "", re.I):
            sa["status"] = "skipped"
            sa["result"] = "student_opted_out"
            return ("\n[Backend self-assessment - P0-8: the student opted out of the opener. "
                    "Proceed with the verified agenda normally; do not scold and do not ask the "
                    "self-assessment question again.]")

        agenda = sa.get("agenda") or []
        normalized_text = self._normalize_label_token(text)
        matched = None
        for item in agenda:
            labels = {
                self._normalize_label_token(item.get("move_label")),
                self._normalize_label_token(item.get("san")),
            }
            if any(label and label in normalized_text for label in labels):
                matched = item
                break

        if matched is None:
            result = "no_move_named"
            detail = "No agenda move label or SAN was clearly named."
        else:
            first_ply = agenda[0]["ply"] if agenda else matched["ply"]
            delta = abs(int(matched["ply"]) - int(first_ply))
            if delta == 0:
                result = "exact_top_agenda_match"
                detail = f"The student named the top agenda moment: {matched['move_label']}."
            elif delta <= 2:
                result = "near_miss"
                detail = (f"The student named {matched['move_label']}, close to the top agenda moment "
                          f"{agenda[0]['move_label']}.")
            else:
                result = "agenda_match_not_top"
                detail = (f"The student named a real agenda moment ({matched['move_label']}), but the "
                          f"top backend agenda moment is {agenda[0]['move_label']}.")
        sa["status"] = "graded"
        sa["result"] = result
        sa["student_text"] = (text or "")[:240]
        if matched:
            sa["matched_ply"] = matched["ply"]
        return (
            "\n[Backend self-assessment result - P0-8: "
            f"{result}. {detail} Acknowledge briefly, then reveal the verified agenda and proceed "
            "normally. The student's guess must not change agenda order.]")

    def _redundancy_note(self, plan: TeachingPlan) -> str:
        ledger = self.state.teaching_ledger or {}
        moments = list(ledger.get("explained_moments") or [])
        concepts = list(ledger.get("explained_concepts") or [])
        if not moments and not concepts:
            return ""
        if plan.intent in {"ask_socratic_question", "give_hint", "generate_exercise"}:
            return ""
        recent_moments = "; ".join(
            f"{m.get('move_label')} ({m.get('diagnosis') or m.get('lesson_objective') or 'lesson'})"
            for m in moments[-4:]
        )
        recent_concepts = ", ".join(c.get("concept") for c in concepts[-6:] if c.get("concept"))
        parts = []
        if recent_moments:
            parts.append(f"already-explained moments: {recent_moments}")
        if recent_concepts:
            parts.append(f"already-explained concepts: {recent_concepts}")
        return ("\n[Backend redundancy ledger - P0-8: " + "; ".join(parts)
                + ". Do not re-quiz or re-teach these from scratch unless the student explicitly asks "
                  "for a recap; if relevant, connect to the broader arc instead.]")

    def _verified_takeaway(self) -> dict:
        review = self.state.review
        if not review:
            return {}
        themes = list(getattr(review, "themes", None) or [])
        confidence_rank = {"high": 0, "medium": 1, "low": 2}
        themes.sort(key=lambda t: (
            confidence_rank.get(getattr(t, "confidence", "low"), 2),
            -len(getattr(t, "plies", []) or []),
            (getattr(t, "plies", [999]) or [999])[0],
        ))
        if themes:
            theme = themes[0]
            label = getattr(theme, "label", "theme")
            summary = getattr(theme, "safe_summary", "") or label
            text = f"Session takeaway: {summary}"
            return {
                "game_id": self.state.current_game_id,
                "theme_id": getattr(theme, "id", None),
                "text": text,
                "source": "review_theme.safe_summary",
            }
        agenda = list(getattr(review, "agenda", []) or [])
        if agenda:
            record = agenda[0]
            label = record.recommended_diagnosis or record.lesson_objective or record.quality
            text = f"Session takeaway: the priority lesson was {label.replace('_', ' ')} at {record.move_label}."
            return {
                "game_id": self.state.current_game_id,
                "theme_id": None,
                "text": text,
                "source": "agenda_record",
            }
        return {}

    def _takeaway_note(self) -> str:
        takeaway = self._verified_takeaway()
        if not takeaway:
            return ("\n[Backend session close: no verified review theme is available. Close briefly; "
                    "do not invent a theme and do not start a new quiz.]")
        self.state.teaching_ledger["last_takeaway"] = dict(takeaway)
        if self._dossier_write_allowed():
            record_session_takeaway(getattr(self.config, "dossier_file", None), takeaway)
        return (
            "\n[Backend session close: use this verified takeaway exactly or as a very close paraphrase: "
            f"{takeaway['text']} Source: {takeaway['source']}. Close the loop in one sentence, optionally "
            "add one short next-step sentence, and do NOT ask a new Socratic question.]")

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

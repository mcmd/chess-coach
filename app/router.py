"""Opt-in message router: a fast model classifies the student's message, the backend
checks the classification and does the grading, and the coach model only writes the
reply. The coach model therefore never chooses what text gets graded.

Any router failure (no model configured, API error, timeout, invalid output) returns
None and the turn falls back to the normal flow, where the coach model decides.
"""
import json
import time
from dataclasses import dataclass

import chess

from app.notation import legal_san_mentions
from app.tools import CoachTools
from app.trace import trace

LABELS = ("move", "threat", "explanation", "candidates", "self_assessment", "hint", "give_up",
          "move_on", "question", "navigate", "other")
GRADING_LABELS = ("move", "threat", "explanation")

PROMPT = """You route one chess student's message inside a coaching session. Classify ONLY; do not answer.

Labels:
- move: the student commits to ONE move of their own as their answer (may also give a reason). A student who
  says "I would play X" or "I play X" is proposing their OWN move, even if the coach asked about the opponent.
- threat: the student names what the OPPONENT is threatening.
- explanation: the student explains WHY a move works or fails (a reason, not a new move).
- candidates: the student lists several possible moves without choosing one.
- self_assessment: the student answers which of their own moves was their biggest mistake.
- hint: the student asks for a hint.
- give_up: the student says they don't know, or asks to be shown/told the answer.
- move_on: the student is ready to continue, return to the lesson, go to the next moment, finish, or accepts the coach's proposal.
- question: the student asks a question or makes a request (explain, translate, summarize, annotate...) without answering the coach's exercise.
- navigate: the student asks to look at a different position or move of the game.
- other: anything else (e.g. loading a game, chit-chat).

If the message both asks a question and answers, choose the answer label only when the answer is explicit.
For move/threat, copy the move EXACTLY as the student wrote it (e.g. "Ne4", "queen to g6", "qe4").
For explanation (or a reason given with a move), copy the student's reason text exactly. Use null when absent.
about_current_moment: true only if the student is answering the coach's exercise about the CURRENT LESSON MOMENT
below (not an earlier position, not a different game).

CURRENT LESSON MOMENT: {moment}

COACH'S PREVIOUS MESSAGE (may be truncated):
<<<{coach}>>>

STUDENT MESSAGE:
<<<{student}>>>"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING", "enum": list(LABELS)},
        "move": {"type": "STRING", "nullable": True},
        "threat": {"type": "STRING", "nullable": True},
        "explanation": {"type": "STRING", "nullable": True},
        "about_current_moment": {"type": "BOOLEAN"},
    },
    "required": ["label", "about_current_moment"],
}


@dataclass
class Route:
    label: str
    move: str | None = None
    threat: str | None = None
    explanation: str | None = None
    about_current_moment: bool = False
    model_version: str | None = None
    seconds: float = 0.0


class Router:
    def __init__(self, client, model: str, thinking_level: str = "low", timeout_s: float = 10.0):
        # 10 s is the API's minimum request deadline; a slower reply falls back to the normal flow.
        self.client = client
        self.model = model
        self.thinking_level = thinking_level
        self.timeout_ms = int(timeout_s * 1000)

    def classify(self, student_text: str, coach_text: str, moment: str | None) -> Route | None:
        from google.genai import types
        coach = " ".join((coach_text or "").split())[-1200:] or "(none: start of session)"
        started = time.monotonic()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=PROMPT.format(moment=moment or "none (no review moment is active)",
                                       coach=coach, student=student_text),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
                    http_options=types.HttpOptions(timeout=self.timeout_ms)),
            )
            data = json.loads(response.text)
        except Exception as exc:
            trace(f"router unavailable ({type(exc).__name__}: {str(exc)[:160]}); using the normal flow")
            return None
        if data.get("label") not in LABELS:
            return None
        return Route(label=data["label"], move=data.get("move"), threat=data.get("threat"),
                     explanation=data.get("explanation"),
                     about_current_moment=bool(data.get("about_current_moment")),
                     model_version=getattr(response, "model_version", None),
                     seconds=round(time.monotonic() - started, 2))


def _quoted(span: str | None, text: str) -> bool:
    compact = CoachTools._compact_student_text
    return bool(span) and bool(compact(span)) and compact(span) in compact(text)


def resolve_move(board: chess.Board, span: str | None, text: str) -> str | None:
    """The one legal move the quoted span denotes on `board`, or None."""
    if not _quoted(span, text):
        return None
    mentions = legal_san_mentions(board, span)
    return mentions[0] if len(mentions) == 1 else None


def describe_moment(fen_before: str, move_label: str | None) -> str:
    """How the router prompt names the current agenda moment."""
    board = chess.Board(fen_before)
    side = "White" if board.turn == chess.WHITE else "Black"
    return f"move {board.fullmove_number}, {side} to move (in the game: {move_label})"


def resolve_threat(board: chess.Board, span: str | None, text: str) -> str | None:
    """The one legal opponent move (after a null move) the quoted span denotes, or None."""
    if board.is_check():
        return None
    probe = board.copy(stack=False)
    probe.push(chess.Move.null())
    return resolve_move(probe, span, text)


def resolve_explanation(span: str | None, text: str) -> str | None:
    return span.strip() if _quoted(span, text) else None

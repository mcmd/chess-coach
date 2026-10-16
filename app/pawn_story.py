"""Pawn-structure evolution storytelling.

This module turns per-position structural facts into a game-level narrative
packet. The model may phrase the lesson, but labels like "open file",
"isolated passer", and "king shield" come from this backend layer.
"""
from dataclasses import asdict, dataclass

import chess

from app.concepts import _get_passed_pawns, _get_pawn_data

FILES = "abcdefgh"


@dataclass
class PawnStructureEvent:
    ply: int
    move: str
    event: str
    facts: dict
    safe_summary: str

    def to_dict(self):
        return asdict(self)


def file_status(board: chess.Board, file_name: str) -> str:
    file_idx = FILES.index(file_name)
    has_white = any(chess.square_file(sq) == file_idx for sq in board.pieces(chess.PAWN, chess.WHITE))
    has_black = any(chess.square_file(sq) == file_idx for sq in board.pieces(chess.PAWN, chess.BLACK))
    if not has_white and not has_black:
        return "open"
    if not has_white and has_black:
        return "half_open_for_white"
    if has_white and not has_black:
        return "half_open_for_black"
    return "blocked_by_pawns"


def file_statuses(board: chess.Board) -> dict:
    return {f: file_status(board, f) for f in FILES}


def _adjacent_files(file_idx: int) -> list:
    return [f for f in (file_idx - 1, file_idx + 1) if 0 <= f <= 7]


def pawn_snapshot(board: chess.Board, color: chess.Color) -> dict:
    label = "white" if color == chess.WHITE else "black"
    data = _get_pawn_data(board, color)
    pawns = sorted(chess.square_name(sq) for sq in board.pieces(chess.PAWN, color))
    return {
        "color": label,
        "pawns": pawns,
        "passed": sorted(_get_passed_pawns(board, color)),
        "isolated": sorted(data["isolated"]),
        "doubled_files": sorted(data["doubled_files"]),
        "file_statuses": file_statuses(board),
    }


def passed_pawn_facts(board: chess.Board, color: chess.Color, pawn_name: str) -> dict:
    sq = chess.parse_square(pawn_name)
    file_idx = chess.square_file(sq)
    adjacent = _adjacent_files(file_idx)
    friendly_adjacent = [
        chess.square_name(p)
        for p in board.pieces(chess.PAWN, color)
        if chess.square_file(p) in adjacent
    ]
    supporting = []
    for atk in board.attackers(color, sq):
        piece = board.piece_at(atk)
        if piece and piece.piece_type != chess.KING:
            supporting.append(f"{chess.piece_name(piece.piece_type)} {chess.square_name(atk)}")
    return {
        "pawn": pawn_name,
        "is_isolated": pawn_name in _get_pawn_data(board, color)["isolated"],
        "adjacent_friendly_pawns": sorted(friendly_adjacent),
        "supporting_pieces": sorted(supporting),
    }


def king_shield_facts(board: chess.Board, color: chess.Color) -> dict:
    king_sq = board.king(color)
    if king_sq is None:
        return {"king": None}
    kf = chess.square_file(king_sq)
    home_rank = 1 if color == chess.WHITE else 6
    zone_files = [f for f in (kf - 1, kf, kf + 1) if 0 <= f <= 7]
    friendly = []
    advanced = []
    pinned = []
    missing_home = []
    for f in zone_files:
        home_sq = chess.square(f, home_rank)
        home_piece = board.piece_at(home_sq)
        if not (home_piece and home_piece.color == color and home_piece.piece_type == chess.PAWN):
            missing_home.append(chess.square_name(home_sq))
        for sq in board.pieces(chess.PAWN, color):
            if chess.square_file(sq) != f:
                continue
            name = chess.square_name(sq)
            if name not in friendly:
                friendly.append(name)
            if chess.square_rank(sq) != home_rank:
                advanced.append(name)
            if board.is_pinned(color, sq):
                pinned.append(name)
    return {
        "king": chess.square_name(king_sq),
        "friendly_pawns_near_king": sorted(friendly),
        "missing_home_shield_pawns": sorted(missing_home),
        "advanced_shield_pawns": sorted(advanced),
        "pinned_shield_pawns": sorted(pinned),
        "forcing_threats": _forcing_threats_against_shield(board, color, pinned),
    }


def _forcing_threats_against_shield(board: chess.Board, color: chess.Color, pinned_pawns: list) -> list:
    if not pinned_pawns:
        return []
    probe = board.copy(stack=False)
    probe.turn = not color
    probe.ep_square = None
    if not probe.is_valid():
        return []
    pinned_squares = {chess.parse_square(name) for name in pinned_pawns}
    threats = []
    for move in probe.legal_moves:
        if move.to_square in pinned_squares and probe.is_capture(move) and probe.gives_check(move):
            threats.append(probe.san(move))
    return sorted(threats)


def analyze_pawn_structure_evolution(review, color: str = None) -> dict:
    side = color or review.student_color or "white"
    chess_color = chess.WHITE if side == "white" else chess.BLACK
    events = []
    previous_shield = None
    previous_snapshot = None
    last_file_change = {}   # file -> (move label, status before that change)
    reversal_note = {}      # id(event) -> "the move whose change this undoes"

    for record in review.records:
        before = chess.Board(record.fen_before)
        after = chess.Board(record.fen_after)
        before_snap = pawn_snapshot(before, chess_color)
        after_snap = pawn_snapshot(after, chess_color)

        for pawn in sorted(set(after_snap["passed"]) - set(before_snap["passed"])):
            facts = passed_pawn_facts(after, chess_color, pawn)
            kind = "isolated passed pawn" if facts["is_isolated"] else "passed pawn"
            events.append(PawnStructureEvent(
                ply=record.ply,
                move=record.move_label,
                event="passed_pawn_created",
                facts=facts,
                safe_summary=(f"{side.title()} created an {kind} on {pawn}. It is an asset, but it needs "
                              "piece support because no friendly pawn on an adjacent file can defend it."
                              if facts["is_isolated"] else
                              f"{side.title()} created a passed pawn on {pawn}."),
            ))

        changed_files = {
            f: after_snap["file_statuses"][f]
            for f in FILES
            if previous_snapshot and previous_snapshot["file_statuses"][f] != after_snap["file_statuses"][f]
        }
        for f, status in changed_files.items():
            event = PawnStructureEvent(
                ply=record.ply,
                move=record.move_label,
                event="file_status_changed",
                facts={"file": f, "status": status},
                safe_summary=f"The {f}-file is now {status.replace('_', ' ')}.",
            )
            earlier = last_file_change.get(f)
            if earlier and earlier[1] == status:
                reversal_note[id(event)] = earlier[0]
            last_file_change[f] = (record.move_label, previous_snapshot["file_statuses"][f])
            events.append(event)

        shield = king_shield_facts(after, chess_color)
        if previous_shield and shield != previous_shield and (
                shield.get("advanced_shield_pawns") or shield.get("pinned_shield_pawns")):
            pinned = shield.get("pinned_shield_pawns") or []
            threat = ", ".join(shield.get("forcing_threats") or [])
            if pinned:
                summary = (f"{side.title()}'s king shield changed: pawn(s) {', '.join(pinned)} are pinned. "
                           + (f"The concrete issue is {threat}, not an automatically open file."
                              if threat else "Do not call this an open file unless file_status says open."))
            else:
                summary = (f"{side.title()}'s king shield changed: advanced pawn(s) "
                           f"{', '.join(shield.get('advanced_shield_pawns', []))}.")
            events.append(PawnStructureEvent(
                ply=record.ply,
                move=record.move_label,
                event="king_shield_changed",
                facts=shield,
                safe_summary=summary,
            ))
        previous_shield = shield
        previous_snapshot = after_snap

    final_board = chess.Board(review.records[-1].fen_after)
    final_snapshot = pawn_snapshot(final_board, chess_color)
    final_shield = king_shield_facts(final_board, chess_color)
    return {
        "color": side,
        "events": [e.to_dict() for e in events],
        # Router run 5kf7rqwj S23: the coach compressed the events and skipped 16. dxe4, which
        # blocked the e-file that 15...dxe4 had opened. The story is told from this list.
        "chronology": [f"{e.move}: {e.safe_summary}"
                       + (f" (this undoes the change at {reversal_note[id(e)]})" if id(e) in reversal_note else "")
                       for e in events],
        "narration_rule": ("Tell the story in this order, using the chronology. Do not skip a step that "
                           "undoes an earlier one (a file that opens and then closes again). Quote each "
                           "move label exactly as written: '15... dxe4' is Black's move, never 'move 15 (dxe4)' "
                           "in a story about White (run pr40jw2f)."),
        "final_pawn_snapshot": final_snapshot,
        "final_king_shield": final_shield,
        "safe_language": safe_language(events, final_snapshot, final_shield),
        "avoid_language": avoid_language(final_snapshot, final_shield),
        "language_guardrails": [
            "Use 'open file' only when file_status is exactly open.",
            "Do not say a pawn is gone unless it is absent from the pawn list.",
            "Avoid 'collapse' unless the concrete missing, advanced, or pinned shield pawns are named.",
            "Do not call centralization a blockade unless the backend event explicitly says blockade.",
        ],
    }


def safe_language(events: list, final_snapshot: dict, final_shield: dict) -> list:
    phrases = []
    for event in events:
        if event.event == "passed_pawn_created":
            pawn = event.facts["pawn"]
            if event.facts.get("is_isolated"):
                phrases.append({
                    "vivid": f"the {pawn} pawn became the spearhead of your position",
                    "concrete": f"{pawn} is an isolated passed pawn, so it can advance without enemy pawn blockers but needs piece support",
                })
            else:
                phrases.append({
                    "vivid": f"the {pawn} pawn became a runner",
                    "concrete": f"{pawn} is passed: no enemy pawn on its file or adjacent files can stop it directly",
                })
    pinned = final_shield.get("pinned_shield_pawns") or []
    threats = final_shield.get("forcing_threats") or []
    if pinned:
        phrases.append({
            "vivid": f"the {', '.join(pinned)} pawn became the hinge of the king shelter",
            "concrete": f"{', '.join(pinned)} is pinned to the king, so it cannot behave like a normal shield pawn",
        })
    if threats:
        phrases.append({
            "vivid": "the king shelter started to fray",
            "concrete": f"the concrete issue was {', '.join(threats)}, not a vague open-file problem",
        })
    if final_shield.get("friendly_pawns_near_king"):
        phrases.append({
            "vivid": "your king still had shelter, but it was specific and fragile",
            "concrete": "the remaining nearby pawns were: " + ", ".join(final_shield["friendly_pawns_near_king"]),
        })
    return phrases


def avoid_language(final_snapshot: dict, final_shield: dict) -> list:
    """False claims a model might be tempted to make, derived from the ACTUAL
    final position — never hardcoded to any particular game."""
    avoid = []
    statuses = final_snapshot["file_statuses"]

    # Claims that shield pawns are gone, when they are demonstrably present.
    for pawn in final_shield.get("friendly_pawns_near_king") or []:
        avoid.append(f"the {pawn[0]}-pawn was gone (it is still on {pawn})")

    # Claims that files near the king are open, when they are not.
    king = final_shield.get("king")
    if king:
        king_file = FILES.index(king[0])
        for f_idx in _adjacent_files(king_file) + [king_file]:
            f = FILES[f_idx]
            if statuses[f] != "open":
                avoid.append(f"the {f}-file was open (it is {statuses[f].replace('_', ' ')})")

    if final_shield.get("pinned_shield_pawns"):
        avoid.append("total kingside collapse without naming the pinned pawn")
    avoid.append("any blockade/outpost/battery label the backend events do not state")
    return avoid

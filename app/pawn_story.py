"""Pawn-structure evolution storytelling.

This module turns per-position structural facts into a game-level narrative
packet. The model may phrase the lesson, but labels like "open file",
"isolated passer", and "king shield" come from this backend layer.
"""
from dataclasses import asdict, dataclass

import chess

from app.concepts import _get_passed_pawns, _get_pawn_data

FILES = "abcdefgh"

# Live runs jwuw0nd6/bgvvfacf: "half open for white" after 8. cxd3 was read as "Black's c-pawn was
# exchanged". Each status says which side's pawn is missing and which remains.
_FILE_STATUS_WORDS = {
    "open": "open (no pawn of either side on it)",
    "half_open_for_white": "half open for White (no White pawn on it; a Black pawn is still on it)",
    "half_open_for_black": "half open for Black (no Black pawn on it; a White pawn is still on it)",
    "blocked_by_pawns": "blocked by pawns (both sides have a pawn on it)",
}


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
    # §10.2: passed is not the same as free to run. Run n6bmuxpr: after 18. exd5 the d5 passer
    # had Black's queen on d6 directly in front of it.
    stop = sq + (8 if color == chess.WHITE else -8)
    blocker = board.piece_at(stop) if 0 <= stop < 64 else None
    return {
        "pawn": pawn_name,
        "is_isolated": pawn_name in _get_pawn_data(board, color)["isolated"],
        "adjacent_friendly_pawns": sorted(friendly_adjacent),
        "supporting_pieces": sorted(supporting),
        "blocked_by": (f"{'white' if blocker.color else 'black'} {chess.piece_name(blocker.piece_type)} on "
                       f"{chess.square_name(stop)}" if blocker else None),
    }


def king_shield_facts(board: chess.Board, color: chess.Color) -> dict:
    king_sq = board.king(color)
    if king_sq is None:
        return {"king": None}
    kf = chess.square_file(king_sq)
    home_rank = 1 if color == chess.WHITE else 6
    # Only pawns on the king's own half shelter it. Run bgvvfacf: Black's pawn on g2 was reported
    # as an "advanced shield pawn" of the king on g8 and narrated as "advancing their king shield to g2".
    own_half = range(0, 4) if color == chess.WHITE else range(4, 8)
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
            if chess.square_file(sq) != f or chess.square_rank(sq) not in own_half:
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


def _played_move(before: chess.Board, after: chess.Board, uci: str):
    """The legal move that turns `before` into `after`, or None when the record does not hold
    exactly one move (a hand-built record spanning several moves)."""
    target = after.board_fen()
    moves = list(before.legal_moves)
    try:
        hinted = chess.Move.from_uci(uci or "")
    except ValueError:
        hinted = None
    if hinted in moves:
        moves.insert(0, hinted)
    for move in moves:
        before.push(move)
        reached = before.board_fen() == target
        before.pop()
        if reached:
            return move
    return None


def _captured_square(board: chess.Board, move: chess.Move):
    if board.is_en_passant(move):
        return move.to_square + (-8 if board.turn == chess.WHITE else 8)
    return move.to_square if board.is_capture(move) else None


def _origin_note(identity: dict, square: chess.Square) -> str:
    """'(the pawn that started on e2)' when the pawn changed file since the game's start board.
    Runs bgvvfacf/n6bmuxpr narrated the e-pawn on f3 as "their f-pawn"."""
    origin, from_start = identity.get(square, (None, False))
    if not from_start or origin is None or origin[0] == chess.square_name(square)[0]:
        return ""
    return f" (the pawn that started on {origin})"


def _passer_events(record, side: str, color: chess.Color, before: chess.Board, after: chess.Board,
                   move: chess.Move, identity_before: dict, identity_after: dict) -> list:
    """Passed-pawn transitions caused by this one move, judged by which pawn moved, was captured, or
    changed status in place. Comparing square names alone reported 'd5 -> d6' as a new passer (§10.2)."""
    Side = side.title()
    passed_before = set(_get_passed_pawns(before, color))
    passed_after = set(_get_passed_pawns(after, color))
    handled_before, handled_after = set(), set()
    events = []

    def add(kind, facts, summary):
        events.append(PawnStructureEvent(ply=record.ply, move=record.move_label, event=kind,
                                         facts=facts, safe_summary=summary))

    def identity_facts(identity, square):
        origin, from_start = identity.get(square, (None, False))
        return {"origin_square": origin, "origin_is_game_start": from_start}

    def created(square_name, cause):
        sq = chess.parse_square(square_name)
        facts = {**passed_pawn_facts(after, color, square_name), **identity_facts(identity_after, sq),
                 "cause": cause}
        note = _origin_note(identity_after, sq)
        if facts["is_isolated"]:
            summary = (f"{Side} created an isolated passed pawn on {square_name}{note}. It is an asset, but it "
                       "needs piece support because no friendly pawn on an adjacent file can defend it.")
        elif cause == "pawn_move":
            summary = f"{Side} created a passed pawn on {square_name}{note}."
        else:
            summary = (f"{Side}'s pawn on {square_name}{note} became passed: no enemy pawn is left ahead of it "
                       "on its own or an adjacent file.")
        if facts["blocked_by"]:
            summary += (f" The {facts['blocked_by']} stands directly in front of it, so it cannot step "
                        "forward while that piece stays there.")
        add("passed_pawn_created", facts, summary)

    if before.turn == color and before.piece_type_at(move.from_square) == chess.PAWN:
        frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
        handled_before.add(frm)
        handled_after.add(to)
        base = {"from": frm, "to": to, **identity_facts(identity_before, move.from_square)}
        if frm in passed_before and move.promotion:
            promoted = chess.piece_name(move.promotion)
            add("passed_pawn_promoted", {**base, "promoted_to": promoted},
                f"{Side}'s passed pawn on {frm} promoted to a {promoted} on {to}.")
        elif frm in passed_before and to in passed_after:
            facts = {**base, **passed_pawn_facts(after, color, to)}
            add("passed_pawn_advanced", facts,
                f"{Side}'s passed pawn moved from {frm} to {to}. It was already passed; this is not a new passer."
                + (f" The {facts['blocked_by']} stands directly in front of it." if facts["blocked_by"] else ""))
        elif frm in passed_before:
            add("passed_pawn_no_longer_passed", base,
                f"{Side}'s passed pawn left {frm} for {to}, where it is no longer passed.")
        elif move.promotion is None and to in passed_after:
            created(to, "pawn_move")
    elif before.turn != color:
        captured = _captured_square(before, move)
        if captured is not None and before.piece_type_at(captured) == chess.PAWN:
            name = chess.square_name(captured)
            handled_before.add(name)
            if name in passed_before:
                add("passed_pawn_captured", {"pawn": name, **identity_facts(identity_before, captured)},
                    f"{Side}'s passed pawn on {name}{_origin_note(identity_before, captured)} was captured.")

    for name in sorted(passed_after - passed_before - handled_after):
        created(name, "position_change")
    for name in sorted(passed_before - passed_after - handled_before):
        add("passed_pawn_no_longer_passed", {"pawn": name, **identity_facts(identity_after, chess.parse_square(name))},
            f"{Side}'s pawn on {name} is no longer passed: an enemy pawn now stands ahead of it on its own "
            "or an adjacent file.")
    return events


def _identity_after(identity: dict, before: chess.Board, move: chess.Move, color: chess.Color) -> dict:
    """Each pawn's identity after one legal move: it follows its own moves and captures (including
    en passant), and disappears when captured or promoted."""
    identity = dict(identity)
    if before.turn == color:
        if before.piece_type_at(move.from_square) == chess.PAWN:
            origin = identity.pop(move.from_square, (chess.square_name(move.from_square), False))
            if move.promotion is None:
                identity[move.to_square] = origin
    else:
        captured = _captured_square(before, move)
        if captured is not None:
            identity.pop(captured, None)
    return identity


def _shield_event(record, side: str, color: chess.Color, before: chess.Board,
                  before_shield: dict, after_shield: dict, move):
    """What this record's own move changed in the king's pawn cover, or None. A king walking past
    unchanged pawns is no shield event (runs n6bmuxpr/bgvvfacf: 23... fxg2 and 24. Kxg2 were both
    reported as 'advanced pawn(s) g4', which had not changed)."""
    if before_shield.get("king") is None or after_shield.get("king") is None:
        return None
    near_before = set(before_shield["friendly_pawns_near_king"])
    near_after = set(after_shield["friendly_pawns_near_king"])
    pinned_before = set(before_shield["pinned_shield_pawns"])
    pinned_after = set(after_shield["pinned_shield_pawns"])
    pawns_changed = before_shield["king"] == after_shield["king"] and near_before != near_after
    if not pawns_changed and pinned_before == pinned_after:
        return None

    changes = []
    if pawns_changed:
        gone, new = near_before - near_after, near_after - near_before
        if move is not None:
            frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
            if (before.turn == color and before.piece_type_at(move.from_square) == chess.PAWN
                    and (frm in gone or to in new)):
                verb = "captured on" if before.is_capture(move) else "advanced to"
                changes.append(f"the {frm} pawn {verb} {to}" if to in new
                               else f"the {frm} pawn left the king's side ({frm}-{to})")
                gone.discard(frm)
                new.discard(to)
            captured = _captured_square(before, move) if before.turn != color else None
            if captured is not None and chess.square_name(captured) in gone:
                changes.append(f"the {chess.square_name(captured)} pawn was captured")
                gone.discard(chess.square_name(captured))
        if gone:
            changes.append(f"no longer near the king: {', '.join(sorted(gone))}")
        if new:
            changes.append(f"now near the king: {', '.join(sorted(new))}")
    if pinned_after - pinned_before:
        changes.append(f"now pinned: {', '.join(sorted(pinned_after - pinned_before))}")
    if pinned_before - pinned_after:
        changes.append(f"no longer pinned: {', '.join(sorted(pinned_before - pinned_after))}")

    Side = side.title()
    summary = f"{Side}'s king shield (king on {after_shield['king']}) changed: " + "; ".join(changes) + "."
    pinned = after_shield["pinned_shield_pawns"]
    if pinned:
        threat = ", ".join(after_shield.get("forcing_threats") or [])
        summary += (f" Pawn(s) {', '.join(pinned)} are pinned to the king. "
                    + (f"The concrete issue is {threat}, not an automatically open file."
                       if threat else "Do not call this an open file unless file_status says open."))
    near = after_shield["friendly_pawns_near_king"]
    summary += f" Pawns near the king now: {', '.join(near) if near else 'none'}."
    return PawnStructureEvent(ply=record.ply, move=record.move_label, event="king_shield_changed",
                              facts={**after_shield, "changes": changes}, safe_summary=summary)


def analyze_pawn_structure_evolution(review, color: str = None) -> dict:
    side = color or review.student_color or "white"
    chess_color = chess.WHITE if side == "white" else chess.BLACK
    if not review.records:
        return {
            "color": side,
            "status": "no_moves",
            "events": [],
            "chronology": [],
            "narration_rule": "The loaded game has no moves, so there is no pawn-structure story to tell.",
        }
    events = []
    last_file_change = {}   # file -> (move label, status before that change, status after it)
    reversal_note = {}      # id(event) -> "the move whose change this undoes"
    identity = None         # square -> (square when tracking began, began at the game's start board)
    tracked_board = None    # board_fen the identity map describes

    for index, record in enumerate(review.records):
        before = chess.Board(record.fen_before)
        after = chess.Board(record.fen_after)
        move = _played_move(before, after, getattr(record, "uci", None))
        # §10.2: identity starts from the loaded game's start board. A gap in the records (or a record
        # that is not one move) restarts it from where the pawns stand; no earlier history is invented.
        if identity is None or tracked_board != before.board_fen():
            from_start = index == 0 and record.ply == 0
            identity = {sq: (chess.square_name(sq), from_start)
                        for sq in before.pieces(chess.PAWN, chess_color)}

        if move is not None:
            moved = _identity_after(identity, before, move, chess_color)
            events.extend(_passer_events(record, side, chess_color, before, after, move, identity, moved))
            identity = moved
            tracked_board = after.board_fen()
        else:
            identity = None

        # File and shield transitions compare this record's own before/after, so the first record
        # counts and a sparse list cannot attribute a skipped move's change to the next record.
        before_files = file_statuses(before)
        after_files = file_statuses(after)
        for f in FILES:
            if before_files[f] == after_files[f]:
                continue
            status = after_files[f]
            event = PawnStructureEvent(
                ply=record.ply,
                move=record.move_label,
                event="file_status_changed",
                facts={"file": f, "status": status, "previous_status": before_files[f]},
                safe_summary=f"The {f}-file is now {_FILE_STATUS_WORDS[status]}.",
            )
            earlier = last_file_change.get(f)
            if earlier and earlier[1] == status and earlier[2] == before_files[f]:
                reversal_note[id(event)] = earlier[0]
            last_file_change[f] = (record.move_label, before_files[f], status)
            events.append(event)

        shield_event = _shield_event(record, side, chess_color, before,
                                     king_shield_facts(before, chess_color),
                                     king_shield_facts(after, chess_color), move)
        if shield_event:
            events.append(shield_event)

    final_board = chess.Board(review.records[-1].fen_after)
    final_snapshot = pawn_snapshot(final_board, chess_color)
    if identity is not None:
        final_snapshot["origins"] = {chess.square_name(sq): origin for sq, (origin, from_start)
                                     in sorted(identity.items()) if from_start}
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
            "Passed or isolated does not mean free to run: name the blocker when the event gives one.",
        ],
    }


def safe_language(events: list, final_snapshot: dict, final_shield: dict) -> list:
    phrases = []
    for event in events:
        if event.event == "passed_pawn_created":
            pawn = event.facts["pawn"]
            # §10.2: isolated/passed says nothing about running safely; a piece may block it.
            blocked = (f"; the {event.facts['blocked_by']} stood directly in front of it"
                       if event.facts.get("blocked_by") else "")
            if event.facts.get("is_isolated"):
                phrases.append({
                    "vivid": f"the {pawn} pawn became the spearhead of your position",
                    "concrete": (f"after {event.move}, {pawn} was an isolated passed pawn: no enemy pawn could stop "
                                 f"it directly, but no friendly pawn could defend it, so it needed piece support"
                                 f"{blocked}"),
                })
            else:
                phrases.append({
                    "vivid": f"no enemy pawn was left in the {pawn} pawn's way",
                    "concrete": (f"after {event.move}, {pawn} was passed: no enemy pawn on its file or adjacent "
                                 f"files could stop it directly{blocked}"),
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

import copy
import json
import os
import re
import shutil
from datetime import date

from google.genai import types

# Conversation history is replayed into the model's context on resume —
# keep it bounded so old sessions don't drown new ones.
MAX_HISTORY_MESSAGES = 80

# Hysteresis for the persisted slide_trend "above_baseline" flag. The per-batch
# detector raises it at >= 1.5x the corpus base rate (app.batch.SLIDE_ABOVE_BASELINE);
# persisting that raw bool would let a player hovering near the line flip the
# user-facing "ABOVE typical" message on and off between batches on eval noise
# alone. Turn the flag ON at the detector's upper threshold, OFF only once the rate
# falls below this lower multiple of baseline, and hold it in the dead-band. (issue #4)
SLIDE_ABOVE_BASELINE_OFF = 1.2
DRILL_RECENT_WINDOW = 10
DRILL_SET_POSITION_LIMIT = 12
PATTERN_RECENT_BATCHES = 3
PATTERN_IMPROVING_FACTOR = 0.6
PATTERN_SLIPPING_FACTOR = 1.4


def _atomic_write_json(path: str, data) -> None:
    """Write-then-rename so a crash mid-write can't corrupt the file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_history(history_file):
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                contents = []
                for item in data:
                    parts = [types.Part.from_text(text=p["text"]) for p in item["parts"]]
                    contents.append(types.Content(role=item["role"], parts=parts))
                return contents
            except (json.JSONDecodeError, KeyError):
                return []
    return []


def save_history(chat_session, history_file):
    serializable = []
    for message in chat_session.get_history():
        parts = []
        for p in message.parts or []:
            text = getattr(p, 'text', None)
            if not text:
                continue
            # Internal scratchpads are bookkeeping, not conversation — drop them.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            if text:
                parts.append({"text": text})
        if parts:
            serializable.append({"role": message.role, "parts": parts})
    _atomic_write_json(history_file, serializable[-MAX_HISTORY_MESSAGES:])


# Quality-SOL §7.3: files without `schema_version` are version 1 (counters without evidence).
# Version 2 keeps every diagnosis as an evidence-qualified event (§7.1) and holds the old totals
# apart as legacy, unverified observations.
DOSSIER_SCHEMA_VERSION = 2
DIAGNOSIS_EVIDENCE_VERSION = 1
DIAGNOSIS_EVENT_LIMIT = 1000
LEGACY_UNVERIFIED = "legacy_unverified"

_DOSSIER_DEFAULT = {
    "schema_version": DOSSIER_SCHEMA_VERSION,
    # Trusted aggregates, regenerated from evidence-qualified, non-superseded diagnosis_events.
    "total_blunders_analyzed": 0,
    "error_frequencies": {},
    "diagnosis_events": [],     # every memorized diagnosis (newest first), superseded ones included
    "recent_habits": [],
    "recent_positions": [],
    "session_takeaways": [],
    "self_assessment_history": [],
    "drill_history": [],
    "drill_outcomes": [],
    "drill_state": {},
    "drill_sets": [],
    "imported_games": [],
    "typicality_history": [],
    "recurring_patterns": [],   # confirmed cross-game patterns from batch analysis
    "batch_trends": [],         # one entry per batch run: top patterns with counts
    "slide_trend": {},          # cross-game within-game-slide habit (active/improving/stale)
}

_LESSON_DEFAULT = {
    "last_game_id": None,
    "last_agenda_index": 0,
    "lesson_status": "unavailable",
    # Batch lessons get their own resume slot so running /batch never clobbers
    # an unfinished single-game lesson (and vice versa).
    "batch_id": None,
    "batch_agenda_index": 0,
    "batch_lesson_status": "unavailable",
}


def load_lesson_metadata(lesson_file: str) -> dict:
    data = dict(_LESSON_DEFAULT)
    if lesson_file and os.path.exists(lesson_file):
        try:
            with open(lesson_file, "r", encoding="utf-8") as f:
                stored = json.load(f)
            for key in _LESSON_DEFAULT:
                if key in stored:
                    data[key] = stored[key]
        except (json.JSONDecodeError, OSError):
            pass
    return data


def save_lesson_metadata(lesson_file: str, metadata: dict) -> None:
    if not lesson_file:
        return
    data = dict(_LESSON_DEFAULT)
    data.update(metadata or {})
    _atomic_write_json(lesson_file, data)


def _read_stored_dossier(dossier_file: str) -> dict | None:
    if not dossier_file or not os.path.exists(dossier_file):
        return None
    try:
        with open(dossier_file, "r", encoding="utf-8") as f:
            stored = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return stored if isinstance(stored, dict) else None


def _unsupported_schema(version) -> bool:
    """Quality-SOL §7.3: a newer (or unreadable) schema version is read-only for this code."""
    return isinstance(version, bool) or not isinstance(version, int) or version > DOSSIER_SCHEMA_VERSION


def load_dossier(dossier_file: str) -> dict:
    """The long-term dossier, with defaults for missing keys. Quality-SOL §7.3: stored fields this
    code does not know are kept; an older schema is migrated once (one-time backup next to the
    file, then an atomic rewrite); a newer schema is returned as read, and `save_dossier` never
    overwrites it."""
    dossier = copy.deepcopy(_DOSSIER_DEFAULT)
    stored = _read_stored_dossier(dossier_file)
    if stored is None:
        return dossier
    dossier.update(stored)
    version = stored.get("schema_version", 1)
    if _unsupported_schema(version) or version >= DOSSIER_SCHEMA_VERSION:
        return dossier
    _migrate_dossier(dossier, version)
    backup = f"{dossier_file}.v{version}.bak"
    if not os.path.exists(backup):
        shutil.copy2(dossier_file, backup)
    _atomic_write_json(dossier_file, dossier)
    return dossier


def save_dossier(dossier_file: str, dossier: dict) -> bool:
    """Atomic dossier write. False, with nothing written, when there is no dossier file or the file
    on disk has a schema version this code does not support (Quality-SOL §7.3)."""
    if not dossier_file:
        return False
    stored = _read_stored_dossier(dossier_file)
    if stored is not None and _unsupported_schema(stored.get("schema_version", 1)):
        return False
    dossier["schema_version"] = DOSSIER_SCHEMA_VERSION
    _atomic_write_json(dossier_file, dossier)
    return True


def _entry_draw_rule(entry: dict) -> str | None:
    """The draw rule a stored entry's move reached: its stored flag, else a python-chess replay of the
    played move from the FEN. A bare FEN shows only the move-count rules, not a repetition, so an old
    entry that cannot be re-checked stays legacy_unverified. None when nothing is found."""
    if entry.get("draw_rule"):
        return entry["draw_rule"]
    fen, played = entry.get("fen"), entry.get("played")
    if not fen or not played:
        return None
    import chess
    from app.review import _draw_rule
    try:
        board = chess.Board(fen)
        board.push_san(played)
    except ValueError:
        return None
    return _draw_rule(board)


def _supersede_if_drawn(entry: dict, today: str) -> bool:
    """Quality-SOL §7.4 slice E: a diagnosis of a move that reached a draw by rule stays in history,
    marked superseded with the reason, and leaves mistake counts, recommendations and drills."""
    if entry.get("superseded"):
        return True
    rule = _entry_draw_rule(entry)
    if not rule:
        return False
    entry["superseded"] = {
        "reason": f"the move reached a draw by rule ({rule.replace('_', ' ')}); it was not a mistake",
        "draw_rule": rule,
        "date": today,
    }
    return True


def _counts_as_mistake(event: dict) -> bool:
    return (bool(event.get("evidence_version")) and not event.get("superseded")
            and (event.get("error_category") or "") not in ("", "none", "unknown"))


def _refresh_aggregates(dossier: dict) -> None:
    """Trusted counts are regenerated from evidence-qualified, non-superseded events (§7.3)."""
    frequencies = {}
    for event in dossier.get("diagnosis_events") or []:
        if _counts_as_mistake(event):
            frequencies[event["error_category"]] = frequencies.get(event["error_category"], 0) + 1
    dossier["error_frequencies"] = frequencies
    dossier["total_blunders_analyzed"] = sum(frequencies.values())


def _migrate_dossier(dossier: dict, from_version: int) -> None:
    """Quality-SOL §7.3 migration to schema 2. Nothing is dropped: entries without evidence are
    marked legacy_unverified, draw-by-rule diagnoses are superseded (§7.4), and the old totals move
    to `legacy_totals` (as recorded, and net of superseded entries) instead of being relabelled as
    verified counts."""
    today = date.today().isoformat()
    superseded = {}
    for pos in dossier.get("recent_positions") or []:
        if not pos.get("evidence_version"):
            pos.setdefault("evidence_status", LEGACY_UNVERIFIED)
        # Only /memorize entries (no `source`) were counted in the old totals; imported ones were not.
        if _supersede_if_drawn(pos, today) and not pos.get("source"):
            category = pos.get("error_category") or "unknown"
            superseded[category] = superseded.get(category, 0) + 1
    for drill_set in dossier.get("drill_sets") or []:
        drill_set.setdefault("evidence_status", LEGACY_UNVERIFIED)
        for pos in drill_set.get("positions") or []:
            pos.setdefault("evidence_status", LEGACY_UNVERIFIED)
            _supersede_if_drawn(pos, today)
    for pattern in dossier.get("recurring_patterns") or []:
        pattern.setdefault("evidence_status", LEGACY_UNVERIFIED)
        dropped = sum(1 for pos in pattern.get("representative_positions") or []
                      if _supersede_if_drawn(pos, today))
        if dropped:
            pattern["superseded_moments"] = dropped
    recorded = {category: int(n or 0) for category, n in (dossier.get("error_frequencies") or {}).items()}
    recorded_total = int(dossier.get("total_blunders_analyzed") or 0)
    net = {category: n - superseded.get(category, 0) for category, n in recorded.items()}
    dossier["legacy_totals"] = {
        "evidence_status": LEGACY_UNVERIFIED,
        "migrated_from_schema_version": from_version,
        "migrated_on": today,
        "as_recorded": {"total_blunders_analyzed": recorded_total, "error_frequencies": recorded},
        "superseded": superseded,
        "total_blunders_analyzed": max(0, recorded_total - sum(superseded.values())),
        "error_frequencies": {category: n for category, n in net.items() if n > 0},
    }
    _refresh_aggregates(dossier)
    dossier["schema_version"] = DOSSIER_SCHEMA_VERSION


def diagnosis_event_id(diag: dict) -> str | None:
    """Quality-SOL §7.3 event identity: game + ply + student side + evidence version, so teaching
    the same moment twice is one event while the same mistake in another game is another. Without
    a game identity the exact FEN (it carries the move counters) stands in for game and ply.
    Legacy records without evidence have no identity."""
    version = diag.get("evidence_version")
    if not version:
        return None
    if diag.get("game_id") and diag.get("ply") is not None and diag.get("student_side"):
        return f"{diag['game_id']}:{diag['ply']}:{diag['student_side']}:v{version}"
    if diag.get("fen"):
        return f"fen:{diag['fen']}:v{version}"
    return None


def _diagnosis_event(diag: dict, today: str) -> dict:
    event = dict(diag)  # §7.3: unfamiliar evidence fields are kept
    # New diagnoses use the closed taxonomy key; tolerate legacy records.
    event["error_category"] = diag.get("error_category") or diag.get("error_diagnosis") or "unknown"
    event.setdefault("date", today)
    event["event_id"] = diagnosis_event_id(diag)
    if not diag.get("evidence_version"):
        event.setdefault("evidence_status", LEGACY_UNVERIFIED)
    _supersede_if_drawn(event, today)
    return event


def _position_from_event(event: dict) -> dict:
    position = {"fen": event["fen"], "error_category": event["error_category"]}
    for key in ("event_id", "evidence_version", "evidence_status", "category_basis", "game_id", "ply",
                "student_side", "played", "error_typicality", "maia_played_prob"):
        if event.get(key) is not None:
            position[key] = event[key]
    return position


def update_dossier(diagnostics, dossier_file) -> bool:
    """Persist a session's diagnoses (/memorize). Teaching an already memorized moment again is a
    no-op (§7.3 event identity). Returns False when nothing could be written: a dossier with a newer
    schema is never overwritten."""
    dossier = load_dossier(dossier_file)
    today = date.today().isoformat()
    known = {e.get("event_id") for e in dossier["diagnosis_events"] if e.get("event_id")}
    added = []
    for diag in diagnostics:
        event = _diagnosis_event(diag, today)
        if event["event_id"] in known:
            continue
        if event["event_id"]:
            known.add(event["event_id"])
        added.append(event)

    counted = []
    for event in added:
        dossier["diagnosis_events"].insert(0, event)
        if event.get("superseded"):
            continue  # kept in history only (§7.4)
        counted.append(event)
        if event["error_category"] not in ("none", "unknown", "") and event.get("fen"):
            dossier["recent_positions"].insert(0, _position_from_event(event))
        habit = event.get("preventative_habit")
        if habit and habit != "N/A":
            dossier["recent_habits"].insert(0, habit)

    if counted:
        # Progress signal: as patterns get learned, the student's mistakes
        # should drift from peer-common ("typical") to rare slips ("atypical").
        dossier["typicality_history"].append({
            "date": today,
            "typical_errors": sum(1 for d in counted
                                  if d.get("error_typicality") == "typical_for_level"),
            "atypical_errors": sum(1 for d in counted
                                   if d.get("error_typicality") == "atypical_for_level"),
            "total_diagnosed": len(counted),
        })

    dossier["diagnosis_events"] = dossier["diagnosis_events"][:DIAGNOSIS_EVENT_LIMIT]
    dossier["recent_habits"] = dossier["recent_habits"][:5]
    dossier["recent_positions"] = dossier["recent_positions"][:20]
    dossier["typicality_history"] = dossier["typicality_history"][-20:]
    _refresh_aggregates(dossier)
    return save_dossier(dossier_file, dossier)


def record_session_takeaway(dossier_file: str, takeaway: dict) -> None:
    """Persist one verified closing takeaway for future session context."""
    if not dossier_file or not takeaway:
        return
    dossier = load_dossier(dossier_file)
    entry = {
        "date": takeaway.get("date") or date.today().isoformat(),
        "game_id": takeaway.get("game_id"),
        "theme_id": takeaway.get("theme_id"),
        "text": takeaway.get("text"),
        "source": takeaway.get("source", "review_theme"),
    }
    if not entry["text"]:
        return
    existing = [
        t for t in dossier.get("session_takeaways", [])
        if not (t.get("game_id") == entry["game_id"] and t.get("text") == entry["text"])
    ]
    dossier["session_takeaways"] = [entry] + existing
    dossier["session_takeaways"] = dossier["session_takeaways"][:10]
    save_dossier(dossier_file, dossier)


def record_self_assessment(dossier_file: str, assessment: dict) -> None:
    """Persist one review-opening self-assessment result for metacognition trends."""
    if not dossier_file or not assessment:
        return
    dossier = load_dossier(dossier_file)
    entry = {
        "date": assessment.get("date") or date.today().isoformat(),
        "game_id": assessment.get("game_id"),
        "result": assessment.get("result"),
        "top_ply": assessment.get("top_ply"),
        "top_move_label": assessment.get("top_move_label"),
        "matched_ply": assessment.get("matched_ply"),
        "matched_move_label": assessment.get("matched_move_label"),
        "student_text": (assessment.get("student_text") or "")[:240],
    }
    if not entry["result"]:
        return
    existing = [
        item for item in dossier.get("self_assessment_history", [])
        if not (
            item.get("game_id") == entry["game_id"]
            and item.get("result") == entry["result"]
            and item.get("student_text") == entry["student_text"]
        )
    ]
    dossier["self_assessment_history"] = [entry] + existing
    dossier["self_assessment_history"] = dossier["self_assessment_history"][:30]
    save_dossier(dossier_file, dossier)


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").casefold())


def _drill_outcome_matches_pattern(outcome: dict, pattern: dict) -> bool:
    pid = pattern.get("id") or ""
    oid = outcome.get("pattern_id") or ""
    category = outcome.get("error_category") or ""
    if oid and oid == pid:
        return True
    npid = _norm_key(pid)
    ncat = _norm_key(category)
    return bool(ncat and (npid.startswith(ncat) or ncat in _norm_key(pattern.get("title"))))


def drill_summary_for_pattern(dossier: dict, pattern: dict) -> dict:
    """Practice evidence for one recurring pattern.

    This is intentionally separate from the batch recurrence lifecycle: solving
    drills is practice evidence, while fewer future game mistakes are efficacy
    evidence.
    """
    outcomes = [
        o for o in dossier.get("drill_outcomes", [])
        if _drill_outcome_matches_pattern(o, pattern)
    ]
    if not outcomes:
        return {}
    recent = outcomes[:DRILL_RECENT_WINDOW]
    successes = sum(1 for o in outcomes if o.get("success"))
    recent_successes = sum(1 for o in recent if o.get("success"))
    last = outcomes[0]
    state = (dossier.get("drill_state") or {}).get(last.get("error_category") or "") or {}
    return {
        "attempts": len(outcomes),
        "successes": successes,
        "success_rate": round(successes / len(outcomes), 3),
        "recent_attempts": len(recent),
        "recent_successes": recent_successes,
        "recent_success_rate": round(recent_successes / len(recent), 3),
        "last_drilled": last.get("date"),
        "last_result": "success" if last.get("success") else "lapse",
        "last_verdict": last.get("verdict"),
        "memory": {
            "difficulty": state.get("difficulty"),
            "stability_days": state.get("stability_days"),
            "retrievability": state.get("retrievability"),
            "due_after_days": state.get("due_after_days"),
        },
    }


def drill_practice_summary(dossier: dict, limit: int = 3) -> list:
    summaries = []
    seen = set()
    for pattern in dossier.get("recurring_patterns", []):
        summary = drill_summary_for_pattern(dossier, pattern)
        if summary:
            summaries.append({
                "id": pattern.get("id"),
                "title": pattern.get("title"),
                "status": pattern.get("status"),
                **summary,
            })
            seen.add(pattern.get("id"))
    if len(summaries) < limit:
        by_category = {}
        for outcome in dossier.get("drill_outcomes", []):
            category = outcome.get("error_category") or "unknown"
            by_category.setdefault(category, []).append(outcome)
        for category, outcomes in by_category.items():
            if any(_drill_outcome_matches_pattern({"error_category": category}, p)
                   for p in dossier.get("recurring_patterns", [])):
                continue
            synthetic_id = f"category:{category}"
            if synthetic_id in seen:
                continue
            recent = outcomes[:DRILL_RECENT_WINDOW]
            successes = sum(1 for o in outcomes if o.get("success"))
            recent_successes = sum(1 for o in recent if o.get("success"))
            summaries.append({
                "id": synthetic_id,
                "title": category.replace("_", " "),
                "status": "drilled_only",
                "attempts": len(outcomes),
                "successes": successes,
                "success_rate": round(successes / len(outcomes), 3),
                "recent_attempts": len(recent),
                "recent_successes": recent_successes,
                "recent_success_rate": round(recent_successes / len(recent), 3),
                "last_drilled": outcomes[0].get("date"),
                "last_result": "success" if outcomes[0].get("success") else "lapse",
                "last_verdict": outcomes[0].get("verdict"),
                "memory": (dossier.get("drill_state") or {}).get(category) or {},
            })
            if len(summaries) >= limit:
                break
    return summaries[:limit]


def _pattern_recurrence_fields(prev: dict | None, pat: dict, batch_summary: dict,
                               today: str) -> dict:
    history = list((prev or {}).get("recurrence_history") or [])
    current = {
        "date": today,
        "batch_id": batch_summary.get("batch_id"),
        "games": int(batch_summary.get("games") or 0),
        "moments": int(pat.get("count") or 0),
        "rate": pat.get("rate"),
    }
    history.append(current)
    history = history[-12:]

    lifetime_games = sum(int(h.get("games") or 0) for h in history)
    lifetime_moments = sum(int(h.get("moments") or 0) for h in history)
    recent = history[-PATTERN_RECENT_BATCHES:]
    recent_games = sum(int(h.get("games") or 0) for h in recent)
    recent_moments = sum(int(h.get("moments") or 0) for h in recent)
    previous_recent = history[:-1][-PATTERN_RECENT_BATCHES:]
    previous_recent_games = sum(int(h.get("games") or 0) for h in previous_recent)
    previous_recent_moments = sum(int(h.get("moments") or 0) for h in previous_recent)

    def rate(moments, games):
        return round(moments / games, 3) if games else None

    return {
        "recurrence_history": history,
        "lifetime_moments": lifetime_moments,
        "lifetime_games": lifetime_games,
        "lifetime_rate": rate(lifetime_moments, lifetime_games),
        "recent_moments": recent_moments,
        "recent_games": recent_games,
        "recent_rate": rate(recent_moments, recent_games),
        "previous_recent_rate": rate(previous_recent_moments, previous_recent_games),
    }


def _drill_set_for_pattern(prev: dict | None, entry: dict, pat: dict,
                           batch_summary: dict, today: str) -> dict:
    """Build/update a ready drill bank for one recurring batch pattern."""
    pattern_id = entry.get("id") or pat.get("id")
    category = (pattern_id or "").split(":", 1)[0] or "unknown"
    previous_positions = list((prev or {}).get("positions") or [])
    seen = set()
    positions = []

    drill_type = _drill_type_for_pattern(pattern_id)

    def add_position(pos: dict, source_batch_id: str | None):
        fen = pos.get("fen")
        key = (fen, category)
        if not fen or key in seen:
            return
        item = {
            "fen": fen,
            "error_category": category,
            "pattern_id": pattern_id,
            "pattern_title": entry.get("title"),
            "source": "batch_drill_set",
            "drill_type": drill_type,
            "source_batch_id": source_batch_id,
            "move_label": pos.get("move_label"),
            "played": pos.get("played"),
            "better": pos.get("better"),
            "category_basis": pos.get("category_basis") or "rule_based",
        }
        # §7.3/§7.4: provenance and supersession marks survive a rebuild of the drill bank.
        for optional in ("error_typicality", "maia_played_prob", "evidence_status", "superseded"):
            if pos.get(optional) is not None:
                item[optional] = pos[optional]
        positions.append(item)
        seen.add(key)

    for pos in pat.get("representative_positions", []):
        add_position(pos, batch_summary.get("batch_id"))
    for pos in previous_positions:
        add_position(pos, pos.get("source_batch_id") or pos.get("batch_id"))

    return {
        "pattern_id": pattern_id,
        "title": entry.get("title"),
        "status": entry.get("status"),
        "drill_type": drill_type,
        "drill_goal": _drill_goal_for_type(drill_type),
        "axis": pat.get("axis"),
        "habit": entry.get("habit"),
        "last_seen": today,
        "source_batch_id": batch_summary.get("batch_id"),
        "positions": positions[:DRILL_SET_POSITION_LIMIT],
    }


def _drill_type_for_pattern(pattern_id: str) -> str:
    key = _norm_key(pattern_id)
    if key.startswith(("defendagainstdirectthreat", "ignoredthreat", "kingsafety")):
        return "defend"
    if key.startswith(("failuretocapitalize", "missedwinningchance", "missedtactic")):
        return "capitalize"
    if key.startswith(("equalize", "holdthedraw", "tablebase", "defensiveresource")):
        return "equalize"
    return "best_move"


def _drill_goal_for_type(drill_type: str) -> str:
    return {
        "defend": "Find the move that defends or neutralizes the opponent's threat.",
        "capitalize": "Find the move that punishes the opponent's mistake.",
        "equalize": "Find the resource that holds or improves a difficult position.",
        "best_move": "Find the best move and name the pattern.",
    }.get(drill_type or "best_move", "Find the best move and name the pattern.")


def update_dossier_with_batch(dossier_file: str, batch_summary: dict) -> None:
    """Distill a batch run into the long-term player model. Only pattern
    summaries enter the dossier — raw per-moment findings stay in the batch
    report file. Re-running batches updates rates so trends stay honest:
    a pattern whose rate falls is marked 'improving', one absent from two
    consecutive batches goes 'stale' (no permanent accusations)."""
    dossier = load_dossier(dossier_file)
    existing = {p.get("id"): dict(p) for p in dossier.get("recurring_patterns", [])}
    drill_sets = {s.get("pattern_id"): dict(s) for s in dossier.get("drill_sets", [])}
    today = date.today().isoformat()
    seen_ids = set()
    for pat in batch_summary.get("patterns", []):
        seen_ids.add(pat["id"])
        prev = existing.get(pat["id"])
        entry = {
            "id": pat["id"],
            "title": pat["title"],
            "evidence_count": pat["count"],
            "games_seen": pat["games_seen"],
            "representative_positions": list(pat.get("representative_positions", []))[:3],
            "habit": pat.get("habit"),
            "first_seen": prev.get("first_seen") if prev else today,
            "last_seen": today,
            "rate": pat.get("rate"),  # moments per game in this batch
            "previous_rate": prev.get("rate") if prev else None,
            "status": "active",
            "missed_batches": 0,
            # Quality-SOL §7.1: counted from engine rules across games; the category names a
            # provisional lesson label, not a cause the student stated.
            "category_basis": "rule_based",
        }
        entry.update(_pattern_recurrence_fields(prev, pat, batch_summary, today))
        if (entry["previous_rate"] and entry["rate"] is not None
                and entry["rate"] <= entry["previous_rate"] * PATTERN_IMPROVING_FACTOR):
            entry["status"] = "improving"
        elif (entry["previous_rate"] and entry["rate"] is not None
              and entry["rate"] >= entry["previous_rate"] * PATTERN_SLIPPING_FACTOR):
            entry["status"] = "slipping"
        drill_summary = drill_summary_for_pattern(dossier, entry)
        if drill_summary:
            entry["drill_practice"] = drill_summary
        existing[pat["id"]] = entry
        drill_sets[pat["id"]] = _drill_set_for_pattern(
            drill_sets.get(pat["id"]), entry, pat, batch_summary, today
        )
    for pid, p in existing.items():
        if pid not in seen_ids:
            p["missed_batches"] = p.get("missed_batches", 0) + 1
            if p["missed_batches"] >= 2:
                p["status"] = "stale"
            if pid in drill_sets:
                drill_sets[pid]["status"] = p.get("status")
    ranked = sorted(existing.values(),
                    key=lambda p: (p.get("status") == "stale", -(p.get("evidence_count") or 0)))
    dossier["recurring_patterns"] = ranked[:8]
    active_pattern_ids = [p.get("id") for p in dossier["recurring_patterns"]]
    dossier["drill_sets"] = [
        drill_sets[pid] for pid in active_pattern_ids
        if pid in drill_sets and drill_sets[pid].get("positions")
    ][:8]
    dossier["batch_trends"].append({
        "date": today,
        "batch_id": batch_summary.get("batch_id"),
        "games": batch_summary.get("games"),
        "top_patterns": [[p["id"], p["count"]] for p in batch_summary.get("patterns", [])],
    })
    dossier["batch_trends"] = dossier["batch_trends"][-12:]

    # Cross-game slide habit: one evolving entry with the same honest lifecycle as
    # recurring_patterns — a falling rate reads 'improving', two slide-free batches
    # in a row go 'stale' (no permanent accusations).
    prev_slide = dossier.get("slide_trend") or {}
    sig = batch_summary.get("slide_signature")
    if sig:
        rate = sig.get("rate")
        baseline = sig.get("baseline")
        prev_rate = prev_slide.get("rate")
        # Hysteresis on the "ABOVE typical" flag (see SLIDE_ABOVE_BASELINE_OFF):
        # ON at the detector's upper threshold, OFF only below the lower bound,
        # held in between so the message doesn't oscillate on eval noise.
        prev_above = prev_slide.get("above_baseline", False)
        if sig.get("above_baseline"):
            above = True
        elif baseline and rate is not None and rate < baseline * SLIDE_ABOVE_BASELINE_OFF:
            above = False
        else:
            above = prev_above
        status = "active"
        if prev_rate and rate is not None and rate <= prev_rate * 0.6:
            status = "improving"
        dossier["slide_trend"] = {
            "slide_games": sig.get("slide_games"),
            "games": sig.get("games_total"),
            "rate": rate,
            "previous_rate": prev_rate,
            "above_baseline": above,
            "markers": sig.get("markers", []),
            "status": status,
            "first_seen": prev_slide.get("first_seen", today) if prev_slide else today,
            "last_seen": today,
            "missed_batches": 0,
        }
    elif prev_slide:
        prev_slide["missed_batches"] = prev_slide.get("missed_batches", 0) + 1
        if prev_slide["missed_batches"] >= 2:
            prev_slide["status"] = "stale"
        dossier["slide_trend"] = prev_slide

    save_dossier(dossier_file, dossier)

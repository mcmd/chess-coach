import json
import os
import re
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


_DOSSIER_DEFAULT = {
    "total_blunders_analyzed": 0,
    "error_frequencies": {},
    "recent_habits": [],
    "recent_positions": [],
    "session_takeaways": [],
    "self_assessment_history": [],
    "drill_history": [],
    "drill_outcomes": [],
    "drill_state": {},
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


def load_dossier(dossier_file: str) -> dict:
    dossier = dict(_DOSSIER_DEFAULT)
    dossier["error_frequencies"] = {}
    dossier["recent_habits"] = []
    dossier["recent_positions"] = []
    dossier["session_takeaways"] = []
    dossier["self_assessment_history"] = []
    dossier["drill_history"] = []
    dossier["drill_outcomes"] = []
    dossier["drill_state"] = {}
    dossier["typicality_history"] = []
    dossier["recurring_patterns"] = []
    dossier["batch_trends"] = []
    dossier["slide_trend"] = {}
    if os.path.exists(dossier_file):
        try:
            with open(dossier_file, "r", encoding="utf-8") as f:
                stored = json.load(f)
            for key in _DOSSIER_DEFAULT:
                if key in stored:
                    dossier[key] = stored[key]
        except (json.JSONDecodeError, OSError):
            pass
    return dossier


def update_dossier(diagnostics, dossier_file):
    dossier = load_dossier(dossier_file)

    for diag in diagnostics:
        # New diagnoses use the closed taxonomy key; tolerate legacy records.
        category = diag.get("error_category") or diag.get("error_diagnosis") or "unknown"
        if category not in ("none", "unknown", ""):
            dossier["total_blunders_analyzed"] += 1
            dossier["error_frequencies"][category] = dossier["error_frequencies"].get(category, 0) + 1
            if diag.get("fen"):
                position = {"fen": diag["fen"], "error_category": category}
                if diag.get("error_typicality"):
                    position["error_typicality"] = diag["error_typicality"]
                if diag.get("maia_played_prob") is not None:
                    position["maia_played_prob"] = diag["maia_played_prob"]
                dossier["recent_positions"].insert(0, position)

        habit = diag.get("preventative_habit")
        if habit and habit != "N/A":
            dossier["recent_habits"].insert(0, habit)

    if diagnostics:
        # Progress signal: as patterns get learned, the student's mistakes
        # should drift from peer-common ("typical") to rare slips ("atypical").
        dossier["typicality_history"].append({
            "date": date.today().isoformat(),
            "typical_errors": sum(1 for d in diagnostics
                                  if d.get("error_typicality") == "typical_for_level"),
            "atypical_errors": sum(1 for d in diagnostics
                                   if d.get("error_typicality") == "atypical_for_level"),
            "total_diagnosed": len(diagnostics),
        })

    dossier["recent_habits"] = dossier["recent_habits"][:5]
    dossier["recent_positions"] = dossier["recent_positions"][:20]
    dossier["typicality_history"] = dossier["typicality_history"][-20:]

    _atomic_write_json(dossier_file, dossier)


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
    _atomic_write_json(dossier_file, dossier)


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
    _atomic_write_json(dossier_file, dossier)


def update_dossier_with_batch(dossier_file: str, batch_summary: dict) -> None:
    """Distill a batch run into the long-term player model. Only pattern
    summaries enter the dossier — raw per-moment findings stay in the batch
    report file. Re-running batches updates rates so trends stay honest:
    a pattern whose rate falls is marked 'improving', one absent from two
    consecutive batches goes 'stale' (no permanent accusations)."""
    dossier = load_dossier(dossier_file)
    existing = {p.get("id"): dict(p) for p in dossier.get("recurring_patterns", [])}
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
        }
        if (entry["previous_rate"] and entry["rate"] is not None
                and entry["rate"] <= entry["previous_rate"] * 0.6):
            entry["status"] = "improving"
        existing[pat["id"]] = entry
    for pid, p in existing.items():
        if pid not in seen_ids:
            p["missed_batches"] = p.get("missed_batches", 0) + 1
            if p["missed_batches"] >= 2:
                p["status"] = "stale"
    ranked = sorted(existing.values(),
                    key=lambda p: (p.get("status") == "stale", -(p.get("evidence_count") or 0)))
    dossier["recurring_patterns"] = ranked[:8]
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

    _atomic_write_json(dossier_file, dossier)

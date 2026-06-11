import json
import os
import re
from datetime import date

from google.genai import types

# Conversation history is replayed into the model's context on resume —
# keep it bounded so old sessions don't drown new ones.
MAX_HISTORY_MESSAGES = 80


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
    "typicality_history": [],
}


def load_dossier(dossier_file: str) -> dict:
    dossier = dict(_DOSSIER_DEFAULT)
    dossier["error_frequencies"] = {}
    dossier["recent_habits"] = []
    dossier["recent_positions"] = []
    dossier["typicality_history"] = []
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

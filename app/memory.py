import json
import os
from google.genai import types

def load_history(history_file):
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                data = json.load(f)
                contents = []
                for item in data:
                    parts =[types.Part.from_text(text=p["text"]) for p in item["parts"]]
                    contents.append(types.Content(role=item["role"], parts=parts))
                return contents
            except json.JSONDecodeError: return []
    return []

def save_history(chat_session, history_file):
    serializable =[]
    for message in chat_session.get_history():
        parts =[{"text": p.text} for p in message.parts if getattr(p, 'text', None)]
        if parts: serializable.append({"role": message.role, "parts": parts})
    with open(history_file, "w") as f: json.dump(serializable, f)

def update_dossier(diagnostics, dossier_file):
    dossier = {"total_blunders_analyzed": 0, "error_frequencies": {}, "recent_habits": []}
    if os.path.exists(dossier_file):
        try:
            with open(dossier_file, "r") as f: dossier = json.load(f)
        except json.JSONDecodeError: pass
            
    for diag in diagnostics:
        err = diag.get("error_diagnosis", "unknown")
        if err != "none":
            dossier["total_blunders_analyzed"] += 1
            dossier["error_frequencies"][err] = dossier["error_frequencies"].get(err, 0) + 1
        
        habit = diag.get("preventative_habit")
        if habit and habit != "N/A":
            dossier["recent_habits"].insert(0, habit)
            
    dossier["recent_habits"] = dossier["recent_habits"][:5]
    
    with open(dossier_file, "w") as f: json.dump(dossier, f, indent=4)
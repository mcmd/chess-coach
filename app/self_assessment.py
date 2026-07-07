"""Shared review-opening self-assessment helpers.

The command and tool game-loading paths both need the same P0-8 opener state.
Keeping the small state mutation and prompt note here prevents the two paths
from drifting.
"""


def start_self_assessment(state, config, review) -> None:
    if not getattr(config, "review_self_assessment", True):
        return
    agenda = list(getattr(review, "agenda", []) or [])
    if not agenda:
        return
    state.teaching_ledger["self_assessment"] = {
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


def self_assessment_open_note(state, review) -> str:
    sa = state.teaching_ledger.get("self_assessment") or {}
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

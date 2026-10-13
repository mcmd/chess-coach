"""Offline routing eval: today's rules vs the opt-in router model, on recorded student messages.

Each message in tests/fixtures/routing/messages.json is a real student turn (live evals and
manual sessions) with the coach's previous reply, the active agenda moment (if any), and a
hand-assigned gold label. The router is scored on its label, on the move/threat it quotes,
and on what the backend would actually grade with it.

Usage (repo root, needs GEMINI_API_KEY for the router):
    python scripts/eval_routing.py rules
    python scripts/eval_routing.py router gemini-3.8-flash medium [--runs 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.commands import CommandProcessor  # noqa: E402
from app.notation import legal_san_mentions  # noqa: E402
from app.planner import _CLOSE_RE, _DETOUR_RE, _HINT_RE, _normalize_text  # noqa: E402
from app.router import (GRADING_LABELS, Route, Router, describe_moment, resolve_explanation,  # noqa: E402
                        resolve_move, resolve_threat)
from app.tools import _EXPLANATION_GIVE_UP_RE  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/routing/messages.json"


def rules_route(message: dict) -> Route:
    """Only the rules that exist in the code today (no router)."""
    text, norm = message["text"], _normalize_text(message["text"])
    if _HINT_RE.search(norm):
        return Route("hint")
    if CommandProcessor._GIVE_UP_RE.search(text) or _EXPLANATION_GIVE_UP_RE.search(norm):
        return Route("give_up")
    if "?" in text or _DETOUR_RE.search(norm):
        return Route("question")
    if CommandProcessor._MOVE_ON_RE.search(text) or _CLOSE_RE.search(norm):
        return Route("move_on")
    if not message["moment_fen"]:
        return Route("other")
    board = chess.Board(message["moment_fen"])
    own = legal_san_mentions(board, text)
    if len(own) >= 2:
        return Route("candidates")
    if own:
        return Route("move", move=own[0], about_current_moment=True)
    if not board.is_check():
        probe = board.copy(stack=False)
        probe.push(chess.Move.null())
        opp = legal_san_mentions(probe, text)
        if opp:
            return Route("threat", threat=opp[0], about_current_moment=True)
    if message["lesson_status"] == "awaiting_explanation":
        return Route("explanation", explanation=text, about_current_moment=True)
    return Route("other")


def graded_action(message: dict, route: Route | None, move_span: str | None = None,
                  threat_span: str | None = None) -> tuple | None:
    """What the backend grades for this route (mirrors CommandProcessor._pregrade)."""
    if route is None or route.label not in GRADING_LABELS or not message["moment_fen"]:
        return None
    if not route.about_current_moment:
        return None
    board, text = chess.Board(message["moment_fen"]), message["text"]
    if route.label == "move":
        san = resolve_move(board, move_span if move_span is not None else route.move, text)
        return ("move", san) if san else None
    if route.label == "threat":
        san = resolve_threat(board, threat_span if threat_span is not None else route.threat, text)
        return ("threat", san) if san else None
    return ("explanation", None)


def gold_action(message: dict) -> tuple | None:
    gold = message["gold"]
    route = Route(gold["label"], move=gold.get("move"), threat=gold.get("threat"),
                  explanation=message["text"], about_current_moment=True)
    return graded_action(message, route)


def score(messages: list, routes: list) -> dict:
    out = {"labels": 0, "total": len(messages), "graded_right": 0, "graded_wrong": 0,
           "left_to_coach": 0, "gold_gradable": 0, "errors": []}
    for message, route in zip(messages, routes):
        gold_label = message["gold"]["label"]
        label = route.label if route else "router_failed"
        if label == gold_label:
            out["labels"] += 1
        else:
            out["errors"].append((message["id"], gold_label, label, message["text"][:80]))
        should, did = gold_action(message), graded_action(message, route) if route else None
        out["gold_gradable"] += should is not None
        if did is None:
            out["left_to_coach"] += should is not None
        elif did == should:
            out["graded_right"] += 1
        else:
            out["graded_wrong"] += 1
            out["errors"].append((message["id"], f"graded {should}", f"graded {did}", message["text"][:80]))
    return out


def run_router(messages: list, model: str, level: str) -> list:
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(ROOT / ".env")
    router = Router(genai.Client(api_key=os.environ["GEMINI_API_KEY"]), model, level)

    def classify(message):
        moment = (describe_moment(message["moment_fen"], message["moment_label"])
                  if message["moment_fen"] else None)
        return router.classify(message["text"], message["coach_previous"], moment)
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(classify, messages))


def report(name: str, result: dict) -> None:
    print(f"{name}: labels {result['labels']}/{result['total']} | gradable messages {result['gold_gradable']}: "
          f"graded right {result['graded_right']}, graded WRONG {result['graded_wrong']}, "
          f"left to the coach model {result['left_to_coach']}")
    for error in result["errors"]:
        print(f"   #{error[0]} gold={error[1]} got={error[2]}: {error[3]!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["rules", "router"])
    parser.add_argument("model", nargs="?")
    parser.add_argument("level", nargs="?", default="medium")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args(argv)
    messages = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if args.mode == "rules":
        report("rules", score(messages, [rules_route(m) for m in messages]))
        return 0
    if not args.model:
        parser.error("router mode needs a model")
    for run in range(1, args.runs + 1):
        routes = run_router(messages, args.model, args.level)
        versions = sorted({r.model_version for r in routes if r and r.model_version})
        seconds = sorted(r.seconds for r in routes if r)
        median = seconds[len(seconds) // 2] if seconds else None
        report(f"router {args.model} {args.level} run {run} ({versions}, median {median}s with 8 parallel)",
               score(messages, routes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the real coach with newline input and flushed, credential-free evidence.

Instrumentation lives here, not in the coaching implementation. JSONL is written
before notifying the inherited pipe, so a consumer can retain partial evidence
when the model, a tool, or the session fails.
"""
from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from functools import wraps
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Direct execution sets sys.path to scripts/, not the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import chess.engine
import chess.pgn
from rich.text import Text


def _sensitive(key):
    key = str(key).lower().replace("-", "_")
    return any(part in key for part in (
        "api_key", "apikey", "authorization", "password", "credential", "secret",
        "access_token", "refresh_token", "id_token", "x_goog_api_key",
    )) or key in {"key", "cookie", "set_cookie"}


class Serializer:
    """Keep structured SDK/tool/engine data intact, marking unknowns explicitly."""

    def __init__(self):
        self.secrets = set()
        for key, value in os.environ.items():
            # *_TOKEN env vars (e.g. LICHESS_TOKEN) are credentials, but a bare
            # "token" field name is not: board tokens must stay in the evidence.
            if value and (_sensitive(key) or key.upper().endswith("_TOKEN")):
                self.secrets.add(value)

    def text(self, value):
        for secret in sorted(self.secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value

    def url(self, value):
        parts = urlsplit(value)
        host = parts.netloc.rsplit("@", 1)[-1]
        query = [(k, "[REDACTED]" if _sensitive(k) or k.lower() == "token" else v)
                 for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        return self.text(urlunsplit((parts.scheme, host, parts.path,
                                    urlencode(query), parts.fragment)))

    def __call__(self, value, seen=None):
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else {"type": "float", "value": str(value)}
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Path):
            return self.text(str(value))
        if isinstance(value, Enum):
            return self(value.value, seen)
        if isinstance(value, bytes):
            return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
        if isinstance(value, chess.Board):
            return {"fen": value.fen(), "moves_uci": [m.uci() for m in value.move_stack]}
        if isinstance(value, chess.Move):
            return value.uci()
        if isinstance(value, chess.engine.PovScore):
            return {"type": "PovScore", "turn": "white" if value.turn else "black",
                    "relative": self(value.relative), "white": self(value.white()),
                    "black": self(value.black())}
        if isinstance(value, chess.engine.Score):
            return {"type": type(value).__name__, "cp": value.score(), "mate": value.mate()}
        if isinstance(value, chess.pgn.Game):
            return {"headers": self(dict(value.headers)),
                    "pgn": self.text(value.accept(chess.pgn.StringExporter()))}
        if callable(value):
            try:
                signature = str(inspect.signature(value))
            except (ValueError, TypeError):
                signature = None
            return {"type": "callable", "name": getattr(value, "__qualname__", type(value).__name__),
                    "module": getattr(value, "__module__", None),
                    "signature": signature, "doc": self.text(inspect.getdoc(value) or "")}
        seen = set() if seen is None else seen
        identity = id(value)
        if identity in seen:
            return {"type": type(value).__name__, "unavailable": "recursive reference"}
        seen.add(identity)
        try:
            if isinstance(value, Mapping):
                # Field names like "key" also label booleans (e.g. a load-bearing
                # reason flag); only string values can carry a credential.
                return {str(k): self(v, seen) for k, v in value.items()
                        if not (_sensitive(k) and isinstance(v, (str, bytes)))}
            if isinstance(value, (list, tuple, set, frozenset)):
                return [self(v, seen) for v in value]
            if hasattr(value, "model_dump"):
                try:
                    dumped = value.model_dump(mode="json")
                except Exception:
                    dumped = value.model_dump(mode="python")
                return self(dumped, seen)
            if is_dataclass(value) and not isinstance(value, type):
                return self({f.name: getattr(value, f.name) for f in fields(value)}, seen)
            if hasattr(value, "__dict__"):
                return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                        "fields": self(vars(value), seen)}
            return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "unavailable": "no structured serialization interface"}
        except Exception as exc:
            return {"type": type(value).__name__, "unavailable": "serialization failed",
                    "error": self.text(str(exc))}
        finally:
            seen.remove(identity)


class EventWriter:
    def __init__(self, path, event_fd=None):
        self.serializer = Serializer()
        self.file = Path(path).open("w", encoding="utf-8")
        self.pipe = None
        try:
            if event_fd is not None:
                self.pipe = os.fdopen(os.dup(event_fd), "w", encoding="utf-8")
        except BaseException:
            self.file.close()
            raise
        self.id = 0
        self.turn = 0
        self.delivery_error = None
        self.lock = threading.RLock()

    def emit(self, kind, **data):
        with self.lock:
            self.id += 1
            event = {"schema_version": 1, "id": self.id,
                     "timestamp": datetime.now(timezone.utc).isoformat(),
                     "kind": kind, "turn": self.turn, "data": self.serializer(data)}
            line = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
            self.file.write(line)
            self.file.flush()
            if self.pipe is not None:
                try:
                    self.pipe.write(line)
                    self.pipe.flush()
                except OSError as exc:
                    # The durable file remains usable if the consumer disconnects.
                    self.delivery_error = str(exc)
                    try:
                        self.pipe.close()
                    except OSError:
                        pass
                    self.pipe = None
                    self.emit("recording_error", error={"type": type(exc).__name__,
                                                       "message": str(exc)})
            return event

    def close(self):
        try:
            if self.pipe is not None:
                self.pipe.close()
        finally:
            self.file.close()


def _arguments(function, args, kwargs, skip_self=True):
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    return {k: v for k, v in bound.arguments.items() if not (skip_self and k == "self")}


class Recorder:
    def __init__(self, writer, stdin=None):
        self.writer = writer
        self.stdin = sys.stdin if stdin is None else stdin
        self.state = None
        self.hooks = []
        self.signals = {}
        self.pending = False
        self.failed = False
        self.interrupted = False
        self.rendered = False
        self.surface_start = 0
        self.error_surface_start = None
        self.call_counter = 0
        self.unknown_calls = []

    def snapshot(self):
        if self.state is None:
            return {"unavailable": "session state not constructed"}
        state = self.state
        names = ("move_index", "current_game_id", "lesson_metadata", "diagnostics",
                 "socratic_attempts", "review_context", "teaching_ledger", "active_dossier_drill",
                 "grading_integrity_events", "claim_gate_events", "grader_verdict_events",
                 "ambiguous_piece_events", "verdict_contradiction_events", "deliberation_events",
                 "router_events", "pregraded")
        data = {name: getattr(state, name, None) for name in names}
        data.update(board_fen=state.board.fen(),
                    loaded_moves_uci=[move.uci() for move in state.loaded_moves])
        review = getattr(state, "review", None)
        agenda = getattr(review, "agenda", []) if review is not None else []
        agenda = [
            {**self.writer.serializer(record),
             "move_label": getattr(record, "move_label", None),
             "best_move_label": getattr(record, "best_move_label", None)}
            for record in agenda
        ]
        headers = getattr(review, "headers", None)
        data.update(review_agenda=agenda, review_headers=headers,
                    review={"agenda": agenda, "headers": headers,
                            "agenda_plies": getattr(review, "agenda_plies", None),
                            "student_color": getattr(review, "student_color", None)})
        return self.writer.serializer(data)

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        self.hooks.append((owner, name, original))
        @wraps(original)
        def wrapped(*args, **kwargs):
            return replacement(*args, **kwargs)
        # lru_cache's operational methods are descriptors, not __dict__ entries.
        # EngineManager.clear_cache must continue to work through this hook.
        for attribute in ("cache_clear", "cache_info", "cache_parameters"):
            if hasattr(original, attribute):
                setattr(wrapped, attribute, getattr(original, attribute))
        setattr(owner, name, wrapped)
        return original

    def error(self, exc):
        self.failed = True
        self.writer.emit("turn_error", error={"type": type(exc).__name__, "message": str(exc)},
                         state=self.snapshot())
        self.error_surface_start = len(self.console.export_text(clear=False))
        self.pending = False

    def finish_turn(self):
        if self.error_surface_start is not None:
            text = self.console.export_text(clear=False)[self.error_surface_start:]
            if text.strip():
                self.writer.emit("coach_reply", text=text, state=self.snapshot(), surface="error")
            self.error_surface_start = None
        if self.pending:
            if not self.rendered:
                text = self.console.export_text(clear=False)[self.surface_start:]
                self.writer.emit("coach_reply", text=text, state=self.snapshot(), surface="command")
            self.writer.emit("turn_end", state=self.snapshot())
            self.pending = False

    def prompt(self, *args, **kwargs):
        self.finish_turn()
        self.console.print("You:", end=" ")
        self.writer.emit("ready", state=self.snapshot())
        line = self.stdin.readline()
        if line == "":
            # coach.main understands exit, not EOFError; use its normal save/quit path.
            return "exit"
        text = line[:-1] if line.endswith("\n") else line
        if text.endswith("\r") and line.endswith("\n"):
            text = text[:-1]
        history = kwargs.get("history")
        if history is not None:
            history.append_string(text)
        is_turn = bool(text.strip()) and text.strip().lower() not in ("exit", "quit")
        if is_turn:
            self.writer.turn += 1
            self.pending = True
            self.rendered = False
            self.surface_start = len(self.console.export_text(clear=False))
        self.writer.emit("student" if is_turn else "control_input", text=text)
        return text

    def _boundary(self, original, name, category):
        def wrapped(*args, **kwargs):
            self.call_counter += 1
            call_id = f"{category}-{self.call_counter}"
            try:
                arguments = _arguments(original, args, kwargs)
            except TypeError:
                # Invalid model arguments must still reach the original method.
                arguments = {"args": args[1:], "kwargs": kwargs}
            self.writer.emit(f"{category}_call", call_id=call_id, name=name,
                             arguments=arguments, state=self.snapshot())
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                self.writer.emit(f"{category}_result", call_id=call_id, name=name, result=None,
                                 error={"type": type(exc).__name__, "message": str(exc)},
                                 state=self.snapshot())
                raise
            data = {"call_id": call_id, "name": name, "result": result, "state": self.snapshot()}
            if isinstance(result, dict) and "error" in result:
                data["error"] = result["error"]
            self.writer.emit(f"{category}_result", **data)
            return result
        return wrapped

    def __enter__(self):
        import coach
        from app import agent_loop, commands, engines, render, tools
        from google.genai import chats, _api_client
        import requests

        self.console = agent_loop.console
        self.old_record = self.console.record
        self.console.record = True
        try:
            self.patch(coach, "cli_prompt", self.prompt)
            original_config = coach.load_config

            def load_config(*args, **kwargs):
                config = original_config(*args, **kwargs)
                if getattr(config, "api_key", None):
                    self.writer.serializer.secrets.add(config.api_key)
                self.writer.emit("session_start", config=config, model=config.gemini_model)
                return config
            self.patch(coach, "load_config", load_config)
            original_state_init = coach.SessionState.__init__

            def state_init(instance, *args, **kwargs):
                result = original_state_init(instance, *args, **kwargs)
                self.state = instance
                return result
            self.patch(coach.SessionState, "__init__", state_init)

            original_init = tools.CoachTools.__init__

            def init(instance, state, *args, **kwargs):
                self.state = state
                return original_init(instance, state, *args, **kwargs)
            self.patch(tools.CoachTools, "__init__", init)

            original_process = commands.CommandProcessor.process

            def process(instance, user_input):
                self.state = instance.state
                try:
                    result = original_process(instance, user_input)
                    self.writer.emit("turn_start", prompt=result.prompt, state=self.snapshot())
                    return result
                except BaseException as exc:
                    self.error(exc)
                    raise
            self.patch(commands.CommandProcessor, "process", process)

            original_create = chats.Chats.create

            def create(*args, **kwargs):
                self.writer.emit("chat_session", **_arguments(original_create, args, kwargs))
                return original_create(*args, **kwargs)
            self.patch(chats.Chats, "create", create)
            original_extract = agent_loop._extract_function_calls

            def extract(response):
                calls = original_extract(response)
                for call in calls:
                    if not hasattr(tools.CoachTools, call.name):
                        self.call_counter += 1
                        call_id = f"tool-{self.call_counter}"
                        self.unknown_calls.append((call.name, call_id))
                        self.writer.emit("tool_call", call_id=call_id, name=call.name,
                                         arguments=dict(call.args or {}), state=self.snapshot())
                return calls
            self.patch(agent_loop, "_extract_function_calls", extract)

            original_send = agent_loop._send_with_spinner

            def send(chat, payload):
                if isinstance(payload, list):
                    for part in payload:
                        tool_response = getattr(part, "function_response", None)
                        if tool_response is None:
                            continue
                        match = next(((name, call_id) for name, call_id in self.unknown_calls
                                      if name == tool_response.name), None)
                        if match is not None:
                            self.unknown_calls.remove(match)
                            result = (tool_response.response or {}).get("result")
                            self.writer.emit("tool_result", call_id=match[1], name=match[0],
                                             result=result, error=result.get("error") if isinstance(result, dict) else None,
                                             state=self.snapshot())
                history = chat.get_history() if hasattr(chat, "get_history") else None
                self.writer.emit("model_request", payload=payload, history=history)
                response = original_send(chat, payload)
                self.writer.emit("model_response", response=response)
                return response
            self.patch(agent_loop, "_send_with_spinner", send)
            original_request = _api_client.BaseApiClient.request

            def model_http(instance, http_method, path, request_dict, http_options=None):
                # This is the SDK's actual transformed body, including generated
                # tool declarations and systemInstruction. Never record options:
                # those carry authentication headers.
                self.writer.emit("model_request", payload=request_dict,
                                 transport="sdk_http", method=http_method, path=path)
                return original_request(instance, http_method, path, request_dict, http_options)
            self.patch(_api_client.BaseApiClient, "request", model_http)

            original_render = agent_loop._render

            def render_reply(full_text, state, config=None):
                self.state = state
                start = len(self.console.export_text(clear=False))
                try:
                    result = original_render(full_text, state, config)
                except BaseException:
                    self.writer.emit("render_partial", text=self.console.export_text(clear=False)[start:],
                                     state=self.snapshot())
                    raise
                text = self.console.export_text(clear=False)[start:]
                if not full_text or not text.strip():
                    self.error(ValueError("Received no student-visible text response from the model"))
                    self.writer.emit("render_partial", text=text, state=self.snapshot())
                else:
                    self.writer.emit("coach_reply", text=text, state=self.snapshot(), surface="renderer")
                    self.rendered = True
                return result
            self.patch(agent_loop, "_render", render_reply)

            original_turn = coach.run_agent_turn

            def turn(chat, coach_tools, message):
                self.state = coach_tools.state
                try:
                    result = original_turn(chat, coach_tools, message)
                except BaseException as exc:
                    self.error(exc)
                    raise
                self.finish_turn()
                return result
            self.patch(coach, "run_agent_turn", turn)

            original_save = coach.save_history

            def save(*args, **kwargs):
                try:
                    return original_save(*args, **kwargs)
                except BaseException as exc:
                    self.failed = True
                    self.writer.emit("session_error", error={"type": type(exc).__name__, "message": str(exc)})
                    raise
            self.patch(coach, "save_history", save)

            # Keep the advertised tool signatures/annotations intact for Gemini.
            tool_names = [name for name, method in vars(tools.CoachTools).items()
                          if not name.startswith("_") and name != "get_tool_list"
                          and inspect.isfunction(method)]
            for name in tool_names:
                self.patch(tools.CoachTools, name,
                           self._boundary(getattr(tools.CoachTools, name), name, "tool"))
            for name in ("quick_eval", "top_lines", "threat_probe", "maia_predict", "human_predict"):
                self.patch(engines.EngineManager, name,
                           self._boundary(getattr(engines.EngineManager, name), name, "engine"))

            original_board = render.render_board_token

            def board(*args, **kwargs):
                arguments = _arguments(original_board, args, kwargs, skip_self=False)
                token = original_board(*args, **kwargs)
                visual = arguments["state"].board_visuals[token]
                self.writer.emit("board_diagram", token=token, fen=arguments["board"].fen(),
                                 pov=arguments["pov"], text=Text.from_markup(visual).plain, markup=visual)
                return token
            # Aliases imported by tools/review/batch must reference the same wrapper.
            for module in list(sys.modules.values()):
                if module is not None and getattr(module, "__name__", "").startswith("app."):
                    for name, value in list(vars(module).items()):
                        if value is original_board:
                            self.patch(module, name, board)

            original_http = requests.Session.send

            def http(instance, request, **kwargs):
                response = original_http(instance, request, **kwargs)
                # Reading stream=True would consume the caller's stream: explicitly
                # mark that boundary instead of changing the request's semantics.
                streamed = kwargs.get("stream", False) and not response._content_consumed
                self.writer.emit("http_response", url=self.writer.serializer.url(response.url),
                                 status=response.status_code,
                                 params={k: "[REDACTED]" if _sensitive(k) or k.lower() == "token" else v
                                         for k, v in parse_qsl(urlsplit(request.url).query,
                                                              keep_blank_values=True)},
                                 body={"unavailable": "unconsumed streaming response"} if streamed else response.text)
                return response
            self.patch(requests.Session, "send", http)

            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    self.signals[signum] = signal.getsignal(signum)
                    signal.signal(signum, self._interrupt)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _interrupt(self, signum, frame):
        self.interrupted = True
        self.writer.emit("session_interrupt", signal=signum, state=self.snapshot())
        if self.pending:
            self.error(KeyboardInterrupt(f"signal {signum}"))
        raise KeyboardInterrupt

    def __exit__(self, exc_type, exc, tb):
        # Close the turn while the console still records: afterwards its text cannot be exported
        # (live eval 19enjp8m crashed here after an interrupted turn and lost session_end).
        if exc is not None and self.pending:
            self.error(exc)
        if self.error_surface_start is not None:
            self.finish_turn()
        for owner, name, original in reversed(self.hooks):
            setattr(owner, name, original)
        self.hooks.clear()
        for signum, handler in self.signals.items():
            signal.signal(signum, handler)
        self.signals.clear()
        self.console.record = self.old_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--event-fd", type=int)
    args, coach_args = parser.parse_known_args(argv)
    writer = EventWriter(args.events, args.event_fd)
    previous_argv = sys.argv
    recorder = Recorder(writer)
    exit_code = None
    status = "error"
    try:
        writer.emit("launcher_start", launcher="live_coach_session", coach_args=coach_args)
        import coach
        sys.argv = [str(ROOT / "coach.py"), *coach_args]
        with recorder:
            exit_code = coach.main()
            exit_code = 0 if exit_code is None else exit_code
            if recorder.interrupted:
                exit_code = exit_code or 130
            elif recorder.failed or writer.delivery_error:
                exit_code = exit_code or 1
            status = "interrupted" if recorder.interrupted else ("error" if exit_code else "completed")
            if recorder.pending:
                recorder.finish_turn()
        return exit_code
    except BaseException as exc:
        exit_code = (exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int)
                     else 130 if isinstance(exc, KeyboardInterrupt) else 1)
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        writer.emit("session_error", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        sys.argv = previous_argv
        try:
            writer.emit("session_end", state=recorder.snapshot(), status=status, exit_code=exit_code)
        finally:
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Exercise recording at real CLI, SDK, renderer, tool, and HTTP boundaries.

Only the engine transport and model endpoint are deterministic test fixtures;
production launching never substitutes them. Assertions inspect recorded runtime
values and the actual Rich surface, not source text or mock call wiring.
"""
import io
import json
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import chess
import chess.engine
from google import genai
from google.genai import types
from prompt_toolkit.history import FileHistory
import pytest
import requests
from rich.console import Console

import coach
from app import agent_loop, trace
from app.engines import EngineManager
from app.review import GameReview, MoveRecord
from app.state import SessionState, TimelineManager
from app.tools import CoachTools
from scripts.live_coach_session import EventWriter, Recorder, Serializer, main


class EngineTransport:
    """Deterministic analyse transport beneath the actual EngineManager methods."""

    def __init__(self):
        self.closed = False

    def analyse(self, board, limit, multipv=None):
        info = {"score": chess.engine.PovScore(chess.engine.Cp(25), board.turn),
                "pv": [next(iter(board.legal_moves))], "depth": 18,
                "nodes": 12345, "extra": {"untruncated": "engine details " * 1000}}
        return [info] if multipv is not None else info

    def quit(self):
        self.closed = True


@pytest.fixture
def endpoint():
    state = SimpleNamespace(requests=[], replies=[], pgn='[Event "Recorded source"]\n\n1. e4 e5 2. Nf3 *\n')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = state.pgn.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-chess-pgn; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(body)
            if not state.replies:
                code, response = 500, {"error": {"message": "unexpected test request"}}
            else:
                reply = state.replies.pop(0)
                code, response = reply(body) if callable(reply) else reply
            encoded = json.dumps(response).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.fixture
def surface(monkeypatch):
    console = Console(file=io.StringIO(), width=100, force_terminal=False, color_system=None)
    # Every app alias refers to this console in a normal launch. Mirror that
    # identity without changing renderer behavior or swallowing its output.
    import sys
    from app.ui import console as original
    for module in list(sys.modules.values()):
        if module is not None and (getattr(module, "__name__", "").startswith("app.") or module is coach):
            for name, value in list(vars(module).items()):
                if value is original:
                    monkeypatch.setattr(module, name, console)
    monkeypatch.setattr(trace, "_enabled", False)
    return console


def local_client(endpoint):
    return genai.Client(api_key="local-test-credential",
                        http_options=types.HttpOptions(base_url=endpoint.url, api_version="v1beta",
                                                       retry_options=types.HttpRetryOptions(attempts=1)))


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def response(parts):
    return 200, {"candidates": [{"content": {"role": "model", "parts": parts},
                                "finishReason": "STOP"}],
                 "usageMetadata": {"promptTokenCount": 57, "candidatesTokenCount": 19,
                                   "totalTokenCount": 76}}


@pytest.fixture
def real_main_dependencies(monkeypatch, endpoint, stub_config, surface):
    client = local_client(endpoint)
    transport = EngineTransport()
    monkeypatch.setattr(coach.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(coach, "load_config", lambda: stub_config)
    monkeypatch.setattr(EngineManager, "start", lambda manager: setattr(manager, "stockfish", transport))
    try:
        yield client, transport
    finally:
        client.close()


def test_real_main_records_sdk_tools_diagrams_and_rendered_surface(
    tmp_path, monkeypatch, endpoint, stub_config, surface, real_main_dependencies
):
    hidden = "private scratchpad " * 1200
    endpoint.replies.append(response([
        {"text": hidden},
        {"functionCall": {"id": "sdk-call", "name": "analyze_position",
                          "args": {"fen": chess.STARTING_FEN, "moves_to_apply": ["e4"], "pov": "black"}}},
    ]))

    def final_reply(body):
        result = next(part["functionResponse"]["response"]["result"]
                      for content in body["contents"] for part in content["parts"]
                      if "functionResponse" in part)
        token = result["board_token_after_moves"]
        return response([{"text": f"<think>{hidden}</think>\nA real diagram follows. {token}\n[[BOARD_unknown]]"}])

    endpoint.replies.append(final_reply)
    monkeypatch.setattr("sys.stdin", io.StringIO("/help\nWhat should I inspect?\n"))
    path = tmp_path / "events.jsonl"
    # Drain the actual pipe concurrently: complete tool declarations can exceed
    # pipe capacity, and the protocol deliberately flushes without truncation.
    read_fd, write_fd = os.pipe()
    received = []

    def drain():
        with os.fdopen(read_fd, encoding="utf-8") as stream:
            received.extend(json.loads(line) for line in stream)

    consumer = threading.Thread(target=drain)
    consumer.start()
    try:
        code = main(["--events", str(path), "--event-fd", str(write_fd), "--profile=fresh", "--trace"])
    finally:
        os.close(write_fd)
        consumer.join(timeout=10)
    recorded = events(path)
    assert code == 0
    assert not consumer.is_alive()
    assert received == recorded
    assert [event["id"] for event in recorded] == list(range(1, len(recorded) + 1))
    assert all(event["schema_version"] == 1 and "+00:00" in event["timestamp"] for event in recorded)
    assert [e["turn"] for e in recorded if e["kind"] == "ready"] == [0, 1, 2]
    assert [e["data"]["text"] for e in recorded if e["kind"] == "student"] == [
        "/help", "What should I inspect?"
    ]
    replies = [e["data"] for e in recorded if e["kind"] == "coach_reply"]
    assert "Commands" in replies[0]["text"] and replies[0]["surface"] == "command"
    rendered = replies[1]["text"]
    assert "Coach:" in rendered and "A real diagram follows." in rendered
    assert "private scratchpad" not in rendered and "<think>" not in rendered
    assert "invented board token(s) scrubbed" in rendered
    assert "diagram unavailable" in rendered and "BOARD_" not in rendered
    assert "♟" in rendered or "♙" in rendered
    raw = [e["data"]["response"] for e in recorded if e["kind"] == "model_response"]
    assert raw[0]["candidates"][0]["content"]["parts"][0]["text"] == hidden
    assert raw[0]["usage_metadata"]["total_token_count"] == 76
    assert "<think>" in raw[1]["candidates"][0]["content"]["parts"][0]["text"]
    diagrams = [e["data"] for e in recorded if e["kind"] == "board_diagram"]
    after = chess.Board()
    after.push_san("e4")
    assert [d["fen"] for d in diagrams] == [chess.STARTING_FEN, after.fen()]
    assert all(d["pov"] == "black" and d["token"].startswith("[[BOARD_") for d in diagrams)
    tool_result = next(e["data"] for e in recorded if e["kind"] == "tool_result")
    assert tool_result["result"]["board_token_after_moves"] == diagrams[1]["token"]
    assert tool_result["result"]["top_stockfish_lines"][0]["expected_engine_line"]
    engine_calls = {e["data"]["call_id"]: e["data"] for e in recorded if e["kind"] == "engine_call"}
    engine_results = [e["data"] for e in recorded if e["kind"] == "engine_result"]
    assert engine_results and all(e["call_id"] in engine_calls for e in engine_results)
    top = next(e for e in engine_results if e["name"] == "top_lines")
    assert top["result"][0]["extra"]["untruncated"] == "engine details " * 1000
    assert engine_calls[top["call_id"]]["arguments"]["board"]["fen"] == after.fen()
    constructed = next(e["data"] for e in recorded if e["kind"] == "chat_session")
    assert constructed["config"]["system_instruction"]
    assert constructed["config"]["tools"] and constructed["history"] == []
    transformed = next(e["data"]["payload"] for e in recorded
                       if e["kind"] == "model_request" and e["data"].get("transport") == "sdk_http")
    assert transformed["tools"] == endpoint.requests[0]["tools"]
    assert transformed["systemInstruction"] == endpoint.requests[0]["systemInstruction"]
    assert any(declaration["name"] == "analyze_position"
               for tool in transformed["tools"] for declaration in tool["functionDeclarations"])
    start = next(e["data"] for e in recorded if e["kind"] == "session_start")
    assert start["model"] == stub_config.gemini_model and "api_key" not in start["config"]
    assert stub_config.api_key not in path.read_text()
    assert "local-test-credential" not in path.read_text()
    assert recorded[-1]["data"]["status"] == "completed"
    assert real_main_dependencies[1].closed
    assert history_strings(stub_config.cli_history_file) == ["/help", "What should I inspect?"]


def history_strings(path):
    return list(reversed(list(FileHistory(path).load_history_strings())))


def test_model_failure_is_partial_readable_and_not_success(
    tmp_path, monkeypatch, endpoint, real_main_dependencies
):
    endpoint.replies.append((400, {"error": {"code": 400, "message": "deliberate SDK failure",
                                           "status": "INVALID_ARGUMENT"}}))
    monkeypatch.setattr("sys.stdin", io.StringIO("Explain this position\n"))
    path = tmp_path / "events.jsonl"
    assert main(["--events", str(path)]) == 1
    recorded = events(path)
    error = next(e for e in recorded if e["kind"] == "turn_error")
    assert "deliberate SDK failure" in error["data"]["error"]["message"]
    assert error["data"]["state"]["board_fen"] == chess.STARTING_FEN
    reply = next(e for e in recorded if e["kind"] == "coach_reply")
    visible = " ".join(reply["data"]["text"].split())
    assert "An API error occurred:" in visible and "deliberate SDK failure" in visible
    assert reply["data"]["surface"] == "error"
    assert not any(e["kind"] == "turn_end" for e in recorded)
    assert recorded[-1]["kind"] == "session_end"
    assert recorded[-1]["data"]["status"] == "error" and recorded[-1]["data"]["exit_code"] == 1
    assert real_main_dependencies[1].closed


def test_raw_thought_without_visible_reply_is_an_error(
    tmp_path, monkeypatch, endpoint, real_main_dependencies
):
    endpoint.replies.append(response([{"text": "<think>private reasoning only</think>"}]))
    monkeypatch.setattr("sys.stdin", io.StringIO("Explain this\n"))
    path = tmp_path / "events.jsonl"
    assert main(["--events", str(path)]) == 1
    recorded = events(path)
    assert any(e["kind"] == "model_response" for e in recorded)
    assert any(e["kind"] == "turn_error" for e in recorded)
    assert not any(e["kind"] in ("coach_reply", "turn_end") for e in recorded)
    partial = next(e["data"] for e in recorded if e["kind"] == "render_partial")
    assert not partial["text"].strip()
    assert recorded[-1]["data"]["status"] == "error"


def test_unknown_tool_records_actual_backend_response(
    tmp_path, monkeypatch, endpoint, real_main_dependencies
):
    endpoint.replies.append(response([{
        "functionCall": {"name": "unavailable_tool", "args": {"question": "details " * 2000}}
    }]))
    endpoint.replies.append(response([{"text": "That analysis tool is unavailable."}]))
    monkeypatch.setattr("sys.stdin", io.StringIO("Please analyze this\n"))
    path = tmp_path / "events.jsonl"
    assert main(["--events", str(path)]) == 0
    recorded = events(path)
    call = next(e["data"] for e in recorded if e["kind"] == "tool_call")
    result = next(e["data"] for e in recorded if e["kind"] == "tool_result")
    assert call["arguments"]["question"] == "details " * 2000
    assert result["call_id"] == call["call_id"]
    assert result["result"] == {"error": "Tool unavailable_tool not found."}
    sent = next(e["data"]["payload"] for e in recorded
                if e["kind"] == "model_request" and isinstance(e["data"]["payload"], list))
    assert sent[0]["function_response"]["response"]["result"] == result["result"]
    assert result["error"] == result["result"]["error"]
    assert next(e["data"]["text"] for e in recorded if e["kind"] == "coach_reply").strip().endswith(
        "That analysis tool is unavailable."
    )


def test_actual_tool_error_result_is_not_hidden(tmp_path, stub_config, surface):
    writer = EventWriter(tmp_path / "events.jsonl")
    state = SessionState()
    try:
        with Recorder(writer):
            tools = CoachTools(state, TimelineManager(state), EngineManager(stub_config), stub_config)
            actual_result = tools.get_game_review_summary()
        result = next(e["data"] for e in events(tmp_path / "events.jsonl")
                      if e["kind"] == "tool_result")
        assert result["result"] == actual_result
        assert result["error"] == actual_result["error"]
        assert result["state"]["review_agenda"] == []
    finally:
        writer.close()


def test_prompt_preserves_whitespace_history_and_eof(tmp_path, surface):
    path = tmp_path / "events.jsonl"
    history = FileHistory(str(tmp_path / "stdin.history"))
    writer = EventWriter(path)
    try:
        with Recorder(writer, io.StringIO("  Nf3?  \r\n\nexit\n")) as recorder:
            assert coach.cli_prompt("ignored", history=history) == "  Nf3?  "
            assert coach.cli_prompt("ignored", history=history) == ""
            assert coach.cli_prompt("ignored", history=history) == "exit"
            assert coach.cli_prompt("ignored", history=history) == "exit"
            recorded = events(path)  # readable before closing the writer
            students = [e for e in recorded if e["kind"] == "student"]
            assert [e["data"]["text"] for e in students] == ["  Nf3?  "]
            assert [e["turn"] for e in students] == [1]
            controls = [e for e in recorded if e["kind"] == "control_input"]
            assert [e["data"]["text"] for e in controls] == ["", "exit"]
            assert all(e["turn"] == 1 for e in controls)
            assert history_strings(history.filename) == ["  Nf3?  ", "", "exit"]
    finally:
        writer.close()


def test_tool_exception_flushes_and_restores_hooks(tmp_path, stub_config, surface):
    path = tmp_path / "events.jsonl"
    original_prompt = coach.cli_prompt
    original_render = agent_loop._render
    original_analysis = CoachTools.analyze_position
    writer = EventWriter(path)
    state = SessionState()
    try:
        with pytest.raises(TypeError, match="unexpected"):
            with Recorder(writer):
                tools = CoachTools(state, TimelineManager(state), EngineManager(stub_config), stub_config)
                tools.analyze_position(fen=chess.STARTING_FEN, unexpected=True)
        recorded = events(path)
        result = next(e["data"] for e in recorded if e["kind"] == "tool_result")
        assert result["result"] is None
        assert result["error"]["type"] == "TypeError"
        assert "unexpected" in result["error"]["message"]
        assert result["state"]["board_fen"] == state.board.fen()
        assert coach.cli_prompt is original_prompt and agent_loop._render is original_render
        assert CoachTools.analyze_position is original_analysis
        assert not surface.record
    finally:
        writer.close()


def test_http_records_actual_source_without_headers_or_credentials(tmp_path, endpoint, surface):
    path = tmp_path / "events.jsonl"
    endpoint.pgn += "{ full source evidence " + "x" * 18000 + " }\n"
    writer = EventWriter(path)
    try:
        with Recorder(writer):
            response_value = requests.get(endpoint.url + "/game", params={"evals": "true", "api_key": "sensitive-query"},
                                          headers={"Authorization": "Bearer secret-header"}, timeout=3)
            assert response_value.text == endpoint.pgn
        recorded = events(path)
        data = next(e["data"] for e in recorded if e["kind"] == "http_response")
        assert data["body"] == endpoint.pgn and data["status"] == 200
        assert data["params"]["evals"] == "true"
        query = parse_qs(urlsplit(data["url"]).query)
        assert query["api_key"] == ["[REDACTED]"]
        assert "sensitive-query" not in path.read_text() and "secret-header" not in path.read_text()
        assert "headers" not in data
    finally:
        writer.close()


def test_engine_cache_and_agenda_state_are_actual_values(tmp_path, stub_config, surface):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    state = SessionState()
    after = state.board.copy()
    after.push_san("e4")
    record = MoveRecord(0, 1, "white", "e4", "e2e4", state.board.fen(), after.fen(),
                        0, 25, 0.0, "good", best_line=["e4", "e5"], punishment_line=["c5"])
    state.review = GameReview({"Event": "real agenda"}, [record], "white", [0])
    state.loaded_moves = [chess.Move.from_uci("e2e4")]
    state.diagnostics.append({"fen": state.board.fen(), "category": "tactics"})
    state.claim_gate_events.append({"tokens": ["Qh5"], "redacted": True})
    try:
        with Recorder(writer) as recorder:
            tools = CoachTools(state, TimelineManager(state), EngineManager(stub_config), stub_config)
            tools.engines.stockfish = EngineTransport()
            tools.engines.clear_cache()
            first = tools.engines.quick_eval(state.board.fen())
            second = tools.engines.quick_eval(state.board.fen())
            assert first is second
            assert tools.engines.quick_eval.cache_info().hits >= 1
            snapshot = recorder.snapshot()
            assert snapshot["loaded_moves_uci"] == ["e2e4"]
            assert snapshot["review_agenda"][0]["move_label"] == "1. e4"
            assert snapshot["review_agenda"][0]["best_line"] == ["e4", "e5"]
            assert snapshot["review_agenda"][0]["fen_after"] == after.fen()
            assert snapshot["review_headers"] == {"Event": "real agenda"}
            assert snapshot["claim_gate_events"][0]["redacted"] is True
        results = [e["data"]["result"] for e in events(path) if e["kind"] == "engine_result"]
        assert results[0][0]["white"]["cp"] == 25
        assert results[0][1] in [m.uci() for m in state.board.legal_moves]
    finally:
        EngineManager.quick_eval.cache_clear()
        writer.close()


def test_error_pipe_flush_and_interrupt_leave_partial_evidence(tmp_path, surface):
    path = tmp_path / "events.jsonl"
    read_fd, write_fd = os.pipe()
    writer = EventWriter(path, write_fd)
    previous = signal.getsignal(signal.SIGINT)
    try:
        with os.fdopen(read_fd, encoding="utf-8") as consumer:
            with Recorder(writer) as recorder:
                recorder.pending = True
                with pytest.raises(KeyboardInterrupt):
                    recorder._interrupt(signal.SIGINT, None)
                interrupted = json.loads(consumer.readline())
                failed = json.loads(consumer.readline())
                assert interrupted["kind"] == "session_interrupt"
                assert failed["kind"] == "turn_error"
                assert failed["data"]["error"]["type"] == "KeyboardInterrupt"
                assert events(path) == [interrupted, failed]
                assert recorder.failed and not recorder.pending
        assert signal.getsignal(signal.SIGINT) is previous
    finally:
        writer.close()
        os.close(write_fd)


def test_serializer_keeps_structured_unknowns_and_no_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sensitive-value")
    serializer = Serializer()
    value = serializer({"api_key": "sensitive-value", "board_token": "[[BOARD_aabbccdd]]",
                        "note": "do not leak sensitive-value", "unknown": object(),
                        "full": {"entries": list(range(2000))}})
    assert "api_key" not in value
    assert value["board_token"] == "[[BOARD_aabbccdd]]"
    assert "sensitive-value" not in json.dumps(value)
    assert value["unknown"]["unavailable"] == "no structured serialization interface"
    assert value["full"]["entries"] == list(range(2000))


def test_serializer_keeps_reason_key_flags_but_drops_string_credentials():
    value = Serializer()({"reasons": [{"text": "Rd4 defends g4", "key": True}],
                          "key": "AIza-credential", "api_key": "other-credential"})
    assert value["reasons"] == [{"text": "Rd4 defends g4", "key": True}]
    assert "key" not in value and "api_key" not in value


def test_serializer_redacts_lichess_token_from_environment(monkeypatch):
    monkeypatch.setenv("LICHESS_TOKEN", "lip_secret_value")
    value = Serializer()({"note": "explorer called with lip_secret_value"})
    assert "lip_secret_value" not in json.dumps(value)

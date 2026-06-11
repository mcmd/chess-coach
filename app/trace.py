"""Lightweight execution tracing for understanding the coach's reasoning.

When enabled (--trace or "trace": true in config), every tool call the LLM
makes and every underlying engine invocation (Stockfish, Maia-2) is printed
as a dim line. The bulk per-move review scan suspends engine-level tracing
and reports summaries instead, so a 60-move game doesn't flood the terminal.
"""
from contextlib import contextmanager

from app.ui import console

_enabled = False
_suspended = 0


def set_enabled(value: bool) -> None:
    global _enabled
    _enabled = bool(value)


def is_enabled() -> bool:
    return _enabled and _suspended == 0


def trace(message: str) -> None:
    if is_enabled():
        # ASCII-safe prefix: legacy Windows consoles (cp1252) crash on fancier glyphs.
        console.print(f"[dim cyan]· trace |[/dim cyan] [dim]{message}[/dim]")


@contextmanager
def suspended():
    """Temporarily silence tracing (used by bulk scans that would flood)."""
    global _suspended
    _suspended += 1
    try:
        yield
    finally:
        _suspended -= 1


def fmt_args(kwargs: dict, limit: int = 180) -> str:
    parts = []
    for k, v in kwargs.items():
        text = repr(v)
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{k}={text}")
    joined = ", ".join(parts)
    return joined if len(joined) <= limit else joined[:limit - 3] + "..."

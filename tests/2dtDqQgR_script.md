# Player script: Lichess 2dtDqQgR

Scenario ID: `lichess-2dtDqQgR-full-v2`.
Role: pass-1 simulated student for a complete game review: load, self-assessment, the
agenda moments, questions about earlier positions, and post-review questions.
Built from the user's manual session of 2026-10-12. v2 (2026-10-16): 31. Kh1 drew by
threefold repetition, so it is no longer an agenda moment; the student asks about it instead.

## Your job

Continue the conversation using the messages below. You are not the evaluator.
Do not check move legality, consult an engine, verify percentages, judge the
coach, or read reviewer material. The runner handles setup, recording, waiting,
and shutdown; the second pass evaluates the frozen recording.

Use only the student-visible conversation and this script to select the next
message. Send only the text in the applicable **Send** block, never the gates
or these instructions.

- Match gates by meaning, not exact wording. Moves may be written with or without
  check signs or move numbers.
- Wait for a complete coach response before choosing a message.
- Keep the messages as written, including typos.
- Do not invent chess answers, repair the coach's analysis, ask for hints, or
  steer around an incompatible gate to make the session succeed.
- A questionable explanation is not a reason to stop if the next gate fits:
  record the conversation unchanged and let the reviewer assess it.
- If no gate fits, return `STOPPED_UNEXPECTED` with the last coach message and
  current step. Do not improvise. A timeout is handled by the runner.
- If the coach offers a rating change, leave the configured Elo unchanged and
  continue only if the normal question is also present.
- `**Checks:**` lines are for the offline checker; ignore them.

## Conversation

### S0 — Load

**Ready when:** the fresh coach session shows its first input prompt.

**Send:**
```text
Load the Lichess game 2dtDqQgR
```

### S1 — Self-assessment

**Ready when:** the coach asks which move you think was your biggest mistake.

**Send:**
```text
I think when I took that pawn with my queen, in the late middlegame
```

**Allowed branch S1a:** if the coach then shows White's move 17 (after `16...e5`) before
move 20, send this once, then wait for S2's gate:
```text
Let's skip this one and go to the next mistake.
```

### S2 — Move 20: threat

**Ready when:** the coach shows White's move 20 (after `19...Rfe8`) and asks for
candidate moves and/or Black's threat.

**Send:**
```text
black is threatening e4
```

### S3 — Move 20: ask about the threat

**Ready when:** the coach has responded to your threat answer for move 20.

**Checks:** detour

**Send:**
```text
Can you expand on why e4 is a threat? What's Black's plan?
```

### S4 — Move 20: candidate and reason

**Ready when:** the coach has answered your question and move 20 is still unsolved
(the better move has not been revealed).

**Send:**
```text
I think I should moe Ne4 to block their pawn's advancement, and to gain tempo by attacking their queen
```

**Allowed branch S4a:** if the next response separately asks why `Ne4` works,
send this once, then wait for S5's gate:
```text
It blocks their pawn's advancement and gains tempo by attacking their queen.
```

### S5 — Back to move 20

**Ready when:** the coach has finished grading move 20 (it may already show move 24).

**Checks:** detour

**Send:**
```text
Wait, go back to move 20 for a moment. Why is Ne4 better than taking on b5 with the knight? What happens next, and what would humans usually do?
```

### S6 — Still on move 20

**Ready when:** the coach has answered your question about move 20.

**Checks:** detour

**Send:**
```text
Still on move 20: after Ne4, is moving the queen literally forced, just the engine's choice, or simply what people usually do?
```

### S7 — Return to move 24

**Ready when:** the coach has answered your follow-up about move 20.

**Checks:** detour, return 24

**Send:**
```text
Thanks, that's clear. Let's go back to move 24.
```

### S8 — Move 24: a legal but inferior move

**Ready when:** the coach shows White's move 24 (after `23...fxg2`) and asks for
candidate moves and/or Black's threat.

**Send:**
```text
I would play Kxg2
```

### S9 — Move 24: hint

**Ready when:** the coach has responded to `Kxg2`.

**Send:**
```text
Give me one small hint without telling me the move.
```

### S10 — Move 24: an illegal move

**Ready when:** the coach has given a hint or asked you to try again.

**Send:**
```text
Then I would play Qh5
```

**Allowed branch S10a:** only if the coach says `Qh5` is not legal and asks you to try again
without naming the better move. If the coach has already named `Rfe1` (or any better move),
do not send S10a: go straight to S11. Send it once, then wait for S11's gate:
```text
I don't know. Please show me the better move.
```

### S11 — Move 24: reason

**Ready when:** the coach has revealed or confirmed the better move for move 24.

**Send:**
```text
it moves my rook to safety, and offers a rook trade
```

### S12 — Move 24: clarification

**Ready when:** the coach has responded to your explanation. It may already have moved on
to move 26 (the grader can tell it to teach the line instead of probing); send this anyway:
it then tests answering a question about the previous moment without losing the current one.

**Checks:** detour

**Send:**
```text
Before we move on, what do you mean by that?
```

### S13 — Ready for the next moment

**Ready when:** the coach has answered your clarification question.

**Send:**
```text
Thanks, I'm ready for the next moment.
```

### S14 — Move 26: undecided between candidates

**Ready when:** the coach shows White's move 26 (after `25...Qg6`) and asks for
candidate moves and/or Black's threat.

**Send:**
```text
I'm considering f3 or Rd4, but I haven't chosen yet.
```

**Allowed branch S14a:** if the coach asks what Black is threatening before you choose,
send this once, then wait for S15's gate:
```text
Black is threatening Qxg4 with check.
```

### S15 — Move 26: commit

**Ready when:** the coach asks you to choose one move.

**Send:**
```text
Rd4
```

### S16 — Move 26: reason

**Ready when:** the coach has responded to `Rd4` and asks why it works (or what it does).

**Send:**
```text
because that one actually defends my g pawn
```

**Allowed branch S16a:** if the coach asks what piece covers g4 or along which
line, send this once, then wait for S17's gate:
```text
The rook on d4 covers g4 along the fourth rank.
```

### S17 — Finish move 26

**Ready when:** the coach has responded to your explanation for move 26, even if the same reply
already summarizes or closes the review (send this anyway; never skip to S18).

**Send:**
```text
Great, next moment please.
```

**Allowed branch S17a:** if the coach goes back to the skipped move 17 (after `16...e5`) instead of
finishing, send this once, then wait for S18's gate:
```text
No thanks, I'd rather skip move 17. I'm done with the review.
```

### S18 — Move 31: a misconception

**Ready when:** the coach has responded to your request for the next moment (it may have
finished the review or summarized it) or to your wish to skip move 17.

**Checks:** detour

**Send:**
```text
At move 31 I played Kh1. Shouldn't I have played Kh2 to avoid the perpetual?
```

### S19 — Move 31: follow-up

**Ready when:** the coach has answered your question about move 31.

**Checks:** detour

**Send:**
```text
So was Kh1 actually a good move?
```

### S20 — Language switch

**Ready when:** the coach has answered your question.

**Send:**
```text
Repeat that in Spanish, briefly.
```

### S21 — Finish the review

**Ready when:** the coach has repeated its answer in Spanish.

**Send:**
```text
OK, let's finish the review.
```

### S22 — Full annotation request

**Ready when:** the coach has finished the last moment or summarized the review.

**Send:**
```text
Annotate every move of the game.
```

### S23 — White's pawn structure

**Ready when:** the coach has declined or answered the annotation request.

**Checks:** detour

**Send:**
```text
How did my pawn structure evolve throughout the game?
```

### S24 — Black's pawn structure

**Ready when:** the coach has described your pawn structure.

**Checks:** detour

**Send:**
```text
And how did Black's pawn structure evolve?
```

### S25 — Open versus half-open files

**Ready when:** the coach has described Black's pawn structure.

**Checks:** detour

**Send:**
```text
Did the g- and h-files actually become open, or were only my pawns removed?
```

### S26 — Moves relative to my level

**Ready when:** the coach has answered about the files.

**Checks:** detour

**Send:**
```text
Which of my moves were better or worse than expected for a player at my level?
```

### S27 — Risk

**Ready when:** the coach has answered about your moves.

**Checks:** detour

**Send:**
```text
How risky was the last position: was I playing for a win or a draw?
```

### S28 — Takeaway

**Ready when:** the coach has answered about risk.

**Send:**
```text
What should I remember and practice from this session?
```

### S29 — What is saved

**Ready when:** the coach has given a takeaway.

**Checks:** detour

**Send:**
```text
What has actually been saved? Do I need /memorize?
```

### S30 — Stop

**Ready when:** the coach has answered what is saved.

Do not continue. Return `SCRIPT_COMPLETE` with the exact final coach message and
the last sent step. The runner closes the disposable session and hands the
recording to the reviewer.

## Player handoff

Report only execution facts, not PASS/FAIL or coaching-quality scores:

- Scenario ID and last sent step.
- Sent step IDs in order; include branches (S1a, S4a, S10a, S14a, S16a, S17a) if used.
- Any unexpected gate and the exact coach message that did not fit.
- `SCRIPT_COMPLETE` or `STOPPED_UNEXPECTED` (the runner can instead record
  `TIMED_OUT` or `INFRASTRUCTURE_BLOCKED`).

Do not call an incomplete conversation complete. Do not summarize away or edit
coach mistakes; the runner must preserve the original output for pass 2.

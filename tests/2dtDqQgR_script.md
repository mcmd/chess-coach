# Player script: Lichess 2dtDqQgR

Scenario ID: `lichess-2dtDqQgR-detour-v1`.
Role: pass-1 simulated student. This is a partial coaching session, not a
complete review of the game.

## Your job

Continue the conversation using the messages below. You are not the evaluator.
Do not check move legality, consult an engine, verify percentages, judge the
coach, or read reviewer material. The runner handles setup, recording, waiting,
and shutdown; the second pass evaluates the frozen recording.

Use only the student-visible conversation and this script to select the next
message. Do not use hidden tool results, the coach's instructions, or grading
criteria. Send only the text in the applicable **Send** block, never the gates
or these instructions.

- Match gates by meaning, not exact wording.
- Wait for a complete coach response before choosing a message.
- Keep the messages as written, including the original typo in S3.
- Do not invent chess answers, repair the coach's analysis, ask for hints, or
  steer around an incompatible gate to make the session succeed.
- A questionable explanation is not a reason to stop if the next gate fits:
  record the conversation unchanged and let the reviewer assess it.
- If no gate fits, return `STOPPED_UNEXPECTED` with the last coach message and
  current step. Do not improvise. A timeout is handled by the runner.
- If the coach offers a rating change, leave the configured Elo unchanged and
  continue only if the normal question is also present. An offer requiring a
  separate answer is outside this script; report it as unexpected.

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

### S2 — Threat answer

**Ready when:** the coach shows White's move 20 after `19...Rfe8` and asks for
candidate moves and/or Black's threat.

**Send:**
```text
black is threatening e4
```

### S3 — Candidate and reason

**Ready when:** the coach asks for a White candidate in that same move-20 position.

**Send:**
```text
I think I should moe Ne4 to block their pawn's advancement, and to gain tempo by attacking their queen
```

**Allowed branch S3a:** if the next response separately asks why `Ne4` works,
send this once, then wait for S4's gate:
```text
It blocks their pawn's advancement and gains tempo by attacking their queen.
```
Record that S3a was used. A request for another chess answer is outside the script.

### S4 — Ask about the previous position

**Ready when:** the coach has reached White's move 24 after `23...fxg2`.

**Send:**
```text
Wait, go back to the previous position for a second. Can you elaborate a bit more on why Ne4 is the best? What happens next? What would humans do in that situation, and what would be a reasonable expectation for me as a follow-up sequence?
```

### S5 — Return to the pending position

**Ready when:** the coach has finished responding to S4. Send this whether it
already returned to move 24 or is asking whether the explanation is clear.
This is navigation, not a chess answer.

**Send:**
```text
Thanks. Let's return to move 24, just after 23...fxg2. Remind me of the position and the question we were working on, without giving me the answer.
```

### S6 — Stop

**Ready when:** the coach has finished responding to S5.

There is no supplied answer to move 24. Do not continue playing. Return
`SCRIPT_COMPLETE` with the exact final coach message and the last sent step.
The runner closes the disposable session and hands the recording to the reviewer.

## Player handoff

Report only execution facts, not PASS/FAIL or coaching-quality scores:

- Scenario ID and last sent step.
- Sent step IDs in order; include S3a if used.
- Any unexpected gate and the exact coach message that did not fit.
- `SCRIPT_COMPLETE` or `STOPPED_UNEXPECTED` (the runner can instead record
  `TIMED_OUT` or `INFRASTRUCTURE_BLOCKED`).

Do not call an incomplete conversation complete. Do not summarize away or edit
coach mistakes; the runner must preserve the original output for pass 2.

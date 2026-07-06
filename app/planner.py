"""Lightweight teaching-act planner (P0-8).

This rules-first layer handles obvious mode contracts before the model sees a
free-text turn. It keeps routing decisions explicit, inspectable, and testable
without turning the existing coach into a new pipeline.
"""
import re
import unicodedata
from dataclasses import dataclass, field


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold()


_LANGUAGE_PATTERNS = [
    (re.compile(r"\b(?:in|en|to)\s+spanish\b|\ben\s+espa(?:n|ñ)ol\b", re.I), "Spanish"),
    (re.compile(r"\b(?:in|en|to)\s+english\b|\ben\s+ingl[eé]s\b", re.I), "English"),
    (re.compile(r"\b(?:in|en|to)\s+french\b|\ben\s+franc[eé]s\b", re.I), "French"),
    (re.compile(r"\b(?:in|en|to)\s+german\b|\ben\s+alem[aá]n\b", re.I), "German"),
    (re.compile(r"\b(?:in|en|to)\s+portuguese\b|\ben\s+portugu[eé]s\b", re.I), "Portuguese"),
    (re.compile(r"\b(?:in|en|to)\s+italian\b|\ben\s+italiano\b", re.I), "Italian"),
]

_ANNOTATE_ALL_RE = re.compile(
    r"\b(?:annotate|comment(?:ary)?|review|analyse|analyze|explain)\b"
    r"(?:(?!\n).){0,60}\b(?:every|all|each)\b(?:(?!\n).){0,30}\bmove(?:s)?\b"
    r"|\bmove[-\s]?by[-\s]?move\b(?:(?!\n).){0,40}\b(?:annotation|commentary|review|analysis)\b"
    r"|\b/annotate_pgn\b",
    re.I,
)
_QUIZ_RE = re.compile(r"\b(?:quiz me|test me|ask me|socratic|withhold(?: the)? answer)\b", re.I)
_DIRECT_RE = re.compile(
    r"\b(?:explain directly|tell me directly|don'?t quiz me|do not quiz me|just tell me|show me the answer)\b",
    re.I,
)
_SUMMARY_RE = re.compile(
    r"\b(?:summari[sz]e|summary|overview)\b(?:(?!\n).){0,50}\b(?:phase by phase|phase-by-phase|whole game|game)\b",
    re.I,
)
_HINT_RE = re.compile(r"\b(?:hint|nudge|clue)\b", re.I)
_COMPARE_RE = re.compile(r"\b(?:compare|candidate(?:s)?|which of|between)\b", re.I)
_STRUCTURE_RE = re.compile(r"\b(?:pawn structure|structure|plan|strategy)\b", re.I)
_TRANSLATE_RE = re.compile(r"\b(?:translate|say that|repeat that|same thing)\b(?:(?!\n).){0,40}\b(?:spanish|english|french|german|portuguese|italian|espa(?:n|Ã±)ol|ingl[eÃ©]s|franc[eÃ©]s|alem[aÃ¡]n|portugu[eÃ©]s|italiano)\b", re.I)
_EXERCISE_RE = re.compile(r"\b(?:exercise|drill|practice|train me|give me a position|make me solve)\b", re.I)
_BLUNDER_RE = re.compile(r"\b(?:why|explain|what happened|what went wrong)\b(?:(?!\n).){0,50}\b(?:blunder|mistake|inaccuracy|bad move)\b", re.I)
_STRENGTH_RE = re.compile(r"\b(?:what did i do well|good move|best move|strong move|where was i accurate|reinforce|praise)\b", re.I)
_BRIEF_RE = re.compile(r"\b(?:briefly|brief|concise|short answer|quickly|tl;dr|tldr)\b", re.I)
_VERBOSE_RE = re.compile(r"\b(?:in detail|detailed|thorough|step by step|deep dive)\b", re.I)
_CLOSE_RE = re.compile(
    r"\b(?:wrap up|finish(?: up)?|close(?: it| this| the session)?|done for now|that'?s enough|"
    r"what should i remember|main takeaway|one takeaway|takeaway)\b",
    re.I,
)

# Normalized multilingual routing overrides. These regexes run against
# _normalize_text(text), so accented input such as "español" and "inglés" is
# matched by the ASCII forms below.
_LANGUAGE_PATTERNS = [
    (re.compile(r"\b(?:in|en|to|al|a)\s+(?:spanish|espanol)\b|\bespanol\b", re.I), "Spanish"),
    (re.compile(r"\b(?:in|en|to|al|a)\s+(?:english|ingles)\b|\bingles\b", re.I), "English"),
    (re.compile(r"\b(?:in|en|to|al|a)\s+(?:french|frances)\b|\bfrances\b", re.I), "French"),
    (re.compile(r"\b(?:in|en|to|al|a)\s+(?:german|aleman)\b|\baleman\b", re.I), "German"),
    (re.compile(r"\b(?:in|en|to|al|a)\s+(?:portuguese|portugues)\b|\bportugues\b", re.I), "Portuguese"),
    (re.compile(r"\b(?:in|en|to|al|a)\s+italian\b|\bitaliano\b", re.I), "Italian"),
]
_QUIZ_RE = re.compile(
    r"\b(?:quiz me|test me|ask me|socratic|withhold(?: the)? answer|"
    r"ponme a prueba|preguntame|hazme preguntas|modo socratico)\b",
    re.I,
)
_DIRECT_RE = re.compile(
    r"\b(?:explain directly|tell me directly|don'?t quiz me|do not quiz me|just tell me|show me the answer|"
    r"explica(?:melo)? directamente|dimelo directamente|no me preguntes|no me hagas preguntas|"
    r"solo dime|solo cuentame|muestrame la respuesta)\b",
    re.I,
)
_SUMMARY_RE = re.compile(
    r"\b(?:summari[sz]e|summary|overview|resum(?:e|en|eme|ir)|resumen|vista general)\b"
    r"(?:(?!\n).){0,50}\b(?:phase by phase|phase-by-phase|whole game|game|fase por fase|partida)\b",
    re.I,
)
_HINT_RE = re.compile(r"\b(?:hint|nudge|clue|pista|ayuda|dame una pista)\b", re.I)
_COMPARE_RE = re.compile(r"\b(?:compare|candidate(?:s)?|which of|between|compara|comparar|candidata(?:s)?|entre)\b", re.I)
_STRUCTURE_RE = re.compile(r"\b(?:pawn structure|structure|plan|strategy|estructura de peones|estructura|plan|estrategia)\b", re.I)
_TRANSLATE_RE = re.compile(
    r"\b(?:translate|say that|repeat that|same thing|traduce(?:lo)?|dilo|repitelo|lo mismo)\b"
    r"(?:(?!\n).){0,40}\b(?:spanish|english|french|german|portuguese|italian|"
    r"espanol|ingles|frances|aleman|portugues|italiano)\b",
    re.I,
)
_EXERCISE_RE = re.compile(
    r"\b(?:exercise|drill|practice|train me|give me a position|make me solve|"
    r"ejercicio|practica|entrename|dame una posicion|ponme una posicion)\b",
    re.I,
)
_BLUNDER_RE = re.compile(
    r"\b(?:why|explain|what happened|what went wrong|por que|explica|que paso|que salio mal)\b"
    r"(?:(?!\n).){0,50}\b(?:blunder|mistake|inaccuracy|bad move|error|fallo|mala jugada|imprecision)\b",
    re.I,
)
_STRENGTH_RE = re.compile(
    r"\b(?:what did i do well|good move|best move|strong move|where was i accurate|reinforce|praise|"
    r"que hice bien|buena jugada|mejor jugada|jugada fuerte|donde estuve preciso|refuerza|elogia)\b",
    re.I,
)
_BRIEF_RE = re.compile(r"\b(?:briefly|brief|concise|short answer|quickly|tl;dr|tldr|breve|brevemente|conciso|rapido)\b", re.I)
_VERBOSE_RE = re.compile(r"\b(?:in detail|detailed|thorough|step by step|deep dive|en detalle|detallado|paso a paso)\b", re.I)
_CLOSE_RE = re.compile(
    r"\b(?:wrap up|finish(?: up)?|close(?: it| this| the session)?|done for now|that'?s enough|"
    r"what should i remember|main takeaway|one takeaway|takeaway|terminemos|acabemos|cierra(?: la sesion)?|"
    r"por hoy basta|ya esta|que debo recordar|conclusion|idea principal)\b",
    re.I,
)


@dataclass
class TeachingPlan:
    intent: str
    language: str
    density: str = "normal"
    withhold_answer: bool = False
    socratic_redirect: bool = False
    no_tail_redirect: bool = False
    decline: bool = False
    decline_message: str = None
    reasons: list[str] = field(default_factory=list)

    def prompt_block(self) -> str:
        if self.decline:
            return ""
        lines = [
            "[Backend Teaching Plan - P0-8]",
            f"intent: {self.intent}",
            f"language: {self.language}",
            f"density: {self.density}",
            f"withhold_answer: {str(self.withhold_answer).lower()}",
            f"socratic_redirect: {str(self.socratic_redirect).lower()}",
            f"no_tail_redirect: {str(self.no_tail_redirect).lower()}",
        ]
        if self.reasons:
            lines.append("routing_reasons: " + ", ".join(self.reasons))
        lines.extend([
            "Policy: Follow this backend plan for this turn. It overrides the default session language "
            "only for this turn when language differs from config.",
            "If no_tail_redirect=true, answer the requested thing and stop; do not append an "
            "unrequested quiz, agenda jump, or 'now let's look at...' redirect.",
            "If socratic_redirect=false or withhold_answer=false, answer directly instead of turning "
            "the request into a quiz.",
            "[/Backend Teaching Plan]",
        ])
        return "\n".join(lines)


@dataclass
class TeachingPlanObedienceFinding:
    code: str
    detail: str


_TAIL_REDIRECT_RESPONSE_RE = re.compile(
    r"\b(?:now let'?s|let'?s now|next,?\s+let'?s|let'?s look at|let'?s zoom in|"
    r"move on to|zoom in on|turn to move|your turn|what candidate moves|what would you "
    r"(?:play|consider|do)|can you find|try to find)\b",
    re.I,
)
_UNREQUESTED_QUIZ_RESPONSE_RE = re.compile(
    r"\b(?:what (?:move|candidate|would|should|is)|which (?:move|candidate)|why (?:does|is|was)|"
    r"can you (?:find|see|tell)|try to (?:find|explain|calculate))\b[^.?!]*\?",
    re.I,
)


def assess_teaching_plan_obedience(
    plan: TeachingPlan | dict,
    response_text: str,
    *,
    withheld_answers: list[str] | None = None,
) -> list[TeachingPlanObedienceFinding]:
    """Deterministic transcript checks for the P0-8 mode contract.

    This is not a semantic quality judge. It catches the concrete regressions the
    plan identified: direct/no-tail turns that append a new quiz or agenda jump,
    and Socratic turns that leak a known withheld answer token.
    """
    text = response_text or ""
    if not text.strip():
        return []

    def field(name: str, default=None):
        if isinstance(plan, dict):
            return plan.get(name, default)
        return getattr(plan, name, default)

    findings: list[TeachingPlanObedienceFinding] = []
    if field("no_tail_redirect", False) and _TAIL_REDIRECT_RESPONSE_RE.search(text):
        findings.append(TeachingPlanObedienceFinding(
            "tail_redirect",
            "Response appended an unrequested quiz, agenda jump, or 'let us look at' redirect.",
        ))

    if ((not field("socratic_redirect", False) or not field("withhold_answer", False))
            and _UNREQUESTED_QUIZ_RESPONSE_RE.search(text)):
        findings.append(TeachingPlanObedienceFinding(
            "unrequested_quiz",
            "Response asked a Socratic question even though the plan requested directness.",
        ))

    if field("withhold_answer", False):
        lower = text.lower()
        for answer in withheld_answers or []:
            token = str(answer).strip()
            if token and re.search(rf"(?<![A-Za-z0-9_]){re.escape(token.lower())}(?![A-Za-z0-9_])", lower):
                findings.append(TeachingPlanObedienceFinding(
                    "withheld_answer_leak",
                    f"Response revealed withheld answer token '{token}'.",
                ))
                break

    return findings


def _language_from_text(text: str, default_language: str) -> tuple[str, str | None]:
    text = _normalize_text(text)
    for pattern, language in _LANGUAGE_PATTERNS:
        if pattern.search(text):
            return language, f"language_override:{language}"
    return default_language, None


def plan_free_text_turn(text: str, *, default_language: str = "English") -> TeachingPlan:
    """Classify obvious free-text mode requests before the LLM sees them."""
    normalized_text = _normalize_text(text)
    language, language_reason = _language_from_text(normalized_text, default_language)
    reasons = []
    if language_reason:
        reasons.append(language_reason)

    density = "normal"
    if _BRIEF_RE.search(normalized_text):
        density = "brief"
        reasons.append("density:brief")
    elif _VERBOSE_RE.search(normalized_text):
        density = "verbose"
        reasons.append("density:verbose")

    if _ANNOTATE_ALL_RE.search(normalized_text):
        return TeachingPlan(
            intent="decline_full_game_annotation",
            language=language,
            density=density,
            decline=True,
            decline_message=(
                "I can't annotate every move inside the interactive coach. "
                "Load or review the game and we'll work through the critical moments one at a time; "
                "a separate `/annotate_pgn` pipeline is future work, not this mode."
            ),
            reasons=reasons + ["decline:full_game_annotation"],
        )

    if _CLOSE_RE.search(normalized_text):
        return TeachingPlan(
            intent="close_with_takeaway",
            language=language,
            density=density,
            no_tail_redirect=True,
            reasons=reasons + ["intent:close_with_takeaway"],
        )
    if _SUMMARY_RE.search(normalized_text):
        return TeachingPlan(
            intent="explain_phase_summary",
            language=language,
            density=density,
            no_tail_redirect=True,
            reasons=reasons + ["intent:phase_summary"],
        )
    if _TRANSLATE_RE.search(normalized_text):
        return TeachingPlan(
            intent="translate_previous",
            language=language,
            density=density,
            no_tail_redirect=True,
            reasons=reasons + ["intent:translate_previous"],
        )
    if _DIRECT_RE.search(normalized_text):
        return TeachingPlan(
            intent="answer_directly",
            language=language,
            density=density,
            no_tail_redirect=True,
            reasons=reasons + ["intent:direct"],
        )
    if _QUIZ_RE.search(normalized_text):
        return TeachingPlan(
            intent="ask_socratic_question",
            language=language,
            density=density,
            withhold_answer=True,
            socratic_redirect=True,
            reasons=reasons + ["intent:quiz"],
        )
    if _HINT_RE.search(normalized_text):
        return TeachingPlan(
            intent="give_hint",
            language=language,
            density=density,
            withhold_answer=True,
            socratic_redirect=True,
            no_tail_redirect=True,
            reasons=reasons + ["intent:hint"],
        )
    if _EXERCISE_RE.search(normalized_text):
        return TeachingPlan(
            intent="generate_exercise",
            language=language,
            density=density,
            withhold_answer=True,
            socratic_redirect=True,
            no_tail_redirect=True,
            reasons=reasons + ["intent:generate_exercise"],
        )
    if _COMPARE_RE.search(normalized_text):
        return TeachingPlan(
            intent="compare_candidates",
            language=language,
            density=density,
            no_tail_redirect=True,
            reasons=reasons + ["intent:compare_candidates"],
        )
    if _BLUNDER_RE.search(normalized_text):
        return TeachingPlan(
            intent="explain_blunder",
            language=language,
            density=density,
            no_tail_redirect=True,
            reasons=reasons + ["intent:explain_blunder"],
        )
    if _STRENGTH_RE.search(normalized_text):
        return TeachingPlan(
            intent="reinforce_strength",
            language=language,
            density=density,
            no_tail_redirect=True,
            reasons=reasons + ["intent:reinforce_strength"],
        )
    if _STRUCTURE_RE.search(normalized_text):
        return TeachingPlan(
            intent="explain_plan_or_structure",
            language=language,
            density=density,
            no_tail_redirect=True,
            reasons=reasons + ["intent:plan_or_structure"],
        )

    return TeachingPlan(
        intent="answer_directly",
        language=language,
        density=density,
        reasons=reasons + ["intent:default_free_text"],
    )

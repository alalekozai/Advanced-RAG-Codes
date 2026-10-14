"""PII detection and redaction.

Finds common PII in text and swaps it for a typed placeholder like [EMAIL].
Runs on the user's question (before it reaches the LLM), on retrieved chunks,
and on the model's answer (before it reaches the user).

Regex-based on purpose: fast, no model download, easy to audit.
Known limit: it does not catch names or street addresses. For those, swap in
Microsoft Presidio (NER-based) behind the same detect()/redact() interface.
"""

import re
from dataclasses import dataclass


@dataclass
class PIIMatch:
    kind: str
    start: int
    end: int


def _luhn_ok(number: str) -> bool:
    """Credit card checksum. Cuts false positives on random 16-digit numbers."""
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Order matters: more specific patterns first, so a card number isn't
# half-eaten by the phone pattern.
PATTERNS = {
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

VALIDATORS = {"CREDIT_CARD": _luhn_ok}


def detect(text: str, enabled: list[str] | None = None) -> list[PIIMatch]:
    """Return non-overlapping PII matches, in text order."""
    kinds = enabled or list(PATTERNS)
    taken: list[PIIMatch] = []
    for kind in kinds:
        for m in PATTERNS[kind].finditer(text):
            check = VALIDATORS.get(kind)
            if check and not check(m.group()):
                continue
            if any(m.start() < t.end and t.start < m.end() for t in taken):
                continue
            taken.append(PIIMatch(kind, m.start(), m.end()))
    return sorted(taken, key=lambda x: x.start)


def redact(text: str, enabled: list[str] | None = None) -> tuple[str, list[PIIMatch]]:
    """Replace each PII match with [KIND]. Returns (clean_text, matches)."""
    matches = detect(text, enabled)
    out = text
    for m in reversed(matches):  # back to front so offsets stay valid
        out = out[: m.start] + f"[{m.kind}]" + out[m.end :]
    return out, matches

"""Prompt-injection and jailbreak screening.

Two places attacks come from in RAG:
  1. Direct: the user types "ignore your instructions..."
  2. Indirect: a document in the vector store contains hidden instructions,
     and retrieval pulls it into the prompt.

This is a rule-based first layer. Each rule has a weight; the scores add up,
and anything at or above the policy threshold is blocked. Rules are cheap
and explainable, but attackers can reword around them, so in production
you'd add a classifier model (e.g. Llama Prompt Guard) as a second layer.
"""

import re
import unicodedata
from dataclasses import dataclass, field

RULES: list[tuple[str, float, re.Pattern]] = [
    ("override_instructions", 0.9, re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b.{0,20}\b(instructions?|rules|prompts?|context)\b", re.I)),
    ("reveal_system_prompt", 0.8, re.compile(
        r"\b(reveal|show|print|repeat|output|tell me)\b.{0,40}\b(system prompt|hidden instructions|initial prompt|your instructions)\b", re.I)),
    ("role_hijack", 0.6, re.compile(
        r"\b(you are now|from now on you are|act as|pretend (to be|you are))\b", re.I)),
    ("jailbreak_persona", 0.8, re.compile(
        r"\b(DAN|do anything now|developer mode|jailbreak|no restrictions|without (any )?(filters|restrictions|rules))\b", re.I)),
    ("fake_role_tag", 0.7, re.compile(
        r"(</?\s*(system|assistant)\s*>|^\s*(system|assistant)\s*:|\[\s*system\s*\])", re.I | re.M)),
    ("exfiltration", 0.7, re.compile(
        r"\b(send|post|upload|exfiltrate|forward)\b.{0,40}\b(https?://|to (this|my) (url|email|server|webhook))", re.I)),
    ("encoded_payload", 0.4, re.compile(r"\b[A-Za-z0-9+/]{60,}={0,2}\b")),
]


@dataclass
class ScanResult:
    score: float
    hits: list[str] = field(default_factory=list)

    def blocked(self, threshold: float) -> bool:
        return self.score >= threshold


def _normalize(text: str) -> str:
    """Undo common tricks: full-width letters, zero-width spaces, extra whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[​-‏⁠﻿]", "", text)
    return re.sub(r"\s+", " ", text)


def scan(text: str) -> ScanResult:
    clean = _normalize(text)
    hits = [name for name, _, pat in RULES if pat.search(clean)]
    score = min(1.0, sum(w for name, w, _ in RULES if name in hits))
    return ScanResult(score=score, hits=hits)

"""Strict GSM8K-style final-answer parsing.

Qwen3 thinking mode uses the added tokens ``<think>`` (151667) and
``</think>`` (151668); the closing marker is ASCII ``</think>``, verified
against the pinned tokenizer. The parser never treats a number from unfinished
reasoning as a final answer:

* thinking mode requires a closing ``</think>``; text before it is reasoning;
* only explicit markers in the answer region are accepted:
  ``#### <number>``, ``\\boxed{<number>}``, or a line beginning ``Answer:``;
* conflicting markers are ``ambiguous``; unparseable markers are ``malformed``;
* absent markers are ``missing``; never fall back to the last number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_FRACTION = re.compile(r"-?\d+/\d+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH = re.compile(r"####\s*([^\n]*)")
_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*:\s*([^\n]*)$")


@dataclass
class ParsedAnswer:
    status: str  # ok | missing | ambiguous | malformed | incomplete_thinking
    value: str | None
    raw: str | None
    marker: str | None
    answer_region: str
    thinking_closed: bool

    @property
    def explicit(self) -> bool:
        return self.status == "ok"


def split_thinking(text: str) -> tuple[str, str, bool]:
    """Split at the last ``</think>``; returns (reasoning, answer_region, closed)."""
    index = text.rfind(THINK_CLOSE)
    if index < 0:
        return text, "", False
    start = index + len(THINK_CLOSE)
    return text[:index], text[start:], True


def normalize_number(raw: str) -> str | None:
    """Normalize an explicitly marked numeric answer; None if not numeric."""
    text = raw.strip()
    boxed = _BOXED.fullmatch(text)
    if boxed:
        text = boxed.group(1)
    # strip surrounding punctuation commonly used in final answers
    text = text.strip().strip("()[]{} ").rstrip(".").strip()
    text = text.replace(",", "").replace("$", "")
    fraction = _FRACTION.fullmatch(text)
    if fraction:
        numerator, denominator = text.split("/")
        try:
            value = float(numerator) / float(denominator)
        except ZeroDivisionError:
            return None
        return _format_float(value)
    if _NUMBER.fullmatch(text):
        return _format_float(float(text))
    return None


def _format_float(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _collect(region: str) -> tuple[list[tuple[str, str]], bool]:
    """Return (marker, raw_value) candidates and whether any malformed marker exists."""
    candidates: list[tuple[str, str]] = []
    malformed = False
    for match in _HASH.finditer(region):
        raw = match.group(1).strip()
        candidates.append(("####", raw))
    for match in _BOXED.finditer(region):
        candidates.append(("\\boxed{}", match.group(1)))
    for match in _ANSWER_LINE.finditer(region):
        candidates.append(("Answer:", match.group(1).strip()))
    return candidates, malformed


def parse_final_answer(text: str, thinking_mode: bool) -> ParsedAnswer:
    """Parse only an explicitly marked final answer."""
    if thinking_mode:
        _, region, closed = split_thinking(text)
        if not closed:
            return ParsedAnswer(
                status="incomplete_thinking",
                value=None,
                raw=None,
                marker=None,
                answer_region="",
                thinking_closed=False,
            )
    else:
        region = text
        closed = True

    candidates, _ = _collect(region)
    if not candidates:
        return ParsedAnswer(
            status="missing",
            value=None,
            raw=None,
            marker=None,
            answer_region=region,
            thinking_closed=closed,
        )

    values: list[str] = []
    malformed: list[str] = []
    markers: list[str] = []
    for marker, raw in candidates:
        value = normalize_number(raw)
        markers.append(marker)
        if value is None:
            malformed.append(raw)
        else:
            values.append(value)

    if len(set(values)) > 1:
        return ParsedAnswer(
            status="ambiguous",
            value=None,
            raw=" | ".join(raw for _, raw in candidates),
            marker=",".join(sorted(set(markers))),
            answer_region=region,
            thinking_closed=closed,
        )
    if not values:
        return ParsedAnswer(
            status="malformed",
            value=None,
            raw=" | ".join(malformed),
            marker=markers[-1],
            answer_region=region,
            thinking_closed=closed,
        )
    return ParsedAnswer(
        status="ok",
        value=values[-1],
        raw=candidates[-1][1],
        marker=candidates[-1][0],
        answer_region=region,
        thinking_closed=closed,
    )

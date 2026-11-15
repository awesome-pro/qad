"""Strict GSM8K-style final-answer parsing (grader v2).

Qwen3 thinking mode uses the added tokens ``<think>`` (151667) and
``</think>`` (151668); the closing marker is ASCII ``</think>``, verified
against the pinned tokenizer. The parser never treats a number from unfinished
reasoning as a final answer:

* thinking mode requires a closing ``</think>``; text before it is reasoning;
* only the **last non-empty line** of the answer region may carry a marker;
* accepted markers: ``#### <number>``, a line consisting of one
  ``\\boxed{<number>}`` (optionally after ``Answer:``), or a line beginning
  ``Answer: <number>``;
* intermediate ``####``/boxed values on earlier lines are ignored;
* conflicting valid/malformed markers are rejected, never resolved by taking
  the first valid value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
PARSER_VERSION = "2"

_FRACTION = re.compile(r"-?\d+/\d+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")

_HASH = re.compile(r"####[ \t]*([^\n]*)")
_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$")


@dataclass
class ParsedAnswer:
    status: str  # ok | missing | ambiguous | malformed | incomplete_thinking
    value: str | None
    raw: str | None
    marker: str | None
    answer_region: str
    thinking_closed: bool
    final_line: str = ""

    @property
    def explicit(self) -> bool:
        return self.status == "ok"

    @property
    def marker_present(self) -> bool:
        return self.status in {"ok", "ambiguous", "malformed"}


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


_LATEX_DELIMITERS = {"$$", "\\[", "\\]", "\\(", "\\)"}


def _last_nonempty_line(region: str) -> str:
    """Last content line, ignoring trailing LaTeX display-math delimiters."""
    for line in reversed(region.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _LATEX_DELIMITERS:
            continue
        return line
    return ""


def _region_conflicts(region: str) -> tuple[list[str], list[str]]:
    """All ####/Answer: markers in the region, to detect conflicting attempts.

    Boxed values are deliberately excluded: only a boxed value on the final
    line can be a final answer, so earlier boxed calculations are ignored.
    """
    values: list[str] = []
    malformed: list[str] = []
    for match in _HASH.finditer(region):
        value = normalize_number(match.group(1).strip())
        values.append(value) if value is not None else malformed.append(match.group(1).strip())
    for match in _ANSWER_LINE.finditer(region):
        value = normalize_number(match.group(1).strip())
        values.append(value) if value is not None else malformed.append(match.group(1).strip())
    return values, malformed


def _boxed_candidates(final_line: str) -> list[tuple[str, str]]:
    """Boxed values that can be final: no numeric token after the last box.

    Accepts display-math decorations and units after the box, but rejects an
    intermediate box followed by more arithmetic.
    """
    matches = list(_BOXED.finditer(final_line))
    if not matches:
        return []
    last = matches[-1]
    tail = final_line[last.end():]
    if _NUMBER.search(tail) or _FRACTION.search(tail):
        return []
    return [("\\boxed{}", match.group(1)) for match in matches]


def _candidates(final_line: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for match in _HASH.finditer(final_line):
        candidates.append(("####", match.group(1).strip()))
    candidates.extend(_boxed_candidates(final_line))
    answer = _ANSWER_LINE.fullmatch(final_line)
    if answer:
        candidates.append(("Answer:", answer.group(1).strip()))
    return candidates


def parse_final_answer(text: str, thinking_mode: bool) -> ParsedAnswer:
    """Parse only an explicitly marked final answer on the final line."""
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

    final_line = _last_nonempty_line(region)
    candidates = _candidates(final_line)
    region_values, region_malformed = _region_conflicts(region)
    if len(set(region_values)) > 1 or (region_values and region_malformed):
        return ParsedAnswer(
            status="ambiguous",
            value=None,
            raw=" | ".join(region_values + region_malformed),
            marker="####/Answer:",
            answer_region=region,
            thinking_closed=closed,
            final_line=final_line,
        )
    if not candidates:
        return ParsedAnswer(
            status="missing",
            value=None,
            raw=None,
            marker=None,
            answer_region=region,
            thinking_closed=closed,
            final_line=final_line,
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

    # A malformed marker alongside any valid marker is a conflict, not a
    # reason to accept the valid one.
    if malformed and values:
        return ParsedAnswer(
            status="ambiguous",
            value=None,
            raw=" | ".join(raw for _, raw in candidates),
            marker=",".join(sorted(set(markers))),
            answer_region=region,
            thinking_closed=closed,
            final_line=final_line,
        )
    if len(set(values)) > 1:
        return ParsedAnswer(
            status="ambiguous",
            value=None,
            raw=" | ".join(raw for _, raw in candidates),
            marker=",".join(sorted(set(markers))),
            answer_region=region,
            thinking_closed=closed,
            final_line=final_line,
        )
    if not values:
        return ParsedAnswer(
            status="malformed",
            value=None,
            raw=" | ".join(malformed),
            marker=markers[-1],
            answer_region=region,
            thinking_closed=closed,
            final_line=final_line,
        )
    return ParsedAnswer(
        status="ok",
        value=values[-1],
        raw=candidates[-1][1],
        marker=candidates[-1][0],
        answer_region=region,
        thinking_closed=closed,
        final_line=final_line,
    )

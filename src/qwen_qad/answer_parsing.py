"""Frozen strict GSM8K final-answer parser (spec v3).

Qwen3 thinking mode uses ``<think>`` (151667) and ``</think>`` (151668),
verified against the pinned tokenizer. Only the final content line may carry a
final-answer marker; markers elsewhere are intermediate or ambiguous attempts.

Accepted terminal forms (see docs/parser_spec_v3.md):
  - ``#### <number>``, including a nested ``#### $\\boxed{8}$``;
  - one boxed value on the final line, with display-math/markdown decoration,
    provided no numeric token follows it;
  - ``Answer: <number>``, including Markdown ``**Answer: $694**``.

Rejected: contradictory markers, malformed markers mixed with valid ones,
unfinished thinking, and intermediate boxed calculations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
PARSER_VERSION = "3.0.0"

_FRACTION = re.compile(r"-?\d+/\d+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_BOXED_WRAPPED = re.compile(
    r"(?is)^[\s$*_`]*(?:answer\s*:\s*)?[\s$*_`]*\\boxed\{([^{}]*)\}[\s$*_`]*$"
)
_HASH = re.compile(r"####[ \t]*([^\n]*)")
# [^\n] and [ \t] prevent the Answer: regex from consuming a following line.
_ANSWER_LINE = re.compile(
    r"(?im)^[ \t]*(?:\*\*|__)?[ \t]*answer[ \t]*:[ \t]*([^\n]*?)[ \t]*(?:\*\*|__)?[ \t]*$"
)


@dataclass
class ParsedAnswer:
    status: str  # ok | missing | ambiguous | malformed | incomplete_thinking
    value: str | None
    raw: str | None
    marker: str | None
    answer_region: str
    thinking_closed: bool
    final_line: str = ""
    marker_seen: bool = False
    terminal_marker_present: bool = False

    @property
    def explicit(self) -> bool:
        return self.status == "ok"

    @property
    def marker_present(self) -> bool:
        """Any marker anywhere in the answer region (not necessarily terminal)."""
        return self.marker_seen


def split_thinking(text: str) -> tuple[str, str, bool]:
    """Split at the last ``</think>``; returns (reasoning, answer_region, closed)."""
    index = text.rfind(THINK_CLOSE)
    if index < 0:
        return text, "", False
    start = index + len(THINK_CLOSE)
    return text[:index], text[start:], True


def normalize_number(raw: str) -> str | None:
    """Normalize an explicitly marked numeric answer; None if not numeric.

    Accepts surrounding LaTeX/markdown decoration and one nested ``\\boxed{}``
    (``#### $\\boxed{8}$``), but not arithmetic around a box.
    """
    text = raw.strip()
    wrapped = _BOXED_WRAPPED.fullmatch(text)
    if wrapped:
        text = wrapped.group(1)
    text = text.strip().strip("*_`$").strip()
    text = text.strip("()[]{} ").rstrip(".").strip()
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
    # A single leading number followed only by non-numeric units is a
    # formatted answer ("14 bags.", "120 minutes"). Reject extra digits.
    leading = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*[A-Za-z%].*", text)
    if leading and not re.search(r"\d", text[leading.end(1):]):
        return _format_float(float(leading.group(1)))
    return None


def _format_float(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


_LATEX_DELIMITERS = {"$$", "\\[", "\\]", "\\(", "\\)"}


def _last_content_line(region: str) -> str:
    for line in reversed(region.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _LATEX_DELIMITERS:
            continue
        return line
    return ""


def _boxed_candidates(final_line: str) -> list[tuple[str, str]]:
    """One-line boxed values; reject arithmetic/numbers after the last box."""
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
        raw = answer.group(1).strip()
        # Nested markers (Answer: #### 8, Answer: $\boxed{8}$) are one answer;
        # the hash/boxed candidates above already represent them.
        if not (_HASH.search(raw) or _BOXED.search(raw)):
            candidates.append(("Answer:", raw))
    return candidates


def _region_conflicts(region: str) -> tuple[list[str], list[str]]:
    """All ####/Answer: attempts in the region, to detect contradictions.

    Boxed values are excluded: only a box on the final line can be terminal, so
    earlier boxed calculations are intermediate.
    """
    values: list[str] = []
    malformed: list[str] = []
    for match in _HASH.finditer(region):
        raw = match.group(1).strip()
        value = normalize_number(raw)
        values.append(value) if value is not None else malformed.append(raw)
    for match in _ANSWER_LINE.finditer(region):
        raw = match.group(1).strip()
        if _HASH.search(raw) or _BOXED.search(raw):
            continue  # nested marker; counted by hash/final boxed candidate
        value = normalize_number(raw)
        values.append(value) if value is not None else malformed.append(raw)
    return values, malformed


def _marker_seen(region: str) -> bool:
    return bool(_HASH.search(region) or _BOXED.search(region) or _ANSWER_LINE.search(region))


def parse_final_answer(text: str, thinking_mode: bool) -> ParsedAnswer:
    """Parse only an explicitly marked terminal answer on the final line."""
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

    final_line = _last_content_line(region)
    candidates = _candidates(final_line)
    terminal = bool(candidates)
    seen = _marker_seen(region)

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
            marker_seen=seen,
            terminal_marker_present=terminal,
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
            marker_seen=seen,
            terminal_marker_present=False,
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

    if malformed and values:
        return ParsedAnswer(
            status="ambiguous",
            value=None,
            raw=" | ".join(raw for _, raw in candidates),
            marker=",".join(sorted(set(markers))),
            answer_region=region,
            thinking_closed=closed,
            final_line=final_line,
            marker_seen=seen,
            terminal_marker_present=True,
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
            marker_seen=seen,
            terminal_marker_present=True,
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
            marker_seen=seen,
            terminal_marker_present=True,
        )
    return ParsedAnswer(
        status="ok",
        value=values[-1],
        raw=candidates[-1][1],
        marker=candidates[-1][0],
        answer_region=region,
        thinking_closed=closed,
        final_line=final_line,
        marker_seen=seen,
        terminal_marker_present=True,
    )

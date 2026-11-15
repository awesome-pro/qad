# GSM8K final-answer parser specification v3.0.0 (frozen)

Status: **frozen** after the seed-42 regrade validation. Any semantic change
requires a version bump and a new regrade; do not silently change v3.

## Input

A decoded generation (special tokens removed for parsing) and a thinking-mode
flag. In thinking mode, only text after the last `</think>` is the answer
region; without a closing token the result is `incomplete_thinking` and no
answer is graded.

## Terminal answer line

Take the last non-empty line of the answer region, skipping lines that contain
only LaTeX display delimiters (`$$`, `\[`, `\]`, `\(`, `\)`). Only this line may
carry a terminal marker.

## Accepted terminal forms

1. `#### <number>` on the terminal line. The number may be decorated with
   markdown/LaTeX/currency, and may be a single nested `\boxed{...}`
   (`#### $\boxed{8}$` → `8`).
2. Exactly one `\boxed{<number>}` on the terminal line, optionally with
   markdown/display-math decoration and non-numeric units after it
   (`$$ \boxed{125} $$`, `\boxed{7} \text{ DVDs}`). A numeric token after the
   last box rejects the line (`\boxed{12} + 3 = 15` is not terminal).
3. A line starting with `Answer:` (optionally Markdown `**Answer:**`), with a
   numeric value possibly `$`-prefixed or bold-wrapped:
   `Answer: 42`, `**Answer: $694**`, `**Answer: $233.00**`. A nested marker
   after the prefix is one answer: `**Answer:** #### 225` → `225`,
   `**Answer:** $\boxed{10}$ games.` → `10`.

`####` matching stops at the newline; the `Answer:` pattern uses `[^\n]` and
`[ \t]` so it cannot consume a following line.

## Rejection rules

- Any two explicit `####`/`Answer:` attempts in the answer region with
  different values, or a malformed attempt alongside a valid one → `ambiguous`.
- Multiple terminal values that differ, or a valid terminal value alongside a
  malformed terminal marker → `ambiguous`.
- Marker exists but no numeric value can be extracted → `malformed`.
- No terminal marker → `missing` (even if an earlier marker exists).
- Intermediate boxed arithmetic → not a terminal answer.
- Unfinished thinking → `incomplete_thinking`.

## Numbers

Optional sign, integer/decimal, commas, optional `$`, fractions `a/b`
(normalized to decimal), surrounding parentheses/periods. A single leading
number followed only by non-numeric units is accepted (`14 bags.` → `14`,
`120 minutes` → `120`); any additional digit rejects the value (`8 and 9`,
`12 + 3`). No implicit last-number fallback.

## Metrics (strictly distinct)

- `marker_present_rate`: any `####`, `\boxed{}`, or `Answer:` appears in the
  answer region, including earlier/intermediate attempts.
- `terminal_marker_present_rate`: a marker appears on the terminal answer line,
  whether it parses or not.
- `parsed_answer_rate`: terminal marker parsed to exactly one value (`ok`).
- `completion_rate`: model emitted EOS before the generation cap.
- `truncation_rate`: generation reached the cap.
- `accuracy`: parsed value equals the gold answer.

## Changelog

- v3.0.0: nested markers (including `#### $\boxed{8}$` and
  `**Answer:** #### 225`) treated as one answer; Markdown
  `**Answer: $694**`; units after one leading number; newline-safe `Answer:`
  regex; explicit marker/terminal-marker metrics.
- v2: final-line-only markers, conflict rejection, no newline consumption.
- v1: historical permissive parser; not valid for strict accuracy.

"""Split noisy webpage records into bounded source units."""

from __future__ import annotations

import re
from dataclasses import dataclass


_TAIL_STOP = re.compile(
    r"(?im)^\s*(?:#{0,4}\s*)?(?:comments?\b|related\s+questions?\b|posted\s+by\b)"
)
_UNIT_STOP = re.compile(r"(?im)^\s*(?:#{0,4}\s*)?problem\s*(?:2|ii)\b")
_QUESTION_HEADER = re.compile(
    r"(?im)^\s*(?:#{0,4}\s*)?(?:problem\s+statement|interview\s+question|"
    r"question(?:\s+\d+)?|problem(?:\s+\d+)?)\s*[:.\-]?\s*"
)
_SOLUTION_HEADER = re.compile(
    r"(?im)^\s*(?:#{0,4}\s*)?(?:solution|(?:correct\s+)?answer)\s*[:.\-]?\s*"
)
_ACTION_QUESTION = re.compile(
    r"(?im)^\s*((?:find|calculate|determine|solve|compute|evaluate|how\s+many|"
    r"what\s+is)\b[^\n]{20,600})"
)


@dataclass(frozen=True)
class SourceUnit:
    text: str
    question: str | None
    solution: str | None
    pattern: str


def clean_text(text: str) -> str:
    """Apply light formatting cleanup without rewriting mathematical content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\[\]\n]{1,180}\.(?:png|jpe?g|gif)\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def truncate_at_stop(text: str) -> str:
    match = _TAIL_STOP.search(text) or _UNIT_STOP.search(text)
    return text[: match.start()].rstrip() if match else text


def _normalize_section(section: str) -> str:
    section = re.sub(r"\s+", " ", section).strip(" #:-\n\t")
    return re.sub(r"(?i)^(?:question|problem(?:\s+\d+)?)\s*[:.\-]\s*", "", section)


def _question_without_header(text: str) -> str | None:
    action = _ACTION_QUESTION.search(text)
    if action:
        return _normalize_section(action.group(1))
    for line in text.splitlines():
        line = _normalize_section(line)
        if 25 <= len(line) <= 600 and line.endswith("?"):
            return line
    return None


def split_source_units(raw_text: str, max_chars: int = 6000) -> list[SourceUnit]:
    """Split explicit question/solution sections and keep each within one document."""
    text = clean_text(raw_text)
    tail = _TAIL_STOP.search(text)
    if tail:
        text = text[: tail.start()].rstrip()
    headers = list(_QUESTION_HEADER.finditer(text))
    units: list[SourceUnit] = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        section = text[header.start() : end][:max_chars]
        local_header = _QUESTION_HEADER.search(section)
        if local_header is None:
            continue
        solution_match = _SOLUTION_HEADER.search(section, local_header.end())
        if solution_match:
            question = _normalize_section(section[local_header.end() : solution_match.start()])
            solution = section[solution_match.start() :].strip()
            units.append(SourceUnit(section, question or None, solution or None, "explicit_question_solution"))
        else:
            question = _normalize_section(section[local_header.end() :])
            units.append(SourceUnit(section, question or None, None, "unpaired_question"))
    if units:
        return units
    text = text[:max_chars]
    return [SourceUnit(text, _question_without_header(text), None, "unpaired_text")]


def resolve_chunk(raw_text: str, max_chars: int = 6000) -> SourceUnit:
    """Return the first source unit for small callers and interactive inspection."""
    return split_source_units(raw_text, max_chars)[0]

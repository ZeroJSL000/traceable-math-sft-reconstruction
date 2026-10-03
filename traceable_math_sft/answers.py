"""Conservative final-answer parsing and comparison."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


_ANSWER_LABEL = re.compile(r"(?im)^\s*(?:correct\s+)?answer\s*[:=]\s*(.+?)\s*$")
_HASH_ANSWER = re.compile(r"(?im)^\s*####\s*(.+?)\s*$")
_ANSWER_IS = re.compile(r"(?i)\banswer\s+is\s+([^;\n]{1,80})")
_THEREFORE = re.compile(r"(?im)^\s*(?:therefore|thus|hence)[^\n]{0,100}?=\s*([^;\n]{1,80})")
_CONCLUSION = re.compile(
    r"(?im)^\s*(?:therefore|thus|hence|so)\b[^\n]{0,180}?\b(?:is|equals)\s+"
    r"(?:approximately\s+)?([+-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?(?:\s+[A-Za-z]+){0,2})"
    r"\s*[.!]?\s*$"
)
_CHOICE = re.compile(r"^[A-D](?:\s*[.)])?$", re.IGNORECASE)
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?\d+(?:\.\d+)?)?$")
_PERCENT = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*%$")
_UNIT = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?\d+(?:\.\d+)?)?\s+[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,2}$")
_SYMBOLIC = re.compile(r"^[A-Za-z](?:\s*[+\-*/^]\s*[A-Za-z0-9]+)*\s*=\s*[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+\-]?\d+(?:\.\d+)?)?$")
_LATEX_MATH = re.compile(r"\\(?:frac|dfrac|tfrac|sqrt|pi|infty|pm|cdot|times|theta|alpha|beta)\b")


def boxed_values(text: str) -> list[str]:
    """Return values from balanced LaTeX box commands, including nested braces."""
    values: list[str] = []
    marker = r"\boxed"
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        opening = start + len(marker)
        while opening < len(text) and text[opening].isspace():
            opening += 1
        if opening >= len(text) or text[opening] != "{":
            cursor = opening
            continue
        depth = 1
        end = opening + 1
        while end < len(text) and depth:
            if text[end] == "{" and (end == 0 or text[end - 1] != "\\"):
                depth += 1
            elif text[end] == "}" and (end == 0 or text[end - 1] != "\\"):
                depth -= 1
            end += 1
        if depth == 0:
            values.append(text[opening + 1 : end - 1].strip())
        cursor = max(end, opening + 1)
    return values


def answer_type(value: str) -> str | None:
    """Classify supported single-answer forms using reconstruction rules."""
    value = value.strip().strip("$ ")
    if not value or len(value) > 80 or "\n" in value:
        return None
    if re.search(r"(?i)\b(?:question|solution|explanation|problem|click|subscribe)\b", value):
        return None
    if len(value.split()) > 8:
        return None
    if re.search(r"(?i)\b(?:and|or)\b", value) and not re.search(r"\\(?:frac|cdot)", value):
        return None
    if _CHOICE.fullmatch(value):
        return "choice"
    compact = re.sub(r"[\s$,]", "", value)
    if _NUMBER.fullmatch(compact):
        return "fraction" if "/" in compact else "numeric"
    if _PERCENT.fullmatch(value):
        return "percent"
    if _UNIT.fullmatch(value):
        return "with_unit"
    if _SYMBOLIC.fullmatch(value):
        return "symbolic_equation"
    if _LATEX_MATH.search(value) and value.count("{") == value.count("}"):
        return "latex_expression"
    return None


def is_plausible_answer(value: str) -> bool:
    return answer_type(value) is not None


def extract_teacher_answer(solution: str) -> tuple[str | None, str]:
    """Find a final answer already present in the source solution."""
    boxes = [value for value in boxed_values(solution) if is_plausible_answer(value)]
    if boxes:
        return boxes[-1], "boxed"
    for pattern, tag in (
        (_ANSWER_LABEL, "answer_label"),
        (_HASH_ANSWER, "hash_answer"),
        (_ANSWER_IS, "answer_is"),
        (_THEREFORE, "therefore_equals"),
        (_CONCLUSION, "conclusion_numeric"),
    ):
        matches = [match.group(1).strip() for match in pattern.finditer(solution)]
        for value in reversed(matches):
            value = value.rstrip(" .,:;)").strip()
            if is_plausible_answer(value):
                return value, tag
    return None, "no_supported_answer"


def _numeric_value(value: str) -> Fraction | None:
    compact = re.sub(r"[\s$,]", "", value)
    if not _NUMBER.fullmatch(compact):
        return None
    try:
        if "/" in compact:
            numerator, denominator = compact.split("/", 1)
            if Decimal(denominator) == 0:
                return None
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(compact))
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def answers_agree(first: str, second: str) -> bool:
    """Compare exact normalized text or ordinary decimal and fraction values."""
    if not (is_plausible_answer(first) and is_plausible_answer(second)):
        return False
    first_number = _numeric_value(first)
    second_number = _numeric_value(second)
    if first_number is not None and second_number is not None:
        return first_number == second_number
    normalize = lambda value: re.sub(r"\s+", "", value.strip().lower().strip("$"))
    return normalize(first) == normalize(second)


def is_choice(value: str) -> bool:
    return bool(_CHOICE.fullmatch(value.strip()))

"""Extract a grounded solution from a source webpage."""

from __future__ import annotations

from dataclasses import dataclass

from .answers import extract_teacher_answer
from .chunking import SourceUnit
from .routing import has_weak_alignment, solution_rejection


@dataclass(frozen=True)
class TeacherResult:
    solution: str | None
    answer: str | None
    reason: str


def extract_teacher(unit: SourceUnit) -> TeacherResult:
    """Prefer source-provided reasoning and an answer from that same source unit."""
    if not unit.question or not unit.solution:
        return TeacherResult(None, None, "teacher_source_pair_missing")
    answer, answer_tag = extract_teacher_answer(unit.solution)
    if answer is None:
        return TeacherResult(None, None, answer_tag)
    rejection = solution_rejection(unit.solution, unit.text, model_generated=False)
    if rejection:
        return TeacherResult(None, None, f"teacher_{rejection}")
    if not has_weak_alignment(unit.question, unit.solution):
        return TeacherResult(None, None, "teacher_alignment_missing")
    reasoning = unit.solution.strip()
    if not reasoning.rstrip().endswith(r"\boxed{" + answer + "}"):
        reasoning += "\n\nFinal answer: " + r"\boxed{" + answer + "}"
    return TeacherResult(reasoning, answer, f"teacher_ok:{answer_tag}")

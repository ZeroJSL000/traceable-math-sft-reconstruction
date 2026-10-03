"""Route source units and apply conservative quality gates."""

from __future__ import annotations

import re

from .chunking import SourceUnit


_CONCEPT = re.compile(r"(?i)\b(?:definition|theorem|formula|concept|properties of)\b")
_PROOF = re.compile(r"(?i)\b(?:prove that|proof of|show that .* holds for all)\b")
_GRAPH = re.compile(r"(?i)\b(?:see (?:the )?(?:figure|diagram|graph)|shown below|image above)\b")
_MULTI_QUESTION = re.compile(r"(?i)\b(?:part\s*[a-d]|find both|respectively|questions?\s*1\s*(?:and|,)\s*2)\b")
_POLLUTION = re.compile(
    r"(?i)(?:actual session|instruction\s*:|do not produce any tool call|�|⚼|⚙)"
)
_TEMPLATE_HALLUCINATIONS = (
    "3x+5=20",
    "subtract 5 from both sides",
    "divide both sides by 3",
)
_STOP_WORDS = {
    "about", "after", "answer", "because", "calculate", "find", "from", "given",
    "have", "many", "more", "number", "problem", "result", "solve", "solution",
    "that", "them", "then", "there", "therefore", "this", "what", "when", "where",
    "which", "with", "would", "your",
}


def classify_route(unit: SourceUnit) -> str:
    """Classify into report-described source categories."""
    if len(unit.text) < 40 or _POLLUTION.search(unit.text) or _GRAPH.search(unit.text):
        return "reject_low_value"
    if unit.solution and unit.question:
        return "route_a_problem_solution"
    if _CONCEPT.search(unit.text):
        return "route_b_concept_doc"
    if unit.question:
        return "route_c_seed_expandable"
    return "reject_low_value"


def question_rejection(question: str) -> str | None:
    """Return a reason when the question is unsuitable for single-answer SFT."""
    if not 25 <= len(question) <= 700:
        return "question_length"
    if _MULTI_QUESTION.search(question) or question.count("?") > 1:
        return "multi_question_answer"
    if _PROOF.search(question):
        return "proof_like"
    if _GRAPH.search(question):
        return "graph_like"
    if _POLLUTION.search(question):
        return "pollution"
    return None


def solution_rejection(solution: str, source_unit: str, *, model_generated: bool) -> str | None:
    if not 30 <= len(solution) <= 2000:
        return "solution_length"
    if _PROOF.search(solution):
        return "proof_like"
    if _GRAPH.search(solution):
        return "graph_like"
    if _POLLUTION.search(solution):
        return "pollution"
    if model_generated:
        compact_source = re.sub(r"\s+", "", source_unit).lower()
        compact_solution = re.sub(r"\s+", "", solution).lower()
        for phrase in _TEMPLATE_HALLUCINATIONS:
            compact_phrase = re.sub(r"\s+", "", phrase).lower()
            if compact_phrase in compact_solution and compact_phrase not in compact_source:
                return "template_hallucination"
    return None


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z]{4,}|\d+(?:\.\d+)?", text)
        if token.lower() not in _STOP_WORDS
    }


def has_weak_alignment(question: str, solution: str) -> bool:
    """Require at least one distinctive term or number shared by question and source solution."""
    question_tokens = _content_tokens(question)
    solution_tokens = _content_tokens(solution)
    return bool(question_tokens & solution_tokens)


def normalized_question(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def quality_score(route: str, source_tag: str, question: str, solution: str) -> float:
    """Compute a transparent reconstruction score; historical weights are unknown."""
    score = 4.0 if source_tag.startswith("teacher:") else 2.0
    if route == "route_a_problem_solution":
        score += 1.0
    if r"\boxed{" in solution:
        score += 1.0
    if 60 <= len(question) <= 350:
        score += 0.5
    if 100 <= len(solution) <= 1200:
        score += 0.5
    return score

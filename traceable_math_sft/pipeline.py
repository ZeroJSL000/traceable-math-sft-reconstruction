"""Stream input records through an auditable SFT construction pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .answers import answer_type, answers_agree, boxed_values, is_plausible_answer
from .chunking import split_source_units
from .model_api import Sampler, build_prompt, prompt_hash
from .routing import (
    classify_route,
    normalized_question,
    quality_score,
    question_rejection,
    solution_rejection,
)
from .teacher import extract_teacher


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_dir: Path
    enable_dual: bool = False
    max_records: int | None = None
    max_model_pairs: int | None = None
    kept_cap: int | None = None
    max_per_document: int = 2
    route_b_max_share: float = 0.10
    temperature_a: float = 0.05
    temperature_b: float = 0.15

    def validate(self) -> None:
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive")
        if self.max_model_pairs is not None and self.max_model_pairs <= 0:
            raise ValueError("max_model_pairs must be positive")
        if self.kept_cap is not None and self.kept_cap <= 0:
            raise ValueError("kept_cap must be positive")
        if self.max_per_document <= 0:
            raise ValueError("max_per_document must be positive")
        if not 0 < self.route_b_max_share < 1:
            raise ValueError("route_b_max_share must be between zero and one")


def _write_jsonl(stream: Any, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reconstruction run report",
        "",
        "This run uses report-guided reconstruction code, not the recovered course source.",
        "Historical counts of 611 and 1,146 are not expected or guaranteed.",
        "",
        f"- Input records read: {report['input_records']}",
        f"- Kept examples: {report['kept']}",
        f"- Model pairs attempted: {report['model_pairs_attempted']}",
        "",
        "## Kept source",
        "",
    ]
    lines += [f"- {key}: {count}" for key, count in sorted(report["source_counts"].items())]
    lines += ["", "## Routed source units", ""]
    lines += [f"- {key}: {count}" for key, count in sorted(report["route_counts"].items())]
    lines += ["", "## Rejection reasons", ""]
    lines += [f"- {key}: {count}" for key, count in sorted(report["reject_counts"].items())]
    return "\n".join(lines) + "\n"


def run_pipeline(config: PipelineConfig, sampler: Sampler | None = None) -> dict[str, Any]:
    """Read one JSONL record at a time and write traceable decisions promptly."""
    config.validate()
    if config.enable_dual and sampler is None:
        raise ValueError("enable_dual requires a sampler")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = config.output_dir / "sft.jsonl"
    audit_path = config.output_dir / "audit.jsonl"
    rejects_path = config.output_dir / "rejects.jsonl"
    route_counts: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    document_kept: defaultdict[str, int] = defaultdict(int)
    seen_questions: set[str] = set()
    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    route_b_kept = 0
    model_pairs_attempted = 0
    input_records = 0

    with (
        config.input_path.open("r", encoding="utf-8") as source,
        sft_path.open("w", encoding="utf-8") as sft_stream,
        audit_path.open("w", encoding="utf-8") as audit_stream,
        rejects_path.open("w", encoding="utf-8") as reject_stream,
    ):
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if config.max_records is not None and input_records >= config.max_records:
                break
            input_records += 1
            record_id = f"line:{line_no}"
            route = "unclassified"
            source_unit = ""
            unit_index = 0

            def reject(reason: str, detail: str | None = None) -> None:
                reject_counts[reason] += 1
                row: dict[str, Any] = {
                    "id": record_id,
                    "line_no": line_no,
                    "unit_index": unit_index,
                    "route": route,
                    "reason": reason,
                    "source_excerpt": source_unit[:500],
                }
                if detail:
                    row["detail"] = detail
                _write_jsonl(reject_stream, row)

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                reject("invalid_json")
                continue
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                reject("invalid_schema")
                continue
            record_id = str(record.get("id") or record_id)
            for unit_index, unit in enumerate(split_source_units(record["text"]), start=1):
                source_unit = unit.text
                route = classify_route(unit)
                route_counts[route] += 1
                if route == "reject_low_value":
                    reject("route_skip:reject_low_value")
                    continue
                if route == "route_c_seed_expandable":
                    reject("route_skip:route_c_seed_expandable")
                    continue
                if not unit.question:
                    reject("question_not_found")
                    continue
                reason = question_rejection(unit.question)
                if reason:
                    reject(reason)
                    continue
                if document_kept[record_id] >= config.max_per_document:
                    reject("per_document_cap")
                    continue
                question_key = normalized_question(unit.question)
                if question_key in seen_questions:
                    reject("dedup_question")
                    continue
                if route == "route_b_concept_doc" and (
                    route_b_kept + 1 > config.route_b_max_share * (len(accepted) + 1)
                ):
                    reject("route_b_over_share")
                    continue

                teacher = extract_teacher(unit)
                answer: str | None = teacher.answer
                solution: str | None = teacher.solution
                source_tag = "teacher:" + teacher.reason if answer else ""
                if answer is None:
                    if not config.enable_dual:
                        reject(teacher.reason)
                        continue
                    if config.max_model_pairs is not None and model_pairs_attempted >= config.max_model_pairs:
                        reject("model_pair_budget")
                        continue
                    prompt = build_prompt(unit.question, unit.text)
                    model_pairs_attempted += 1
                    try:
                        first = sampler.sample(prompt, config.temperature_a)  # type: ignore[union-attr]
                        second = sampler.sample(prompt, config.temperature_b)  # type: ignore[union-attr]
                    except Exception as exc:
                        reject("request_error", type(exc).__name__)
                        continue
                    first_boxes = boxed_values(first)
                    second_boxes = boxed_values(second)
                    if not first_boxes or not second_boxes:
                        reject("boxed_missing_dual")
                        continue
                    if not answers_agree(first_boxes[-1], second_boxes[-1]):
                        reject("dual_boxed_mismatch")
                        continue
                    reason = solution_rejection(first, unit.text, model_generated=True)
                    if reason:
                        reject("model_" + reason)
                        continue
                    reason = solution_rejection(second, unit.text, model_generated=True)
                    if reason:
                        reject("model_" + reason)
                        continue
                    answer = first_boxes[-1]
                    solution = first.strip()
                    if not solution.endswith(r"\boxed{" + answer + "}"):
                        solution += "\n\nFinal answer: " + r"\boxed{" + answer + "}"
                    source_tag = "model_dual_consistent"
                if answer is None or solution is None or not is_plausible_answer(answer):
                    reject("answer_type_unknown")
                    continue
                messages = [
                    {"role": "system", "content": "Solve the math problem and end with \\boxed{final answer}."},
                    {"role": "user", "content": unit.question},
                    {"role": "assistant", "content": solution},
                ]
                sft_row = {"messages": messages}
                built_prompt = build_prompt(unit.question, unit.text)
                audit_row = {
                    "id": record_id,
                    "line_no": line_no,
                    "unit_index": unit_index,
                    "route": route,
                    "extraction_pattern": unit.pattern,
                    "source_tag": source_tag,
                    "question": unit.question,
                    "answer": answer,
                    "answer_type": answer_type(answer),
                    "quality_score": quality_score(route, source_tag, unit.question, solution),
                    "source_hash": hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
                    "prompt_hash": prompt_hash(built_prompt),
                    "source_unit": unit.text,
                }
                _write_jsonl(sft_stream, sft_row)
                _write_jsonl(audit_stream, audit_row)
                accepted.append((sft_row, audit_row))
                seen_questions.add(question_key)
                document_kept[record_id] += 1
                if route == "route_b_concept_doc":
                    route_b_kept += 1
                source_counts[source_tag.split(":", 1)[0]] += 1

    accepted.sort(
        key=lambda pair: (-pair[1]["quality_score"], pair[1]["line_no"], pair[1]["unit_index"])
    )
    if config.kept_cap is not None and len(accepted) > config.kept_cap:
        removed = accepted[config.kept_cap :]
        accepted = accepted[: config.kept_cap]
        with rejects_path.open("a", encoding="utf-8") as reject_stream:
            for _, audit in removed:
                reject_counts["kept_cap"] += 1
                _write_jsonl(
                    reject_stream,
                    {
                        "id": audit["id"],
                        "line_no": audit["line_no"],
                        "unit_index": audit["unit_index"],
                        "route": audit["route"],
                        "reason": "kept_cap",
                        "source_excerpt": audit["source_unit"][:500],
                    },
                )
        source_counts = Counter(audit["source_tag"].split(":", 1)[0] for _, audit in accepted)
    for path, index in ((sft_path, 0), (audit_path, 1)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for pair in accepted:
                _write_jsonl(stream, pair[index])
        os.replace(temporary, path)

    report: dict[str, Any] = {
        "reconstruction_version": __version__,
        "historical_run_reproduced": False,
        "input_records": input_records,
        "kept": len(accepted),
        "model_pairs_attempted": model_pairs_attempted,
        "route_counts": dict(route_counts),
        "source_counts": dict(source_counts),
        "reject_counts": dict(reject_counts),
        "settings": {
            "enable_dual": config.enable_dual,
            "max_records": config.max_records,
            "max_model_pairs": config.max_model_pairs,
            "kept_cap": config.kept_cap,
            "max_per_document": config.max_per_document,
            "route_b_max_share": config.route_b_max_share,
            "temperature_a": config.temperature_a,
            "temperature_b": config.temperature_b,
        },
    }
    (config.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "report.md").write_text(_report_markdown(report), encoding="utf-8")
    return report

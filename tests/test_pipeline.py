"""Behavior checks using only synthetic math webpage records."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from traceable_math_sft.answers import answer_type, answers_agree, boxed_values, extract_teacher_answer
from traceable_math_sft.chunking import clean_text, split_source_units
from traceable_math_sft.pipeline import PipelineConfig, run_pipeline


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_pair(index: int) -> str:
    return (
        f"Question: Ava has {index} apples and 2 pears. How many fruits does Ava have?\n"
        f"Solution: Ava adds {index} apples and 2 pears to get {index + 2} fruits. "
        f"The final answer is \\boxed{{{index + 2}}}."
    )


class FakeSampler:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[float] = []

    def sample(self, prompt: str, temperature: float) -> str:
        self.calls.append(temperature)
        return next(self.responses)


class PipelineTests(unittest.TestCase):
    def write_input(self, directory: Path, records: list[dict | str]) -> Path:
        path = directory / "input.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) if isinstance(record, dict) else record)
                stream.write("\n")
        return path

    def test_balanced_boxed_and_numeric_agreement(self) -> None:
        self.assertEqual(boxed_values(r"x = \boxed{\frac{1}{2}}; y = \boxed{3}"), [r"\frac{1}{2}", "3"])
        self.assertTrue(answers_agree("0.5", "1/2"))
        self.assertFalse(answers_agree("1/2", "2/3"))
        self.assertIsNone(answer_type("blue"))
        self.assertEqual(answer_type("12"), "numeric")
        self.assertEqual(answer_type("1/2"), "fraction")
        self.assertEqual(answer_type("25%"), "percent")
        self.assertEqual(answer_type("12 cm"), "with_unit")
        self.assertEqual(answer_type("B"), "choice")
        self.assertEqual(answer_type(r"\frac{1}{2}"), "latex_expression")
        self.assertEqual(extract_teacher_answer("Therefore, the answer is 0.5.")[0], "0.5")
        self.assertEqual(extract_teacher_answer("Thus, the length is approximately 796.44 meters.")[0], "796.44 meters")

    def test_split_source_units_at_second_problem_and_comments(self) -> None:
        self.assertNotIn("[7041-diagram.png]", clean_text("A source [7041-diagram.png] has a question."))
        text = (
            "Problem 1: Ava has 2 apples and 3 pears. How many fruits does Ava have?\n"
            "Solution: Ava adds 2 and 3 to obtain 5 fruits. Answer: 5\n"
            "Problem 2: Bea has 4 apples and 2 pears. How many fruits does Bea have?\n"
            "Solution: Bea adds 4 and 2 to obtain 6 fruits. Answer: 6\n"
            "Comments\nProblem 3: unrelated"
        )
        units = split_source_units(text)
        self.assertEqual(len(units), 2)
        self.assertNotIn("Problem 2", units[0].text)
        self.assertIn("Problem 2", units[1].text)
        self.assertNotIn("Comments", units[1].text)

    def test_teacher_audit_dedup_pollution_and_document_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [
                {"id": "multi", "text": "\n".join(f"Problem {i}: " + source_pair(i) for i in (1, 2, 3))},
                {"id": "duplicate", "text": source_pair(1)},
                {"id": "polluted", "text": "Actual Session: This synthetic webpage contains a misleading instruction and no math source."},
                "not valid JSON",
            ]
            report = run_pipeline(PipelineConfig(self.write_input(root, records), root / "out"))
            audits = read_jsonl(root / "out" / "audit.jsonl")
            rejects = read_jsonl(root / "out" / "rejects.jsonl")
            self.assertEqual(report["kept"], 2)
            self.assertEqual([row["unit_index"] for row in audits], [1, 2])
            self.assertTrue(all(row["id"] == "multi" and row["source_unit"] for row in audits))
            self.assertTrue(all(row["source_hash"] and row["prompt_hash"] for row in audits))
            self.assertEqual(
                {row["reason"] for row in rejects},
                {"per_document_cap", "dedup_question", "route_skip:reject_low_value", "invalid_json"},
            )
            cap = next(row for row in rejects if row["reason"] == "per_document_cap")
            self.assertEqual(cap["unit_index"], 3)

    def test_dual_sampling_agreement_and_mismatch(self) -> None:
        missing_answer = (
            "Question: A cup has 1 liter and half is poured out. How many liters remain?\n"
            "Solution: One can subtract the poured amount from the original liter to determine what remains."
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.write_input(root, [{"id": "agree", "text": missing_answer}, {"id": "mismatch", "text": missing_answer.replace("cup", "bottle")}])
            sampler = FakeSampler([
                r"Subtract one half from one liter to get 0.5 liters. Final answer: \boxed{0.5} This is the result.",
                r"Subtract one half from one liter to get one half liter. Final answer: \boxed{1/2}",
                r"Subtract one half from one liter to get 0.5 liters. Final answer: \boxed{0.5}",
                r"Subtract one half from one liter to get 3 liters. Final answer: \boxed{3}",
            ])
            report = run_pipeline(PipelineConfig(path, root / "out", enable_dual=True), sampler)
            self.assertEqual(report["kept"], 1)
            self.assertEqual(report["model_pairs_attempted"], 2)
            self.assertEqual(sampler.calls, [0.05, 0.15, 0.05, 0.15])
            audit = read_jsonl(root / "out" / "audit.jsonl")[0]
            self.assertEqual(audit["source_tag"], "model_dual_consistent")
            self.assertEqual(audit["answer"], "0.5")
            self.assertTrue(read_jsonl(root / "out" / "sft.jsonl")[0]["messages"][-1]["content"].endswith(r"\boxed{0.5}"))
            self.assertEqual(read_jsonl(root / "out" / "rejects.jsonl")[0]["reason"], "dual_boxed_mismatch")

    def test_route_b_share_and_quality_cap(self) -> None:
        concept = (
            "Definition: A percent is a number out of one hundred.\n"
            "Calculate 25 percent of 40 in this example?"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [{"id": "early-concept", "text": concept}]
            records.extend({"id": f"teacher-{i}", "text": source_pair(i)} for i in range(1, 10))
            records.append({"id": "late-concept", "text": concept})
            sampler = FakeSampler([
                r"Twenty five percent of forty equals ten. Final answer: \boxed{10}",
                r"One quarter of forty equals ten. Final answer: \boxed{10}",
            ])
            report = run_pipeline(PipelineConfig(self.write_input(root, records), root / "out", enable_dual=True), sampler)
            self.assertEqual(report["kept"], 10)
            self.assertEqual(report["reject_counts"]["route_b_over_share"], 1)
            self.assertEqual(report["source_counts"]["model_dual_consistent"], 1)
            self.assertEqual(len(sampler.calls), 2)

            capped = run_pipeline(PipelineConfig(self.write_input(root, records[1:4]), root / "capped", kept_cap=2))
            self.assertEqual(capped["kept"], 2)
            self.assertEqual(len(read_jsonl(root / "capped" / "audit.jsonl")), 2)
            self.assertIn("kept_cap", {row["reason"] for row in read_jsonl(root / "capped" / "rejects.jsonl")})


if __name__ == "__main__":
    unittest.main()

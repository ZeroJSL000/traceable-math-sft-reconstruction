# Traceable Math SFT: report-guided reconstruction

This repository is a **new implementation reconstructed from a course project report and an archived pipeline README**. The original server-side Python source was not recovered. The code here is not an archived copy of the original project and should not be used to claim exact reproduction of its historical experiments.

The original personal work, conducted within a group course project, centered on turning a noisy 100,000-record Nemotron math webpage subset into auditable SFT candidates. The archived evidence describes a teacher-first pipeline with source-unit chunking, route decisions, answer extraction, optional Qwen3-0.6B dual sampling, quality gates, and audit outputs. It records a 611-example teacher-path run and a separate 1,146-example teacher-plus-dual run. **Those counts belong to the historical runs; this reconstruction does not promise either count or their sample identities.**

## Scope and provenance

This repository implements only the data-construction core. It does not include the original training scripts, LoRA weights, course data, validation set, model outputs, course notebook, team materials, or private server files. It does not claim that the reconstruction trained or improved a model.

The implementation follows the reported processing order:

1. Read `id`, `text`, and optional `metadata` from JSONL one record at a time.
2. Apply light cleanup, stop at webpage tails such as `Comments`, `Related Questions`, and `Posted by`, and treat `Problem 2` as the boundary between source units when it has its own question/solution pair.
3. Split multiple explicit question and solution pairs within a document into separate source units.
4. Route units as problem/solution pages, concept documents, weak seeds, or low-value material.
5. Extract a source-provided solution and final answer first, including balanced nested `\boxed{...}`.
6. If explicitly enabled, request two local model samples at temperatures 0.05 and 0.15 and require final-answer agreement.
7. Apply answer, contamination, hallucination, length, document-cap, and question-deduplication checks.
8. Write SFT, audit, rejection, and summary files.

The report did not preserve the exact historical regular expressions, numeric thresholds, answer-type taxonomy, quality-score weights, or API request payload. The choices in this repository are **new, inspectable design values**, not recovered facts. The reconstruction accepts numeric values, fractions, percentages, numeric values with units, single multiple-choice letters, and a limited set of mathematical expressions; ordinary words are rejected as final answers. The route-B cap is 10% and the per-document cap is two because these values are explicit in the personal report. The deterministic prompt hash is new reconstruction instrumentation; the evidence does not establish that this exact hash existed in the original code. Weak seed route C is recorded and skipped; no unverified seed-expansion method is invented.

## Run with synthetic data

Python 3.10 or newer is sufficient; no package installation is needed from the repository root:

```bash
python -m traceable_math_sft --input examples/synthetic_webpages.jsonl --output-dir outputs/demo
python -m unittest discover -s tests -v
```

The default mode performs only source-based teacher extraction. It makes **no model or network requests**. To try the model fallback, run an OpenAI-compatible server hosting the permitted model locally and opt in:

```bash
python -m traceable_math_sft \
  --input /path/to/approved_math_webpages.jsonl \
  --output-dir outputs/dual \
  --enable-dual \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen3-0.6B-Base \
  --max-model-pairs 100
```

For the original course setting, the source JSONL must come from the course-approved subset. The public repository contains only synthetic records. The optional client is designed for a local OpenAI-compatible endpoint; this repository does not download or bundle a model.

## Outputs

| File | Contents |
| --- | --- |
| `sft.jsonl` | Three-message SFT examples with source-derived or model-generated reasoning and a boxed final answer. |
| `audit.jsonl` | Kept example ID, input line, source-unit index, route, answer, source tag, quality score, source unit, source hash, and deterministic prompt hash. |
| `rejects.jsonl` | Rejected ID, input line, source-unit index, route, reason, and a short source excerpt. |
| `report.json`, `report.md` | Run counts and settings, explicitly marked as reconstruction results. |

Input, accepted examples, and rejected examples are processed and written as records arrive. At the end, accepted examples are ranked by the reconstruction quality score and the final SFT/audit files are rewritten atomically. If `--kept-cap` is set, lower-ranked examples beyond the cap are logged with `kept_cap` as their rejection reason.

The model path uses answer agreement as a consistency filter. It is **not proof of answer correctness**. The teacher path also uses only a weak question/solution alignment check, so retained examples require manual review before serious training. The pipeline does not evaluate model accuracy.

## Repository layout

- `traceable_math_sft/chunking.py`: cleanup, stop boundaries, and source-unit resolution.
- `traceable_math_sft/answers.py`: balanced boxed parsing and conservative answer comparison.
- `traceable_math_sft/routing.py`: routes, quality gates, deduplication key, and scoring.
- `traceable_math_sft/teacher.py`: grounded teacher extraction.
- `traceable_math_sft/model_api.py`: optional local OpenAI-compatible model client.
- `traceable_math_sft/pipeline.py`: streaming decisions, provenance, and reports.
- `tests/`: synthetic behavior tests.

## Evidence boundary

The archived report recorded a 407/1,000 base-model validation result and a separate GSM8K diagnostic run at 301/1,000. The 611-example LoRA run's 16-example generation probe deteriorated during training. None of these are produced by this repository, and they should not be presented as outcomes of this implementation.

"""Command-line entry point for the reconstructed pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .model_api import OpenAICompatibleSampler
from .pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable math SFT data from JSONL webpages.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with id, text, and metadata fields")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enable-dual", action="store_true", help="Use two local model samples after teacher extraction fails")
    parser.add_argument("--endpoint", default=os.getenv("QWEN_API_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL_NAME", "Qwen3-0.6B-Base"))
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-model-pairs", type=int)
    parser.add_argument("--kept-cap", type=int)
    parser.add_argument("--route-b-max-share", type=float, default=0.10)
    parser.add_argument("--temperature-a", type=float, default=0.05)
    parser.add_argument("--temperature-b", type=float, default=0.15)
    args = parser.parse_args()
    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        enable_dual=args.enable_dual,
        max_records=args.max_records,
        max_model_pairs=args.max_model_pairs,
        kept_cap=args.kept_cap,
        route_b_max_share=args.route_b_max_share,
        temperature_a=args.temperature_a,
        temperature_b=args.temperature_b,
    )
    sampler = (
        OpenAICompatibleSampler(args.endpoint, args.model, args.timeout_seconds, args.max_tokens)
        if args.enable_dual
        else None
    )
    report = run_pipeline(config, sampler)
    print(json.dumps({"kept": report["kept"], "input_records": report["input_records"]}))


if __name__ == "__main__":
    main()

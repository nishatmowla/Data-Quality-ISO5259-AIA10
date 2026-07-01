"""
Agentic Data Quality Evaluator
Based on ISO/IEC 5259 series, powered by Mistral AI.

Usage:
    python -m dq_evaluator.main \
        --dataset path/to/data.csv \
        --domain "Healthcare patient records for disease classification" \
        --dataset-name "Patient Records 2024" \
        [--priority accuracy=1,completeness=1,consistency=2,diversity=2,credibility=3,currentness=3] \
        [--metadata '{"source_name": "...", "creation_date": "2023"}'] \
        [--output report.md] \
        [--model-large mistral-large-latest] \
        [--model-small mistral-small-latest]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from mistralai.client import Mistral

from dq_evaluator.tools.data_profiler import load_dataset, profile_dataset, profile_to_text
from dq_evaluator.agents.domain_analyst import analyze_domain
from dq_evaluator.agents.rule_generator import generate_rules
from dq_evaluator.agents.evaluators.accuracy import evaluate_accuracy
from dq_evaluator.agents.evaluators.completeness import evaluate_completeness
from dq_evaluator.agents.evaluators.consistency import evaluate_consistency
from dq_evaluator.agents.evaluators.diversity import evaluate_diversity
from dq_evaluator.agents.evaluators.credibility import evaluate_credibility
from dq_evaluator.agents.evaluators.currentness import evaluate_currentness
from dq_evaluator.agents.report_generator import generate_report, format_report_text
from dq_evaluator.models.iso5259 import QualityCharacteristic


def parse_priorities(s: str) -> dict[QualityCharacteristic, int]:
    result = {}
    for part in s.split(","):
        key, _, val = part.partition("=")
        key = key.strip()
        try:
            result[QualityCharacteristic(key)] = int(val.strip())
        except (ValueError, KeyError):
            print(f"Warning: ignoring unknown characteristic '{key}'")
    return result


def build_context_text(context, subject) -> str:
    return (
        f"Domain: {context.domain}\n"
        f"Purpose: {context.purpose}\n"
        f"Task type: {context.task_type}\n"
        f"Standards: {', '.join(context.standards) or 'N/A'}\n"
        f"Data subjects: {', '.join(subject.entities)}"
    )


def run_evaluation(
    dataset_path: str,
    domain_description: str,
    dataset_name: str = "",
    priority_overrides: dict[QualityCharacteristic, int] | None = None,
    dataset_metadata: dict | None = None,
    output_path: str | None = None,
    model_large: str = "mistral-large-latest",
    model_small: str = "mistral-small-latest",
    currentness_short_threshold: int = 5,
    currentness_long_threshold: int = 15,
    verbose: bool = True,
) -> str:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set.")

    client = Mistral(api_key=api_key)
    dataset_name = dataset_name or Path(dataset_path).stem
    metadata = dataset_metadata or {}

    def log(msg: str):
        if verbose:
            print(f"[dq-eval] {msg}")

    # ── Step 1: Load & profile dataset ──────────────────────────────────────
    log("Loading dataset...")
    df = load_dataset(dataset_path)
    log(f"  {df.shape[0]} rows × {df.shape[1]} columns")

    log("Profiling dataset...")
    profile = profile_dataset(df)
    profile_text = profile_to_text(profile)

    # ── Step 2: Domain analysis ──────────────────────────────────────────────
    log("Agent 1: Analyzing domain context...")
    context, subject, auto_priorities = analyze_domain(
        client, domain_description, profile_text, model=model_large
    )
    log(f"  Domain: {context.domain}")
    log(f"  Task: {context.task_type}")
    log(f"  Standards: {context.standards}")

    # Merge auto-detected priorities with user overrides
    priorities = {**auto_priorities, **(priority_overrides or {})}
    # Fill any missing characteristics with default priority 2
    for c in QualityCharacteristic:
        priorities.setdefault(c, 2)

    log("  Priority order: " + ", ".join(
        f"{k.value}={v}" for k, v in sorted(priorities.items(), key=lambda x: x[1])
    ))

    context_text = build_context_text(context, subject)

    # ── Step 3: Rule generation ──────────────────────────────────────────────
    log("Agent 2: Generating syntactic and semantic rules...")
    syntactic_rules, semantic_rules = generate_rules(
        client, context, profile_text, model=model_large
    )
    log(f"  Generated {len(syntactic_rules)} syntactic rules, {len(semantic_rules)} semantic rules")

    # ── Step 4: Evaluate each characteristic ────────────────────────────────
    results = []

    log("Agent 3: Evaluating Accuracy...")
    results.append(evaluate_accuracy(
        client, df, profile_text, syntactic_rules, semantic_rules,
        context_text, priorities[QualityCharacteristic.ACCURACY], model=model_small
    ))

    log("Agent 4: Evaluating Completeness...")
    results.append(evaluate_completeness(
        client, df, profile, profile_text, context_text,
        priorities[QualityCharacteristic.COMPLETENESS], model=model_small
    ))

    log("Agent 5: Evaluating Consistency...")
    results.append(evaluate_consistency(
        client, df, profile_text, semantic_rules, context_text,
        priorities[QualityCharacteristic.CONSISTENCY], model=model_small
    ))

    log("Agent 6: Evaluating Diversity...")
    results.append(evaluate_diversity(
        client, df, profile, profile_text, context_text,
        priorities[QualityCharacteristic.DIVERSITY], model=model_small
    ))

    log("Agent 7: Evaluating Credibility...")
    results.append(evaluate_credibility(
        client, profile_text, context_text, metadata,
        priorities[QualityCharacteristic.CREDIBILITY], model=model_small
    ))

    log("Agent 8: Evaluating Currentness...")
    results.append(evaluate_currentness(
        client, profile_text, context_text, metadata,
        priorities[QualityCharacteristic.CURRENTNESS],
        short_threshold_years=currentness_short_threshold,
        long_threshold_years=currentness_long_threshold,
        model=model_small
    ))

    # ── Step 5: Generate report ──────────────────────────────────────────────
    log("Agent 9: Generating final report...")
    report = generate_report(
        client, dataset_name, context, subject, results, model=model_large
    )

    report_text = format_report_text(report)

    if output_path:
        Path(output_path).write_text(report_text, encoding="utf-8")
        log(f"Report written to: {output_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Agentic ISO/IEC 5259 Data Quality Evaluator")
    parser.add_argument("--dataset", required=True, help="Path to dataset (CSV, JSON, Parquet, XLSX)")
    parser.add_argument("--domain", required=True, help="Natural language description of the domain and use case")
    parser.add_argument("--dataset-name", default="", help="Human-readable dataset name")
    parser.add_argument("--priority", default="", help="Override characteristic priorities, e.g. accuracy=1,completeness=1")
    parser.add_argument("--metadata", default="{}", help="JSON string with dataset metadata")
    parser.add_argument("--output", default="", help="Output path for Markdown report")
    parser.add_argument("--model-large", default="mistral-large-latest", help="Mistral model for complex reasoning")
    parser.add_argument("--model-small", default="mistral-small-latest", help="Mistral model for evaluations")
    parser.add_argument("--currentness-short", type=int, default=5, help="Short currentness threshold (years)")
    parser.add_argument("--currentness-long", type=int, default=15, help="Long currentness threshold (years)")

    args = parser.parse_args()

    priority_overrides = parse_priorities(args.priority) if args.priority else None
    metadata = json.loads(args.metadata)

    report_text = run_evaluation(
        dataset_path=args.dataset,
        domain_description=args.domain,
        dataset_name=args.dataset_name,
        priority_overrides=priority_overrides,
        dataset_metadata=metadata,
        output_path=args.output or None,
        model_large=args.model_large,
        model_small=args.model_small,
        currentness_short_threshold=args.currentness_short,
        currentness_long_threshold=args.currentness_long,
    )

    print("\n" + "=" * 70)
    print(report_text)


if __name__ == "__main__":
    main()

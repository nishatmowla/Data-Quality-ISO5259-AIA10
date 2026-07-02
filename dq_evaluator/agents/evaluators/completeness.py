"""Completeness evaluator — ISO/IEC 5259 dimensions."""

import json
import pandas as pd
from mistralai.client import Mistral

from dq_evaluator.models.iso5259 import (
    CharacteristicResult, DimensionResult, QualityCharacteristic
)

SYSTEM_PROMPT = """You are a data quality evaluator specializing in the Completeness characteristic of ISO/IEC 5259.
Evaluate the five completeness dimensions based on the dataset statistics provided.
Respond with a JSON object only."""

DIMENSIONS = [
    "value_completeness",
    "value_occurrence_completeness",
    "feature_completeness",
    "record_completeness",
    "label_completeness",
]


def _compute_completeness_stats(df: pd.DataFrame, profile: dict) -> dict:
    stats: dict = {}

    # Value completeness: ratio of non-null values across all cells
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isna().sum().sum()
    stats["value_completeness_score"] = float(1 - null_cells / total_cells) * 100 if total_cells else 100.0
    stats["null_cells"] = int(null_cells)
    stats["total_cells"] = int(total_cells)

    # Feature completeness: columns with zero nulls
    fully_populated = int((df.isna().sum() == 0).sum())
    stats["feature_completeness_score"] = float(fully_populated / df.shape[1]) * 100

    # Record completeness: rows with zero nulls
    complete_rows = int((df.isna().sum(axis=1) == 0).sum())
    stats["record_completeness_score"] = float(complete_rows / len(df)) * 100 if len(df) else 100.0
    stats["complete_rows"] = complete_rows
    stats["total_rows"] = len(df)

    # Label / value occurrence completeness
    if "label_distribution" in profile:
        dist = profile["label_distribution"]
        total = sum(dist.values())
        n_classes = len(dist)
        expected_per_class = total / n_classes if n_classes else total
        min_observed = min(dist.values())
        voc = float(sum(min(v, expected_per_class) for v in dist.values()) / (expected_per_class * n_classes)) * 100
        stats["label_distribution"] = dist
        stats["expected_per_class"] = expected_per_class
        stats["value_occurrence_completeness_score"] = voc
    else:
        stats["value_occurrence_completeness_score"] = None

    # Attack completeness (domain-specific; agent will assess qualitatively)
    stats["columns"] = list(df.columns)
    stats["n_features"] = df.shape[1]
    stats["n_samples"] = len(df)

    return stats


def evaluate_completeness(
    client: Mistral,
    df: pd.DataFrame,
    profile: dict,
    profile_text: str,
    context_text: str,
    priority: int,
    model: str = "mistral-small-latest",
) -> CharacteristicResult:
    stats = _compute_completeness_stats(df, profile)

    user_message = f"""Context:
{context_text}

Computed completeness statistics:
{json.dumps(stats, indent=2)}

Dataset profile:
{profile_text}

Evaluate all five ISO/IEC 5259 completeness dimensions:
- value_completeness: ratio of non-null values (computed: {stats['value_completeness_score']:.1f}%)
- value_occurrence_completeness: are expected data values/classes represented in expected proportions? (computed if available: {stats.get('value_occurrence_completeness_score')})
- feature_completeness: do all expected features contain meaningful (non-null) data? (computed: {stats['feature_completeness_score']:.1f}%)
- record_completeness: are all records complete (no empty fields)? (computed: {stats['record_completeness_score']:.1f}%)
- label_completeness: are all samples correctly and fully labeled?

Also assess domain-specific completeness concerns (e.g., attack type coverage, OSI layer coverage, sample size adequacy).

Respond with JSON:
{{
  "value_completeness": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "value_occurrence_completeness": {{"score": <0-100 or null>, "passed": <bool>, "explanation": "..."}},
  "feature_completeness": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "record_completeness": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "label_completeness": {{"score": <0-100 or null>, "passed": <bool>, "explanation": "..."}},
  "domain_notes": "..."
}}"""

    response = client.chat.complete(
        timeout_ms=120000,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    result = json.loads(response.choices[0].message.content)

    dimensions = []
    scores = []
    for dim in DIMENSIONS:
        d = result.get(dim, {})
        raw_score = d.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        if score is not None:
            scores.append(score)
        dimensions.append(DimensionResult(
            dimension=dim,
            score=score if score is not None else "N/A",
            passed=d.get("passed"),
            explanation=d.get("explanation", ""),
            details={"domain_notes": result.get("domain_notes", "")} if dim == DIMENSIONS[-1] else {},
        ))

    overall = sum(scores) / len(scores) if scores else None
    summary = f"Overall completeness score: {overall:.1f}%" if overall is not None else "Could not compute overall score."

    return CharacteristicResult(
        characteristic=QualityCharacteristic.COMPLETENESS,
        priority=priority,
        overall_score=overall,
        dimensions=dimensions,
        summary=summary,
    )

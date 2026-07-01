"""Consistency evaluator — ISO/IEC 5259 dimensions."""

import json
import pandas as pd
from mistralai.client import Mistral

from dq_evaluator.models.iso5259 import (
    CharacteristicResult, DimensionResult, QualityCharacteristic, SemanticRule
)
from dq_evaluator.tools.rule_checker import check_semantic_rules

SYSTEM_PROMPT = """You are a data quality evaluator specializing in the Consistency characteristic of ISO/IEC 5259.
Evaluate the four consistency dimensions. Respond with JSON only."""

DIMENSIONS = [
    "data_record_consistency",
    "distribution_of_data_values",
    "data_format_consistency",
    "semantic_consistency",
]


def _compute_consistency_stats(df: pd.DataFrame) -> dict:
    stats: dict = {}

    # Data record consistency: duplicate detection
    n_dup = int(df.duplicated().sum())
    stats["duplicate_rows"] = n_dup
    stats["duplicate_rate"] = float(n_dup / len(df)) if len(df) else 0.0
    stats["data_record_consistency_score"] = float(1 - n_dup / len(df)) * 100 if len(df) else 100.0

    # Data format consistency: per-column dtype homogeneity
    dtype_issues = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.dtype == object:
            inferred_types = series.apply(type).value_counts()
            if len(inferred_types) > 1:
                dtype_issues[col] = {str(k.__name__): int(v) for k, v in inferred_types.items()}
    stats["dtype_mixed_columns"] = dtype_issues
    pct_consistent = float(1 - len(dtype_issues) / df.shape[1]) * 100
    stats["data_format_consistency_score"] = pct_consistent

    # Distribution statistics per numeric column
    dist_stats = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col]
        dist_stats[col] = {
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
            "cv": float(s.std() / s.mean()) if s.mean() != 0 else None,
        }
    stats["distributions"] = dist_stats

    return stats


def evaluate_consistency(
    client: Mistral,
    df: pd.DataFrame,
    profile_text: str,
    semantic_rules: list[SemanticRule],
    context_text: str,
    priority: int,
    model: str = "mistral-small-latest",
) -> CharacteristicResult:
    stats = _compute_consistency_stats(df)
    semantic_stubs = check_semantic_rules(df, semantic_rules)

    user_message = f"""Context:
{context_text}

Computed consistency statistics:
{json.dumps(stats, indent=2)}

Semantic rule field data (for semantic consistency assessment):
{json.dumps(semantic_stubs, indent=2)}

Evaluate the four ISO/IEC 5259 consistency dimensions:
- data_record_consistency: duplicate records (computed score: {stats['data_record_consistency_score']:.1f}%, duplicates: {stats['duplicate_rows']})
- distribution_of_data_values: do feature distributions look statistically reasonable and free of anomalous shifts?
- data_format_consistency: consistent data types per column (computed score: {stats['data_format_consistency_score']:.1f}%, issues: {list(stats['dtype_mixed_columns'].keys())})
- semantic_consistency: do values conform to predefined semantic rules and domain constraints?

Respond with JSON:
{{
  "data_record_consistency": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "distribution_of_data_values": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "data_format_consistency": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "semantic_consistency": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}}
}}"""

    response = client.chat.complete(
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
        ))

    overall = sum(scores) / len(scores) if scores else None
    summary = f"Overall consistency score: {overall:.1f}%" if overall is not None else "Could not compute overall score."

    return CharacteristicResult(
        characteristic=QualityCharacteristic.CONSISTENCY,
        priority=priority,
        overall_score=overall,
        dimensions=dimensions,
        summary=summary,
    )

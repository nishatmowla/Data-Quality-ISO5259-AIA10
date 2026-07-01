"""Diversity evaluator — ISO/IEC 5259 dimensions."""

import json
import pandas as pd
from mistralai.client import Mistral

from dq_evaluator.models.iso5259 import (
    CharacteristicResult, DimensionResult, QualityCharacteristic
)

SYSTEM_PROMPT = """You are a data quality evaluator specializing in the Diversity characteristic of ISO/IEC 5259.
Evaluate the three diversity dimensions. Respond with JSON only."""

DIMENSIONS = [
    "label_richness",
    "relative_label_abundance",
    "category_size_diversity",
]


def _compute_diversity_stats(df: pd.DataFrame, profile: dict) -> dict:
    stats: dict = {}

    label_col = profile.get("label_column")
    if label_col and label_col in df.columns:
        dist = profile.get("label_distribution", {})
        n_classes = len(dist)
        total = sum(dist.values())
        stats["label_column"] = label_col
        stats["n_classes"] = n_classes
        stats["label_distribution"] = dist

        if total > 0:
            abundances = {k: v / total for k, v in dist.items()}
            stats["relative_abundances"] = abundances
            min_abundance = min(abundances.values())
            max_abundance = max(abundances.values())
            stats["min_class_abundance"] = min_abundance
            stats["max_class_abundance"] = max_abundance
            stats["imbalance_ratio"] = max_abundance / min_abundance if min_abundance > 0 else float("inf")

            # Category size diversity: % of categories above 50% threshold
            below_50 = sum(1 for v in abundances.values() if v < 0.5)
            stats["categories_below_50pct"] = below_50
            stats["category_size_diversity_score"] = float(below_50 / n_classes) * 100 if n_classes else 0.0
    else:
        # No label column: assess diversity of categorical features instead
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        stats["note"] = "No label column found; assessing categorical feature diversity"
        stats["categorical_columns"] = cat_cols
        diversity_info = {}
        for col in cat_cols[:10]:  # limit
            vc = df[col].value_counts(normalize=True)
            diversity_info[col] = {
                "unique_values": int(df[col].nunique()),
                "top5_pct": vc.head(5).to_dict(),
            }
        stats["feature_diversity"] = diversity_info

    return stats


def evaluate_diversity(
    client: Mistral,
    df: pd.DataFrame,
    profile: dict,
    profile_text: str,
    context_text: str,
    priority: int,
    model: str = "mistral-small-latest",
) -> CharacteristicResult:
    stats = _compute_diversity_stats(df, profile)

    user_message = f"""Context:
{context_text}

Computed diversity statistics:
{json.dumps(stats, indent=2)}

Evaluate the three ISO/IEC 5259 diversity dimensions:
- label_richness: number of distinct classes/labels and whether they sufficiently cover the domain scenarios
- relative_label_abundance: distribution of labels — is any class over- or under-represented?
- category_size_diversity: are category sizes balanced? (% of categories below 50% threshold)

Consider domain context: for ML/anomaly detection, class imbalance is common but must be flagged.

Respond with JSON:
{{
  "label_richness": {{"score": <0-100 or null>, "passed": <bool>, "explanation": "..."}},
  "relative_label_abundance": {{"score": <0-100 or null>, "passed": <bool>, "explanation": "..."}},
  "category_size_diversity": {{"score": <0-100 or null>, "passed": <bool>, "explanation": "..."}}
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
    summary = f"Overall diversity score: {overall:.1f}%" if overall is not None else "Could not compute overall score."

    return CharacteristicResult(
        characteristic=QualityCharacteristic.DIVERSITY,
        priority=priority,
        overall_score=overall,
        dimensions=dimensions,
        summary=summary,
    )

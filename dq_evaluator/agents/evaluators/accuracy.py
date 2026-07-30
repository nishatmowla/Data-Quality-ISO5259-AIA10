"""Accuracy evaluator — ISO/IEC 5259 dimensions."""

import json
from mistralai import Mistral

from dq_evaluator.models.iso5259 import (
    CharacteristicResult, DimensionResult, QualityCharacteristic, SyntacticRule, SemanticRule
)
from dq_evaluator.tools.rule_checker import check_syntactic_rules, check_semantic_rules
import pandas as pd

SYSTEM_PROMPT = """You are a data quality evaluator specializing in the Accuracy characteristic of ISO/IEC 5259.
Evaluate the following six accuracy dimensions based on the rule check results and dataset profile.
Respond with a JSON object only."""

DIMENSIONS = [
    "syntactic_accuracy",
    "semantic_accuracy",
    "data_accuracy_assurance",
    "risk_of_inaccuracy",
    "data_accuracy_range",
    "data_model_accuracy",
]

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        dim: {
            "type": "object",
            "properties": {
                "score": {"description": "0-100 numeric score or 'N/A'"},
                "passed": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": ["score", "passed", "explanation"]
        }
        for dim in DIMENSIONS
    },
    "required": DIMENSIONS
}


def evaluate_accuracy(
    client: Mistral,
    df: pd.DataFrame,
    profile_text: str,
    syntactic_rules: list[SyntacticRule],
    semantic_rules: list[SemanticRule],
    context_text: str,
    priority: int,
    model: str = "mistral-small-latest",
) -> CharacteristicResult:
    syntactic_results = check_syntactic_rules(df, syntactic_rules)
    semantic_stubs = check_semantic_rules(df, semantic_rules)

    user_message = f"""Context:
{context_text}

Dataset profile:
{profile_text}

Syntactic rule check results:
{json.dumps(syntactic_results, indent=2)}

Semantic rule field statistics (for LLM-based assessment):
{json.dumps(semantic_stubs, indent=2)}

Evaluate all six accuracy dimensions for this dataset. For each dimension:
- syntactic_accuracy: based on syntactic rule pass rates
- semantic_accuracy: based on semantic rule field stats and domain logic
- data_accuracy_assurance: what % of data items were measurable/checkable for accuracy
- risk_of_inaccuracy: presence of outliers or anomalous values (0% risk = good, 100% = bad; invert for score)
- data_accuracy_range: are numeric fields within expected domain ranges
- data_model_accuracy: does the data model (schema, column types, structure) accurately describe the system?
  Consider: are data types appropriate for each field? Does the schema match domain conventions?
  Are there mismatched types, poorly named columns, or structural issues that reduce model accuracy?

Respond with JSON where each key is a dimension name with fields: score (0-100 or null), passed (bool), explanation (string)."""

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
            score=score if score is not None else (raw_score or "N/A"),
            passed=d.get("passed"),
            explanation=d.get("explanation", ""),
        ))

    overall = sum(scores) / len(scores) if scores else None
    summary = f"Overall accuracy score: {overall:.1f}%" if overall is not None else "Could not compute overall score."

    return CharacteristicResult(
        characteristic=QualityCharacteristic.ACCURACY,
        priority=priority,
        overall_score=overall,
        dimensions=dimensions,
        summary=summary,
    )

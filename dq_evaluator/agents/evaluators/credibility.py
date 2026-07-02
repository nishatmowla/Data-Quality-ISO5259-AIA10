"""Credibility evaluator — ISO/IEC 5259 dimensions."""

import json
import pandas as pd
from mistralai.client import Mistral

from dq_evaluator.models.iso5259 import (
    CharacteristicResult, DimensionResult, QualityCharacteristic
)

SYSTEM_PROMPT = """You are a data quality evaluator specializing in the Credibility characteristic of ISO/IEC 5259.
Credibility is the extent to which data is considered believable and reliable within its usage context.
Evaluate all four credibility dimensions. Respond with JSON only."""

DIMENSIONS = ["value_credibility", "source_credibility", "data_dictionary_credibility", "data_model_credibility"]


def evaluate_credibility(
    client: Mistral,
    profile_text: str,
    context_text: str,
    dataset_metadata: dict,
    priority: int,
    model: str = "mistral-small-latest",
) -> CharacteristicResult:
    """
    dataset_metadata: dict with optional keys:
      - source_name: who produced the dataset
      - source_url: link to dataset
      - collection_method: how data was collected
      - documentation_url: link to documentation
      - creation_date: when dataset was created
      - anonymization: any anonymization applied
      - license: data license
    """
    user_message = f"""Context:
{context_text}

Dataset profile:
{profile_text}

Dataset metadata provided:
{json.dumps(dataset_metadata, indent=2)}

Evaluate the four ISO/IEC 5259 credibility dimensions:

- value_credibility: Do data values follow expected behavioral patterns for this domain?
  Consider: do numeric ranges, distributions, and value types match what domain knowledge would predict?
  Are there signs of synthetic data, simulation artifacts, or anonymization that reduce credibility?

- source_credibility: Is the dataset from a qualified, reputable source?
  Consider: is the source named and known? Is there documentation? Is collection methodology described?
  Does it adhere to relevant standards? Is it peer-reviewed or institutionally backed?

- data_dictionary_credibility: Does the dataset have a data dictionary or schema documentation that is validated
  or certified? Consider: are column names and types clearly defined? Are there descriptions/units for each field?
  Is the schema versioned or formally reviewed? Score based on available metadata and column naming conventions.

- data_model_credibility: Does the data model (schema structure, relationships, constraints) provide credible
  and accurate information about the domain? Consider: are data types appropriate? Are relationships between
  fields logically coherent? Is the schema consistent with domain standards for this type of data?

Respond with JSON:
{{
  "value_credibility": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "source_credibility": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "data_dictionary_credibility": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "data_model_credibility": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}}
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
        ))

    overall = sum(scores) / len(scores) if scores else None
    summary = f"Overall credibility score: {overall:.1f}%" if overall is not None else "Could not compute overall score."

    return CharacteristicResult(
        characteristic=QualityCharacteristic.CREDIBILITY,
        priority=priority,
        overall_score=overall,
        dimensions=dimensions,
        summary=summary,
    )

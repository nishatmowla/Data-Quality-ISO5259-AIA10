"""Currentness evaluator — ISO/IEC 5259 dimensions."""

import json
from datetime import datetime
from mistralai import Mistral

from dq_evaluator.models.iso5259 import (
    CharacteristicResult, DimensionResult, QualityCharacteristic
)

SYSTEM_PROMPT = """You are a data quality evaluator specializing in the Currentness characteristic of ISO/IEC 5259.
Currentness is the time difference (ΔT) between when data was recorded and when it is used.
Evaluate feature currentness and record currentness. Respond with JSON only."""

DIMENSIONS = ["feature_currentness", "record_currentness"]


def evaluate_currentness(
    client: Mistral,
    profile_text: str,
    context_text: str,
    dataset_metadata: dict,
    priority: int,
    short_threshold_years: int = 5,
    long_threshold_years: int = 15,
    model: str = "mistral-small-latest",
) -> CharacteristicResult:
    """
    dataset_metadata keys used:
      - creation_date: ISO date string (YYYY-MM-DD or YYYY)
      - last_updated: ISO date string
      - feature_collection_period: e.g., "2018-2021"
    """
    current_year = datetime.now().year

    creation_date = dataset_metadata.get("creation_date", "unknown")
    last_updated = dataset_metadata.get("last_updated")
    feature_period = dataset_metadata.get("feature_collection_period", "unknown")

    # Compute composite currentness metrics (ISO/IEC 5259-2: ΔT, time span, update frequency)
    age_info: dict = {}

    for label, date_str in [("creation_date", creation_date), ("last_updated", last_updated or creation_date)]:
        if date_str and date_str != "unknown":
            try:
                year = int(str(date_str)[:4])
                age_info[label] = {"year": year, "age_years": current_year - year}
            except ValueError:
                age_info[label] = {"raw": date_str}

    # Time span: difference between start and end year of collection period (e.g. "2018-2021")
    time_span_years: int | None = None
    if feature_period and feature_period != "unknown":
        parts = str(feature_period).replace(" ", "").split("-")
        try:
            years = [int(p[:4]) for p in parts if len(p) >= 4]
            if len(years) >= 2:
                time_span_years = years[-1] - years[0]
            elif len(years) == 1:
                time_span_years = 0
        except ValueError:
            pass
    age_info["feature_collection_period"] = feature_period
    age_info["time_span_years"] = time_span_years

    # Age since last update (most relevant for record currentness)
    age_since_update: int | None = None
    if "last_updated" in age_info and "age_years" in age_info["last_updated"]:
        age_since_update = age_info["last_updated"]["age_years"]
    elif "creation_date" in age_info and "age_years" in age_info["creation_date"]:
        age_since_update = age_info["creation_date"]["age_years"]
    age_info["age_since_last_update_years"] = age_since_update

    user_message = f"""Context:
{context_text}

Dataset profile:
{profile_text}

Dataset metadata:
{json.dumps(dataset_metadata, indent=2)}

Composite currentness calculation (current year: {current_year}):
{json.dumps(age_info, indent=2)}

Currentness thresholds for this domain:
- Short threshold (acceptable for fast-evolving domains): {short_threshold_years} years
- Long threshold (acceptable for slow-changing domains): {long_threshold_years} years

Evaluate the two ISO/IEC 5259 currentness dimensions using the composite metric defined in ISO/IEC 5259-2,
which incorporates: (1) age since creation, (2) age since last update, (3) time span of data collection.

- feature_currentness: do the data features/columns reflect current domain knowledge?
  Consider: collection period span = {time_span_years if time_span_years is not None else 'unknown'} years,
  age since last update = {age_since_update if age_since_update is not None else 'unknown'} years.

- record_currentness: are all data records within an acceptable time window?
  Consider: creation date = {creation_date}, age since last update = {age_since_update if age_since_update is not None else 'unknown'} years,
  short threshold = {short_threshold_years}yr, long threshold = {long_threshold_years}yr.

Score interpretation: 100 = fully current (within short threshold), 50 = within long threshold only, 0 = beyond both thresholds.

Respond with JSON:
{{
  "feature_currentness": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}},
  "record_currentness": {{"score": <0-100>, "passed": <bool>, "explanation": "..."}}
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
            details={"age_info": age_info},
        ))

    overall = sum(scores) / len(scores) if scores else None
    summary = f"Overall currentness score: {overall:.1f}%" if overall is not None else "Could not compute overall score."

    return CharacteristicResult(
        characteristic=QualityCharacteristic.CURRENTNESS,
        priority=priority,
        overall_score=overall,
        dimensions=dimensions,
        summary=summary,
    )

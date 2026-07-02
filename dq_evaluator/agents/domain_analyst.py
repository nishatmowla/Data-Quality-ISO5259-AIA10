"""
Agent 1 — Domain Analyst

Given a domain description and dataset profile, this agent:
- Defines the DataUsageContext (scope, purpose, relevant standards)
- Identifies DataQualitySubject (entities affected)
- Suggests which quality characteristics to prioritize and why
"""

import json
from mistralai.client import Mistral

from dq_evaluator.models.iso5259 import DataUsageContext, DataQualitySubject, QualityCharacteristic

SYSTEM_PROMPT = """You are a data quality expert specializing in ISO/IEC 5259 series standards.
Your role is to analyze a dataset's domain context and define the data quality evaluation scope.

You must respond with a valid JSON object matching the schema provided.
Be specific about domain standards that apply and explain your prioritization reasoning."""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {"type": "string", "description": "Domain name (e.g., 'Healthcare', 'Automotive IDS')"},
        "purpose": {"type": "string", "description": "Primary purpose/use case of the dataset"},
        "task_type": {"type": "string", "description": "ML/analytics task (classification, regression, anomaly_detection, etc.)"},
        "standards": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Relevant domain standards and regulations"
        },
        "notes": {"type": "string", "description": "Additional context notes"},
        "subject_entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Entities affected by data quality (e.g., ML models, end users, regulators)"
        },
        "stakeholders": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Stakeholder groups"
        },
        "characteristic_priorities": {
            "type": "object",
            "description": "Priority 1-3 for each characteristic (1=highest). Must include all six.",
            "properties": {
                "accuracy": {"type": "integer", "minimum": 1, "maximum": 3},
                "completeness": {"type": "integer", "minimum": 1, "maximum": 3},
                "consistency": {"type": "integer", "minimum": 1, "maximum": 3},
                "diversity": {"type": "integer", "minimum": 1, "maximum": 3},
                "credibility": {"type": "integer", "minimum": 1, "maximum": 3},
                "currentness": {"type": "integer", "minimum": 1, "maximum": 3},
            }
        },
        "priority_rationale": {"type": "string", "description": "Explanation of prioritization choices"}
    },
    "required": ["domain", "purpose", "task_type", "standards", "subject_entities",
                 "stakeholders", "characteristic_priorities", "priority_rationale"]
}


def analyze_domain(
    client: Mistral,
    domain_description: str,
    dataset_profile_text: str,
    model: str = "mistral-large-latest",
) -> tuple[DataUsageContext, DataQualitySubject, dict[QualityCharacteristic, int]]:
    """
    Returns:
        context: DataUsageContext
        subject: DataQualitySubject
        priorities: mapping of QualityCharacteristic -> priority (1=highest)
    """
    user_message = f"""Domain description provided by the user:
{domain_description}

Dataset profile:
{dataset_profile_text}

Analyze this domain and dataset. Define:
1. The data usage context (purpose, task type, applicable standards)
2. The data quality subjects (who/what is affected)
3. Priority ranking of the six ISO/IEC 5259 quality characteristics for this domain

Respond with a JSON object matching this schema:
{json.dumps(ANALYSIS_SCHEMA, indent=2)}"""

    response = client.chat.complete(
        timeout_ms=120000,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    result = json.loads(response.choices[0].message.content)

    context = DataUsageContext(
        domain=result["domain"],
        purpose=result["purpose"],
        task_type=result["task_type"],
        standards=result.get("standards", []),
        notes=result.get("notes", result.get("priority_rationale", "")),
    )

    subject = DataQualitySubject(
        entities=result.get("subject_entities", []),
        stakeholders=result.get("stakeholders", []),
    )

    raw_priorities = result.get("characteristic_priorities", {})
    priorities = {
        QualityCharacteristic(k): v
        for k, v in raw_priorities.items()
        if k in QualityCharacteristic._value2member_map_
    }

    return context, subject, priorities

"""
Agent 2 — Rule Generator

Given domain context and dataset profile, generates:
- Syntactic rules (format, dtype, range, enum, length)
- Semantic rules (cross-field logic, domain constraints)

Rules are grounded in domain standards and the specific dataset columns.
"""

import json
from mistralai.client import Mistral

from dq_evaluator.models.iso5259 import (
    DataUsageContext, SemanticRule, SyntacticRule
)

SYSTEM_PROMPT = """You are a data quality engineer with deep expertise in ISO/IEC 5259 and domain-specific data standards.
Your task is to generate concrete data quality rules for a dataset based on its domain context and column profiles.

Syntactic rules check structural/format conformance.
Semantic rules check meaning, cross-field logic, and domain business constraints.

Respond only with a valid JSON object."""

RULES_SCHEMA = {
    "type": "object",
    "properties": {
        "syntactic_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "rule_type": {"type": "string", "enum": ["format", "range", "enum", "dtype", "length", "not_null"]},
                    "specification": {
                        "description": "For 'range': {min, max}. For 'enum': list of allowed values. For 'format': regex string. For 'dtype': type name. For 'length': {min, max}. For 'not_null': true."
                    },
                    "description": {"type": "string"},
                    "standard_reference": {"type": "string"}
                },
                "required": ["field", "rule_type", "specification", "description"]
            }
        },
        "semantic_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "fields_involved": {"type": "array", "items": {"type": "string"}},
                    "rule_type": {"type": "string", "enum": ["cross_field", "domain_logic", "temporal", "referential"]},
                    "standard_reference": {"type": "string"}
                },
                "required": ["name", "description", "fields_involved", "rule_type"]
            }
        }
    },
    "required": ["syntactic_rules", "semantic_rules"]
}


def generate_rules(
    client: Mistral,
    context: DataUsageContext,
    dataset_profile_text: str,
    model: str = "mistral-large-latest",
) -> tuple[list[SyntacticRule], list[SemanticRule]]:
    """Generate syntactic and semantic rules for the dataset."""

    standards_text = ", ".join(context.standards) if context.standards else "general data quality standards"

    user_message = f"""Domain: {context.domain}
Purpose: {context.purpose}
Task type: {context.task_type}
Applicable standards: {standards_text}

Dataset profile:
{dataset_profile_text}

Generate specific syntactic and semantic data quality rules for this dataset.

Guidelines:
- Syntactic rules: cover field formats, valid value ranges/enums, required types, and length constraints for each column
- Semantic rules: cover cross-field dependencies, domain logic (e.g., timestamps must be ordered, attack labels must correspond to known attack types), and any constraints implied by the domain standards
- Reference the applicable standards where relevant
- Be specific about field names found in the dataset profile above

Respond with JSON matching this schema:
{json.dumps(RULES_SCHEMA, indent=2)}"""

    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    result = json.loads(response.choices[0].message.content)

    syntactic_rules = [
        SyntacticRule(
            field=r["field"],
            rule_type=r["rule_type"],
            specification=r["specification"],
            description=r["description"],
            standard_reference=r.get("standard_reference", ""),
        )
        for r in result.get("syntactic_rules", [])
    ]

    semantic_rules = [
        SemanticRule(
            name=r["name"],
            description=r["description"],
            fields_involved=r.get("fields_involved", []),
            rule_type=r["rule_type"],
            standard_reference=r.get("standard_reference", ""),
        )
        for r in result.get("semantic_rules", [])
    ]

    return syntactic_rules, semantic_rules

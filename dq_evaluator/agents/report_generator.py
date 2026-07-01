"""
Agent — Report Generator

Synthesizes all evaluation results into a structured final report with:
- Per-characteristic scores and findings
- Priority-weighted overall assessment
- Recommendations
- AI Act Article 10 compliance notes
"""

import json
from mistralai.client import Mistral

from dq_evaluator.models.iso5259 import (
    DataQualityReport, DataUsageContext, DataQualitySubject, CharacteristicResult, QualityCharacteristic
)

SYSTEM_PROMPT = """You are a data quality reporting expert aligned with ISO/IEC 5259 and EU AI Act Article 10.
Synthesize evaluation results into a concise, actionable report.
Respond with JSON only."""


def _results_to_text(results: list[CharacteristicResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"\n### {r.characteristic.value.upper()} (priority={r.priority}, score={r.overall_score})")
        lines.append(r.summary)
        for dim in r.dimensions:
            lines.append(f"  - {dim.dimension}: score={dim.score}, passed={dim.passed}")
            lines.append(f"    {dim.explanation}")
    return "\n".join(lines)


def generate_report(
    client: Mistral,
    dataset_name: str,
    context: DataUsageContext,
    subject: DataQualitySubject,
    characteristic_results: list[CharacteristicResult],
    model: str = "mistral-large-latest",
) -> DataQualityReport:

    results_text = _results_to_text(characteristic_results)
    priorities = {r.characteristic: r.priority for r in characteristic_results}

    user_message = f"""Dataset: {dataset_name}
Domain: {context.domain}
Purpose: {context.purpose}
Task type: {context.task_type}
Applicable standards: {', '.join(context.standards)}
Data quality subjects: {', '.join(subject.entities)}

Evaluation results (sorted by priority):
{results_text}

Priority mapping (1=highest): {{{', '.join(f'{k.value}: {v}' for k, v in priorities.items())}}}

Generate a final data quality report. Respond with JSON:
{{
  "overall_assessment": "<2-3 sentence summary of overall data quality, weighted by priorities>",
  "recommendations": ["<actionable recommendation 1>", "<recommendation 2>", ...],
  "ai_act_compliance_notes": "<Assessment against EU AI Act Article 10 requirements: free of errors, complete, relevant, sufficiently representative. Note gaps.>"
}}"""

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

    # Normalise recommendations — LLM sometimes returns dicts or a single string
    recs_raw = result.get("recommendations", [])
    if isinstance(recs_raw, str):
        recs_raw = [recs_raw]
    recommendations = []
    for r in recs_raw:
        if isinstance(r, dict):
            r = r.get("recommendation") or r.get("text") or str(r)
        recommendations.append(str(r))

    return DataQualityReport(
        dataset_name=dataset_name,
        context=context,
        subject=subject,
        characteristic_results=characteristic_results,
        overall_assessment=str(result.get("overall_assessment", "")),
        recommendations=recommendations,
        ai_act_compliance_notes=str(result.get("ai_act_compliance_notes", "")),
    )


def format_report_text(report: DataQualityReport) -> str:
    """Render the report as readable text."""
    lines = [
        f"# Data Quality Evaluation Report: {report.dataset_name}",
        f"",
        f"## Context",
        f"- Domain: {report.context.domain}",
        f"- Purpose: {report.context.purpose}",
        f"- Task type: {report.context.task_type}",
        f"- Standards: {', '.join(report.context.standards) or 'N/A'}",
        f"- Data subjects: {', '.join(report.subject.entities)}",
        f"",
        f"## Overall Assessment",
        report.overall_assessment,
        f"",
        f"## Quality Characteristic Results",
        f"(Ordered by priority — 1 = highest)",
        f"",
    ]

    sorted_results = sorted(report.characteristic_results, key=lambda r: r.priority)
    for r in sorted_results:
        score_str = f"{r.overall_score:.1f}%" if r.overall_score is not None else "N/A"
        lines.append(f"### {r.characteristic.value.title()}  |  Priority: {r.priority}  |  Score: {score_str}")
        lines.append(r.summary)
        lines.append("")
        for dim in r.dimensions:
            passed_str = "PASS" if dim.passed else ("FAIL" if dim.passed is False else "N/A")
            score_disp = f"{dim.score:.1f}%" if isinstance(dim.score, float) else str(dim.score)
            lines.append(f"  - **{dim.dimension}**: {score_disp} [{passed_str}]")
            lines.append(f"    {dim.explanation}")
        lines.append("")

    lines += [
        f"## Recommendations",
    ]
    for i, rec in enumerate(report.recommendations, 1):
        # LLM occasionally returns dicts instead of strings
        if isinstance(rec, dict):
            rec = rec.get("recommendation") or rec.get("text") or str(rec)
        lines.append(f"{i}. {rec}")

    compliance = report.ai_act_compliance_notes
    if isinstance(compliance, dict):
        compliance = str(compliance)

    lines += [
        f"",
        f"## EU AI Act Article 10 Compliance Notes",
        compliance or "",
    ]

    return "\n".join(str(l) for l in lines)

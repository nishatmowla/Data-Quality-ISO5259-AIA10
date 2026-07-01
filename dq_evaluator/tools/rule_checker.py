"""Apply syntactic and semantic rules to a dataset and return violation counts."""

import re
from typing import Any

import pandas as pd

from dq_evaluator.models.iso5259 import SemanticRule, SyntacticRule


def check_syntactic_rules(df: pd.DataFrame, rules: list[SyntacticRule]) -> dict:
    """Return per-rule violation statistics."""
    results = {}
    for rule in rules:
        col = rule.field
        if col not in df.columns:
            results[rule.field] = {
                "rule_type": rule.rule_type,
                "description": rule.description,
                "status": "column_missing",
                "violations": None,
                "violation_rate": None,
            }
            continue

        series = df[col].dropna()
        violations = _apply_syntactic_rule(series, rule)
        total = len(df)
        results[f"{col}:{rule.rule_type}"] = {
            "rule_type": rule.rule_type,
            "field": col,
            "description": rule.description,
            "violations": int(violations),
            "total": total,
            "violation_rate": violations / total if total > 0 else 0.0,
            "pass_rate": 1.0 - (violations / total) if total > 0 else 1.0,
        }
    return results


def check_semantic_rules(df: pd.DataFrame, rules: list[SemanticRule]) -> dict:
    """Semantic rules are expressed as natural language; we return a stub result with field stats for LLM evaluation."""
    results = {}
    for rule in rules:
        field_stats = {}
        for field in rule.fields_involved:
            if field in df.columns:
                s = df[field]
                field_stats[field] = {
                    "dtype": str(s.dtype),
                    "null_rate": float(s.isna().mean()),
                    "unique": int(s.nunique()),
                    "sample": s.dropna().head(5).tolist(),
                }
        results[rule.name] = {
            "description": rule.description,
            "rule_type": rule.rule_type,
            "fields": rule.fields_involved,
            "field_stats": field_stats,
            "note": "Requires LLM evaluation of semantic conformance",
        }
    return results


def _apply_syntactic_rule(series: pd.Series, rule: SyntacticRule) -> int:
    spec = rule.specification
    rt = rule.rule_type

    if rt == "dtype":
        try:
            series.astype(spec)
            return 0
        except (ValueError, TypeError):
            return len(series)

    if rt == "range":
        lo, hi = spec.get("min"), spec.get("max")
        mask = pd.Series([False] * len(series), index=series.index)
        if lo is not None:
            mask |= series < lo
        if hi is not None:
            mask |= series > hi
        return int(mask.sum())

    if rt == "enum":
        allowed = set(spec) if not isinstance(spec, set) else spec
        return int((~series.isin(allowed)).sum())

    if rt == "format":
        pattern = re.compile(spec)
        return int(series.astype(str).apply(lambda x: not pattern.match(x)).sum())

    if rt == "length":
        lo = spec.get("min", 0)
        hi = spec.get("max", float("inf"))
        lengths = series.astype(str).str.len()
        return int(((lengths < lo) | (lengths > hi)).sum())

    if rt == "not_null":
        return int(series.isna().sum())

    return 0

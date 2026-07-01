"""
ISO/IEC 5259 data quality model structures.

The model has four core elements:
- DataUsageContext: scope and purpose
- DataQualitySubject: entities affected by data quality
- DataQualityCharacteristic: categories (accuracy, completeness, etc.)
- DataQualityRequirement: specific acceptance criteria per characteristic
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualityCharacteristic(str, Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    DIVERSITY = "diversity"
    CREDIBILITY = "credibility"
    CURRENTNESS = "currentness"


# Sub-dimensions per characteristic, as per ISO/IEC 5259 and the RISE methodology
CHARACTERISTIC_DIMENSIONS: dict[QualityCharacteristic, list[str]] = {
    QualityCharacteristic.ACCURACY: [
        "syntactic_accuracy",
        "semantic_accuracy",
        "data_accuracy_assurance",
        "risk_of_inaccuracy",
        "data_accuracy_range",
    ],
    QualityCharacteristic.COMPLETENESS: [
        "value_completeness",
        "value_occurrence_completeness",
        "feature_completeness",
        "record_completeness",
        "label_completeness",
    ],
    QualityCharacteristic.CONSISTENCY: [
        "data_record_consistency",
        "distribution_of_data_values",
        "data_format_consistency",
        "semantic_consistency",
    ],
    QualityCharacteristic.DIVERSITY: [
        "label_richness",
        "relative_label_abundance",
        "category_size_diversity",
    ],
    QualityCharacteristic.CREDIBILITY: [
        "value_credibility",
        "source_credibility",
    ],
    QualityCharacteristic.CURRENTNESS: [
        "feature_currentness",
        "record_currentness",
    ],
}


@dataclass
class DataUsageContext:
    domain: str
    purpose: str
    task_type: str  # e.g., "classification", "anomaly detection", "regression"
    standards: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DataQualitySubject:
    entities: list[str]  # who/what is affected by data quality
    stakeholders: list[str] = field(default_factory=list)


@dataclass
class SyntacticRule:
    field: str
    rule_type: str  # "format", "range", "enum", "dtype", "length"
    specification: Any
    description: str
    standard_reference: str = ""


@dataclass
class SemanticRule:
    name: str
    description: str
    fields_involved: list[str]
    rule_type: str  # "cross_field", "domain_logic", "temporal", "referential"
    standard_reference: str = ""


@dataclass
class DataQualityRequirement:
    characteristic: QualityCharacteristic
    dimension: str
    description: str
    acceptance_criterion: str
    priority: int = 1  # 1=highest, 3=lowest
    syntactic_rules: list[SyntacticRule] = field(default_factory=list)
    semantic_rules: list[SemanticRule] = field(default_factory=list)


@dataclass
class DimensionResult:
    dimension: str
    score: float | str  # numeric 0-100 or qualitative
    passed: bool | None
    explanation: str
    details: dict = field(default_factory=dict)


@dataclass
class CharacteristicResult:
    characteristic: QualityCharacteristic
    priority: int
    overall_score: float | None
    dimensions: list[DimensionResult]
    summary: str


@dataclass
class DataQualityReport:
    dataset_name: str
    context: DataUsageContext
    subject: DataQualitySubject
    characteristic_results: list[CharacteristicResult]
    overall_assessment: str
    recommendations: list[str]
    ai_act_compliance_notes: str = ""

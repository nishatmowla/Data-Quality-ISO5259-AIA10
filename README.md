# Agentic Data Quality Evaluator

An agentic application for evaluating dataset quality against the **ISO/IEC 5259 series** standards, powered by **Mistral AI**. Designed to be domain-agnostic: it generates domain-specific syntactic and semantic rules automatically, presents them to a human reviewer for approval, then runs a structured multi-agent evaluation pipeline.

Originally developed at **RISE Research Institutes of Sweden** as part of the INTERSTICE project, grounded in work on data quality evaluation for automotive intrusion detection systems (IDS).

---

## Overview

Data quality evaluation is essential for trustworthy AI, particularly under the **EU AI Act Article 10** data governance requirements. This tool implements the ISO/IEC 5259 data quality model — adapted from the RISE automotive IDS methodology — as a reusable, domain-independent pipeline.

### The 9-Agent Pipeline

```
Dataset + Domain Description
        │
  [Agent 1] Domain Analyst          ← mistral-large
        │  DataUsageContext · DataQualitySubject · Characteristic priorities
        │
  [Agent 2] Rule Generator          ← mistral-large
        │  Syntactic rules (format/range/enum/dtype)
        │  Semantic rules (cross-field/domain-logic)
        │
  ── ✋ HUMAN REVIEW & APPROVAL ─────────────────────────
        │  Edit, add, or delete rules before proceeding
  ──────────────────────────────────────────────────────
        │
  [Agent 3]  Accuracy               ← mistral-small
  [Agent 4]  Completeness
  [Agent 5]  Consistency
  [Agent 6]  Diversity
  [Agent 7]  Credibility
  [Agent 8]  Currentness
        │
  [Agent 9] Report Generator        ← mistral-large
        │
  Markdown Report (ISO/IEC 5259 + EU AI Act Article 10 notes)
```

### Quality Characteristics (ISO/IEC 5259)

| Characteristic | Sub-dimensions |
|---|---|
| **Accuracy** | Syntactic accuracy · Semantic accuracy · Data accuracy assurance · Risk of inaccuracy · Data accuracy range |
| **Completeness** | Value completeness · Value occurrence completeness · Feature completeness · Record completeness · Label completeness |
| **Consistency** | Data record consistency · Distribution of data values · Data format consistency · Semantic consistency |
| **Diversity** | Label richness · Relative label abundance · Category size diversity |
| **Credibility** | Value credibility · Source credibility |
| **Currentness** | Feature currentness · Record currentness |

Each characteristic has a configurable **priority (1–3)**, allowing domain experts to weight evaluation focus (e.g. accuracy and completeness are typically highest priority for ML training data).

---

## Interfaces

### Streamlit Web UI (recommended)
Full three-phase UI: Input → Rule Review & Approval → Results.

```bash
streamlit run dq_evaluator/app_streamlit.py
```

### Gradio Web UI
Two-step side-by-side layout, easy to share externally.

```bash
python dq_evaluator/app_gradio.py
```

### Jupyter Notebook
Step-by-step interactive walkthrough with visualisations.

```bash
jupyter notebook dq_evaluator/demo.ipynb
```

### Command Line

```bash
python -m dq_evaluator.main \
  --dataset data.csv \
  --domain "Healthcare claims data for fraud detection AI models" \
  --dataset-name "Claims 2024" \
  --priority "accuracy=1,completeness=1,diversity=2,consistency=2,credibility=3,currentness=3" \
  --metadata '{"source_name": "NHS", "creation_date": "2024", "license": "CC BY 4.0"}' \
  --output report.md
```

---

## Installation

**Requirements:** Python 3.12+, Mistral API key

```bash
# Clone
git clone git@github.com:nishatmowla/Data-Quality-ISO5259-AIA10.git
cd Data-Quality-ISO5259-AIA10

# Install
pip install -e dq_evaluator/

# If using Anaconda with NumPy 2.x (fixes compatibility errors)
pip install --upgrade pyarrow numexpr bottleneck matplotlib pandas

# Set API key
export MISTRAL_API_KEY="your_key_here"
```

Dependencies: `mistralai`, `pandas`, `numpy`, `matplotlib`, `streamlit`, `gradio`, `openpyxl`, `pyarrow`

---

## Supported Dataset Formats

| Format | Extension |
|---|---|
| CSV | `.csv` |
| JSON / JSON Lines | `.json` / `.jsonl` |
| Parquet | `.parquet` |
| Excel | `.xlsx` / `.xls` |

---

## Project Structure

```
dq_evaluator/
├── main.py                         # Orchestrator + CLI entry point
├── app_streamlit.py                # Streamlit web UI (3-phase, human-in-the-loop)
├── app_gradio.py                   # Gradio web UI (2-step, shareable)
├── demo.ipynb                      # Jupyter notebook demo
├── requirements.txt
├── setup.py
│
├── models/
│   └── iso5259.py                  # ISO/IEC 5259 data structures & types
│
├── tools/
│   ├── data_profiler.py            # Statistical dataset profiling
│   └── rule_checker.py             # Programmatic syntactic rule enforcement
│
└── agents/
    ├── domain_analyst.py           # Agent 1: Domain context & priorities
    ├── rule_generator.py           # Agent 2: Syntactic + semantic rules
    ├── report_generator.py         # Agent 9: Final report synthesis
    └── evaluators/
        ├── accuracy.py             # Agent 3
        ├── completeness.py         # Agent 4
        ├── consistency.py          # Agent 5
        ├── diversity.py            # Agent 6
        ├── credibility.py          # Agent 7
        └── currentness.py          # Agent 8
```

---

## Dataset Metadata

Providing metadata improves credibility and currentness evaluation. Pass as JSON:

| Key | Description | Example |
|---|---|---|
| `source_name` | Organisation that produced the dataset | `"RISE Research Institutes of Sweden"` |
| `creation_date` | Year or ISO date the dataset was created | `"2024"` |
| `last_updated` | Year or ISO date of last update | `"2025"` |
| `collection_method` | How data was collected | `"Real vehicle CAN traces"` |
| `license` | Data license | `"CC BY 4.0"` |
| `feature_collection_period` | Time span of data collection | `"2022-2024"` |

---

## Human-in-the-Loop Rule Review

After Agent 2 generates rules, the application pauses for human review before any evaluation runs. In both UIs you can:

- **Edit** any rule inline (field, type, specification, description, standard reference)
- **Delete** rules that are incorrect or irrelevant for your domain
- **Add** new rules the agent missed
- **Go back** to change the domain description and regenerate

Only the approved rule set is passed to the evaluators.

### Syntactic Rule Types

| Type | Specification format | Example |
|---|---|---|
| `range` | `{"min": 0, "max": 8}` | DLC field must be 0–8 |
| `enum` | `["A", "B", "C"]` | Status must be one of allowed values |
| `format` | Regex string | Timestamp must match `\d{4}-\d{2}-\d{2}` |
| `dtype` | `int64` / `float64` / `object` | Field must be numeric |
| `length` | `{"min": 1, "max": 255}` | String field length constraint |
| `not_null` | `true` | Field must not be null |

---

## Output Report

The generated Markdown report includes:

- **Context**: domain, purpose, task type, applicable standards
- **Overall assessment**: priority-weighted summary
- **Per-characteristic results**: score (0–100%) and pass/fail for every sub-dimension with explanation
- **Recommendations**: actionable improvement suggestions
- **EU AI Act Article 10 compliance notes**: gaps in completeness, representativeness, accuracy, and error-freedom

---

## Background & Methodology

This tool implements the data quality evaluation methodology developed at RISE Research Institutes of Sweden for the INTERSTICE project. The methodology was originally applied to automotive IDS datasets (SOME/IP and SAD) and published as *RISE Report 2025:43*.

The ISO/IEC 5259 data quality model defines four elements:
1. **DataUsageContext** — scope and purpose of the data
2. **DataQualitySubject** — entities affected by data quality
3. **DataQualityCharacteristic** — the six quality dimensions above
4. **DataQualityRequirement** — acceptance criteria per dimension

Three types of expertise are encoded in the agents:
- **Domain Expertise (DOEX)** — knowledge about the domain and its data
- **Data Expertise (DAEX)** — understanding of data quality attributes and methodologies
- **Standard Expertise (SAEX)** — familiarity with ISO/IEC 5259 and domain-specific standards

---

## Example Domains

The tool has been tested on:
- Automotive network intrusion detection (CAN bus, SOME/IP)
- Botanical taxonomy classification (Iris dataset)
- Healthcare / clinical records
- Industrial IoT sensor data

Any tabular dataset with a domain description works.

---

## License

This project is licensed under the **CC BY 4.0** License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this tool in research, please cite:

> Mowla, N. I. et al. *Data Quality Evaluation for Automotive Intrusion Detection Systems*. RISE Report 2025:43. RISE Research Institutes of Sweden, 2025.

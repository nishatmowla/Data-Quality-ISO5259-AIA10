"""
Gradio web UI — Agentic Data Quality Evaluator
Human-in-the-loop: rules are shown for review/edit before evaluation runs.

Run:
    python dq_evaluator/app_gradio.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))

from mistralai.client import Mistral
from dq_evaluator.tools.data_profiler import profile_dataset, profile_to_text
from dq_evaluator.tools.rule_checker import check_syntactic_rules
from dq_evaluator.agents.domain_analyst import analyze_domain
from dq_evaluator.agents.rule_generator import generate_rules
from dq_evaluator.agents.evaluators.accuracy    import evaluate_accuracy
from dq_evaluator.agents.evaluators.completeness import evaluate_completeness
from dq_evaluator.agents.evaluators.consistency  import evaluate_consistency
from dq_evaluator.agents.evaluators.diversity    import evaluate_diversity
from dq_evaluator.agents.evaluators.credibility  import evaluate_credibility
from dq_evaluator.agents.evaluators.currentness  import evaluate_currentness
from dq_evaluator.agents.report_generator import generate_report, format_report_text
from dq_evaluator.models.iso5259 import (
    QualityCharacteristic, SyntacticRule, SemanticRule
)

CHAR_COLORS = {1: "#2196F3", 2: "#FF9800", 3: "#9E9E9E"}


def build_context_text(context, subject) -> str:
    return (
        f"Domain: {context.domain}\n"
        f"Purpose: {context.purpose}\n"
        f"Task type: {context.task_type}\n"
        f"Standards: {', '.join(context.standards) or 'N/A'}\n"
        f"Data subjects: {', '.join(subject.entities)}"
    )


def syn_rules_to_df(rules: list[SyntacticRule]) -> pd.DataFrame:
    return pd.DataFrame([{
        "field": r.field,
        "rule_type": r.rule_type,
        "specification": json.dumps(r.specification) if not isinstance(r.specification, str) else r.specification,
        "description": r.description,
        "standard_reference": r.standard_reference,
    } for r in rules])


def sem_rules_to_df(rules: list[SemanticRule]) -> pd.DataFrame:
    return pd.DataFrame([{
        "name": r.name,
        "rule_type": r.rule_type,
        "fields_involved": ", ".join(r.fields_involved),
        "description": r.description,
        "standard_reference": r.standard_reference,
    } for r in rules])


def df_to_syntactic(df: pd.DataFrame) -> list[SyntacticRule]:
    rules = []
    for _, row in df.iterrows():
        if not str(row.get("field", "")).strip():
            continue
        spec_raw = row.get("specification", "")
        try:
            spec = json.loads(spec_raw) if isinstance(spec_raw, str) else spec_raw
        except (json.JSONDecodeError, TypeError):
            spec = spec_raw
        rules.append(SyntacticRule(
            field=str(row["field"]),
            rule_type=str(row["rule_type"]),
            specification=spec,
            description=str(row.get("description", "")),
            standard_reference=str(row.get("standard_reference", "")),
        ))
    return rules


def df_to_semantic(df: pd.DataFrame) -> list[SemanticRule]:
    rules = []
    for _, row in df.iterrows():
        if not str(row.get("name", "")).strip():
            continue
        fields_raw = row.get("fields_involved", "")
        fields = [f.strip() for f in str(fields_raw).split(",") if f.strip()]
        rules.append(SemanticRule(
            name=str(row["name"]),
            description=str(row.get("description", "")),
            fields_involved=fields,
            rule_type=str(row.get("rule_type", "domain_logic")),
            standard_reference=str(row.get("standard_reference", "")),
        ))
    return rules


def make_charts(results) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    names  = [r.characteristic.value.title() for r in results]
    scores = [r.overall_score or 0 for r in results]
    colors = [CHAR_COLORS[r.priority] for r in results]
    bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (%)")
    ax.set_title("Quality Characteristic Scores", fontweight="bold")
    for bar, score in zip(bars, scores):
        ax.text(score + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}%", va="center", fontsize=9)
    ax.legend(handles=[
        mpatches.Patch(color=CHAR_COLORS[1], label="Priority 1 (highest)"),
        mpatches.Patch(color=CHAR_COLORS[2], label="Priority 2"),
        mpatches.Patch(color=CHAR_COLORS[3], label="Priority 3"),
    ], fontsize=8)

    ax2 = axes[1]
    max_dims = max(len(r.dimensions) for r in results)
    matrix = np.full((len(results), max_dims), np.nan)
    for i, r in enumerate(results):
        for j, d in enumerate(r.dimensions):
            s = d.score if isinstance(d.score, (int, float)) else None
            if s is not None:
                matrix[i, j] = float(s)
    im = ax2.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax2.set_yticks(range(len(results)))
    ax2.set_yticklabels([r.characteristic.value.title() for r in results])
    ax2.set_xticks(range(max_dims))
    ax2.set_xticklabels([f"D{i+1}" for i in range(max_dims)], fontsize=8)
    ax2.set_title("Dimension Heatmap", fontweight="bold")
    plt.colorbar(im, ax=ax2, label="Score (%)")
    for i in range(len(results)):
        for j in range(max_dims):
            v = matrix[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    return fig


# ── Phase 1: Generate rules ───────────────────────────────────────────────────
def phase1_generate_rules(
    api_key, dataset_file, domain_desc, dataset_name,
    source_name, creation_date, license_, collection_method,
    prio_acc, prio_comp, prio_cons, prio_div, prio_cred, prio_curr,
    model_large, model_small,
    short_thresh, long_thresh,
    progress=gr.Progress(),
):
    if not api_key:
        raise gr.Error("Mistral API key is required.")
    if dataset_file is None:
        raise gr.Error("Upload a dataset first.")
    if not domain_desc.strip():
        raise gr.Error("Domain description is required.")

    client = Mistral(api_key=api_key)
    path = Path(dataset_file.name)
    suffix = path.suffix.lower()

    progress(0.1, desc="Loading dataset…")
    readers = {".csv": pd.read_csv, ".xlsx": pd.read_excel,
               ".xls": pd.read_excel, ".parquet": pd.read_parquet}
    df = readers.get(suffix, lambda p: pd.read_json(p, lines=(suffix == ".jsonl")))(path)
    profile = profile_dataset(df)
    profile_text = profile_to_text(profile)

    progress(0.3, desc="Agent 1: Analyzing domain…")
    context, subject, auto_priorities = analyze_domain(
        client, domain_desc, profile_text, model=model_large
    )
    priority_overrides = {
        QualityCharacteristic.ACCURACY:     int(prio_acc),
        QualityCharacteristic.COMPLETENESS: int(prio_comp),
        QualityCharacteristic.CONSISTENCY:  int(prio_cons),
        QualityCharacteristic.DIVERSITY:    int(prio_div),
        QualityCharacteristic.CREDIBILITY:  int(prio_cred),
        QualityCharacteristic.CURRENTNESS:  int(prio_curr),
    }
    priorities = {**auto_priorities, **priority_overrides}
    for c in QualityCharacteristic:
        priorities.setdefault(c, 2)

    progress(0.6, desc="Agent 2: Generating rules…")
    syn_rules, sem_rules = generate_rules(client, context, profile_text, model=model_large)

    syn_df = syn_rules_to_df(syn_rules)
    sem_df = sem_rules_to_df(sem_rules)

    # Pack state for phase 2
    state = {
        "api_key": api_key,
        "df": df.to_dict(),
        "profile": profile,
        "profile_text": profile_text,
        "context_domain": context.domain,
        "context_purpose": context.purpose,
        "context_task_type": context.task_type,
        "context_standards": context.standards,
        "context_notes": context.notes,
        "subject_entities": subject.entities,
        "subject_stakeholders": subject.stakeholders,
        "priorities": {k.value: v for k, v in priorities.items()},
        "context_text": build_context_text(context, subject),
        "metadata": {
            "source_name": source_name, "creation_date": creation_date,
            "license": license_, "collection_method": collection_method,
        },
        "dataset_name": dataset_name or path.stem,
        "model_large": model_large,
        "model_small": model_small,
        "short_thresh": int(short_thresh),
        "long_thresh": int(long_thresh),
    }

    info_md = (
        f"**Domain:** {context.domain}  \n"
        f"**Task:** {context.task_type}  \n"
        f"**Standards detected:** {', '.join(context.standards) or 'N/A'}  \n\n"
        f"✅ Generated **{len(syn_rules)} syntactic** and **{len(sem_rules)} semantic** rules.  \n"
        "Review and edit the tables below, then click **Approve & Evaluate**."
    )

    return syn_df, sem_df, info_md, state, gr.update(visible=True), gr.update(visible=False)


# ── Phase 2: Evaluate ─────────────────────────────────────────────────────────
def phase2_evaluate(
    syn_df: pd.DataFrame,
    sem_df: pd.DataFrame,
    state: dict,
    progress=gr.Progress(),
):
    if not state:
        raise gr.Error("Run rule generation first.")

    from dq_evaluator.models.iso5259 import DataUsageContext, DataQualitySubject

    client = Mistral(api_key=state["api_key"])
    df = pd.DataFrame(state["df"])
    profile = state["profile"]
    profile_text = state["profile_text"]
    context_text = state["context_text"]
    metadata = state["metadata"]
    model_small = state["model_small"]
    model_large = state["model_large"]
    priorities = {QualityCharacteristic(k): v for k, v in state["priorities"].items()}

    context = DataUsageContext(
        domain=state["context_domain"],
        purpose=state["context_purpose"],
        task_type=state["context_task_type"],
        standards=state["context_standards"],
        notes=state["context_notes"],
    )
    subject = DataQualitySubject(
        entities=state["subject_entities"],
        stakeholders=state["subject_stakeholders"],
    )

    syn_rules = df_to_syntactic(syn_df)
    sem_rules = df_to_semantic(sem_df)

    logs = [f"Human approved {len(syn_rules)} syntactic + {len(sem_rules)} semantic rules."]
    results = []

    for i, (label, fn, kwargs) in enumerate([
        ("Accuracy",    evaluate_accuracy,    dict(
            df=df, profile_text=profile_text,
            syntactic_rules=syn_rules, semantic_rules=sem_rules,
            context_text=context_text,
            priority=priorities[QualityCharacteristic.ACCURACY], model=model_small)),
        ("Completeness", evaluate_completeness, dict(
            df=df, profile=profile, profile_text=profile_text,
            context_text=context_text,
            priority=priorities[QualityCharacteristic.COMPLETENESS], model=model_small)),
        ("Consistency",  evaluate_consistency,  dict(
            df=df, profile_text=profile_text, semantic_rules=sem_rules,
            context_text=context_text,
            priority=priorities[QualityCharacteristic.CONSISTENCY], model=model_small)),
        ("Diversity",    evaluate_diversity,    dict(
            df=df, profile=profile, profile_text=profile_text,
            context_text=context_text,
            priority=priorities[QualityCharacteristic.DIVERSITY], model=model_small)),
        ("Credibility",  evaluate_credibility,  dict(
            profile_text=profile_text, context_text=context_text,
            dataset_metadata=metadata,
            priority=priorities[QualityCharacteristic.CREDIBILITY], model=model_small)),
        ("Currentness",  evaluate_currentness,  dict(
            profile_text=profile_text, context_text=context_text,
            dataset_metadata=metadata,
            priority=priorities[QualityCharacteristic.CURRENTNESS],
            short_threshold_years=state["short_thresh"],
            long_threshold_years=state["long_thresh"], model=model_small)),
    ], start=3):
        progress(0.1 + i * 0.12, desc=f"Agent {i}: {label}…")
        logs.append(f"Agent {i}: Evaluating {label}…")
        results.append(fn(client, **kwargs))

    progress(0.92, desc="Agent 9: Generating report…")
    logs.append("Agent 9: Generating report…")
    report = generate_report(
        client, state["dataset_name"], context, subject, results, model=model_large
    )
    report_text = format_report_text(report)
    logs.append("Done ✅")

    # Score summary table
    score_rows = [{"Characteristic": r.characteristic.value.title(),
                   "Priority": r.priority,
                   "Score": f"{r.overall_score:.1f}%" if r.overall_score else "N/A",
                   "Summary": r.summary}
                  for r in sorted(results, key=lambda x: x.priority)]

    # Rule check table
    check = check_syntactic_rules(df, syn_rules)
    rule_rows = [{"Rule": k,
                  "Type": v.get("rule_type",""),
                  "Violations": v.get("violations","N/A"),
                  "Pass Rate": f"{v.get('pass_rate',0)*100:.1f}%" if v.get("pass_rate") is not None else "N/A",
                  "Description": v.get("description","")}
                 for k, v in check.items()]

    return (
        pd.DataFrame(score_rows),
        make_charts(results),
        report_text,
        pd.DataFrame(rule_rows),
        "\n".join(logs),
        gr.update(visible=False),   # hide review panel
        gr.update(visible=True),    # show results panel
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Agentic Data Quality Evaluator",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    state = gr.State({})

    gr.Markdown(
        "# 🔍 Agentic Data Quality Evaluator\n"
        "**ISO/IEC 5259 series · Powered by Mistral AI · Human-in-the-Loop**\n\n"
        "**Step 1** — Fill in inputs and click *Generate Rules*. "
        "**Step 2** — Review/edit the generated rules, then click *Approve & Evaluate*."
    )

    # ── Input row ────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 API & Models")
            api_key = gr.Textbox(label="Mistral API Key", type="password",
                                 value=os.environ.get("MISTRAL_API_KEY", ""))
            with gr.Row():
                model_large = gr.Dropdown(
                    ["mistral-large-latest", "mistral-medium-latest", "open-mistral-nemo"],
                    value="mistral-large-latest", label="Model (reasoning)")
                model_small = gr.Dropdown(
                    ["mistral-small-latest", "mistral-large-latest", "open-mistral-nemo"],
                    value="mistral-small-latest", label="Model (evaluators)")

            gr.Markdown("### 📂 Dataset")
            dataset_file = gr.File(label="Upload CSV / JSON / Parquet / XLSX",
                file_types=[".csv",".json",".jsonl",".parquet",".xlsx",".xls"])
            domain_desc  = gr.Textbox(label="Domain description", lines=3,
                placeholder="e.g. CAN bus traffic for automotive IDS. Complies with ISO/SAE 21434.")
            dataset_name = gr.Textbox(label="Dataset name (optional)")

            gr.Markdown("### 📋 Metadata")
            with gr.Row():
                source_name   = gr.Textbox(label="Source")
                creation_date = gr.Textbox(label="Creation date (YYYY)")
            with gr.Row():
                license_      = gr.Textbox(label="License", value="CC BY 4.0")
                collection_method = gr.Textbox(label="Collection method")

            gr.Markdown("### ⚙️ Priorities (1=highest, 3=lowest)")
            with gr.Row():
                prio_acc  = gr.Slider(1, 3, value=1, step=1, label="Accuracy")
                prio_comp = gr.Slider(1, 3, value=1, step=1, label="Completeness")
                prio_cons = gr.Slider(1, 3, value=2, step=1, label="Consistency")
            with gr.Row():
                prio_div  = gr.Slider(1, 3, value=2, step=1, label="Diversity")
                prio_cred = gr.Slider(1, 3, value=3, step=1, label="Credibility")
                prio_curr = gr.Slider(1, 3, value=3, step=1, label="Currentness")
            with gr.Row():
                short_thresh = gr.Number(value=5,  label="Short currentness threshold (yrs)")
                long_thresh  = gr.Number(value=15, label="Long currentness threshold (yrs)")

            generate_btn = gr.Button("⚙️ Generate Rules", variant="primary", size="lg")

        # ── Right column: rule review + results ──────────────────────────────
        with gr.Column(scale=2):

            # ── Rule review panel (shown after phase 1) ───────────────────
            with gr.Group(visible=False) as review_panel:
                gr.Markdown("## ✏️ Step 2 — Review & Approve Rules")
                review_info = gr.Markdown()

                gr.Markdown("### Syntactic Rules")
                gr.Markdown(
                    "*Rule types: `range` → `{\"min\":0,\"max\":8}` · "
                    "`enum` → `[\"A\",\"B\"]` · "
                    "`format` → regex · "
                    "`dtype` → `int64` / `object` · "
                    "`length` → `{\"min\":1,\"max\":255}` · "
                    "`not_null` → `true`*"
                )
                syn_editor = gr.Dataframe(
                    headers=["field","rule_type","specification","description","standard_reference"],
                    datatype=["str","str","str","str","str"],
                    interactive=True, wrap=True, label="",
                )

                gr.Markdown("### Semantic Rules")
                gr.Markdown("*Assessed by the LLM evaluator, not enforced programmatically.*")
                sem_editor = gr.Dataframe(
                    headers=["name","rule_type","fields_involved","description","standard_reference"],
                    datatype=["str","str","str","str","str"],
                    interactive=True, wrap=True, label="",
                )

                approve_btn = gr.Button("✅ Approve Rules & Run Evaluation",
                                        variant="primary", size="lg")

            # ── Results panel (shown after phase 2) ───────────────────────
            with gr.Group(visible=False) as results_panel:
                gr.Markdown("## 📊 Results")
                with gr.Tabs():
                    with gr.TabItem("Scores"):
                        score_table  = gr.DataFrame(label="Characteristic Scores")
                        chart_output = gr.Plot()
                    with gr.TabItem("Report"):
                        report_output = gr.Markdown()
                    with gr.TabItem("Rules"):
                        rule_table = gr.DataFrame(label="Syntactic Rule Check")
                    with gr.TabItem("Log"):
                        log_output = gr.Textbox(lines=15, interactive=False)

    # ── Wire up Phase 1 ───────────────────────────────────────────────────────
    generate_btn.click(
        fn=phase1_generate_rules,
        inputs=[api_key, dataset_file, domain_desc, dataset_name,
                source_name, creation_date, license_, collection_method,
                prio_acc, prio_comp, prio_cons, prio_div, prio_cred, prio_curr,
                model_large, model_small, short_thresh, long_thresh],
        outputs=[syn_editor, sem_editor, review_info, state,
                 review_panel, results_panel],
    )

    # ── Wire up Phase 2 ───────────────────────────────────────────────────────
    approve_btn.click(
        fn=phase2_evaluate,
        inputs=[syn_editor, sem_editor, state],
        outputs=[score_table, chart_output, report_output, rule_table, log_output,
                 review_panel, results_panel],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)

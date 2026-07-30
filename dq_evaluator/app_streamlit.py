"""
Streamlit web UI — Agentic Data Quality Evaluator
Human-in-the-loop: rules are shown for review/edit before evaluation runs.

Run:
    streamlit run dq_evaluator/app_streamlit.py
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
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from mistralai import Mistral
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

CHARACTERISTICS = list(QualityCharacteristic)
CHAR_COLORS = {1: "#2196F3", 2: "#FF9800", 3: "#9E9E9E"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Data Quality Evaluator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "phase": 0,          # 0=input, 1=rule_review, 2=results
    "df": None,
    "profile": None,
    "profile_text": None,
    "context": None,
    "subject": None,
    "priorities": None,
    "context_text": None,
    "syn_rules_df": None,  # editable DataFrame of syntactic rules
    "sem_rules_df": None,  # editable DataFrame of semantic rules
    "results": None,
    "report_text": None,
    "logs": [],
    "client": None,
    "metadata": {},
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def log(msg: str):
    st.session_state.logs.append(msg)


def build_context_text(context, subject) -> str:
    return (
        f"Domain: {context.domain}\n"
        f"Purpose: {context.purpose}\n"
        f"Task type: {context.task_type}\n"
        f"Standards: {', '.join(context.standards) or 'N/A'}\n"
        f"Data subjects: {', '.join(subject.entities)}"
    )


def rules_to_syntactic(df_rules: pd.DataFrame) -> list[SyntacticRule]:
    rules = []
    for _, row in df_rules.iterrows():
        if not row.get("field", "").strip():
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


def rules_to_semantic(df_rules: pd.DataFrame) -> list[SemanticRule]:
    rules = []
    for _, row in df_rules.iterrows():
        if not row.get("name", "").strip():
            continue
        fields_raw = row.get("fields_involved", "")
        if isinstance(fields_raw, str):
            fields = [f.strip() for f in fields_raw.split(",") if f.strip()]
        else:
            fields = list(fields_raw) if fields_raw else []
        rules.append(SemanticRule(
            name=str(row["name"]),
            description=str(row.get("description", "")),
            fields_involved=fields,
            rule_type=str(row.get("rule_type", "domain_logic")),
            standard_reference=str(row.get("standard_reference", "")),
        ))
    return rules


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


def make_charts(results):
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "Mistral API Key", type="password",
        value=os.environ.get("MISTRAL_API_KEY", ""),
    )
    st.divider()

    model_large = st.selectbox("Model — reasoning",
        ["mistral-large-latest", "mistral-medium-latest", "open-mistral-nemo"])
    model_small = st.selectbox("Model — evaluators",
        ["mistral-small-latest", "mistral-large-latest", "open-mistral-nemo"])

    st.divider()
    st.subheader("Characteristic Priorities")
    st.caption("1 = highest priority, 3 = lowest")
    priority_overrides = {}
    defaults = {"accuracy": 1, "completeness": 1, "consistency": 2,
                "diversity": 2, "credibility": 3, "currentness": 3}
    for c in CHARACTERISTICS:
        priority_overrides[c] = st.slider(
            c.value.title(), 1, 3, defaults.get(c.value, 2), step=1
        )

    st.divider()
    st.subheader("Currentness Thresholds")
    short_thresh = st.number_input("Short threshold (years)", value=5, min_value=1)
    long_thresh  = st.number_input("Long threshold (years)",  value=15, min_value=1)

    # Phase indicator
    st.divider()
    phases = ["📂 Input", "✏️ Rule Review", "📊 Results"]
    for i, label in enumerate(phases):
        marker = "▶ " if st.session_state.phase == i else "  "
        color  = "#2196F3" if st.session_state.phase == i else "#888"
        st.markdown(f"<span style='color:{color}'>{marker}{label}</span>",
                    unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔍 Agentic Data Quality Evaluator")
st.caption("ISO/IEC 5259 series · Powered by Mistral AI")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 0 — INPUT
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 0:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📂 Dataset")
        uploaded = st.file_uploader("Upload CSV / JSON / Parquet / XLSX",
            type=["csv", "json", "jsonl", "parquet", "xlsx", "xls"])

        if uploaded:
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                readers = {".csv": pd.read_csv, ".xlsx": pd.read_excel,
                           ".xls": pd.read_excel, ".parquet": pd.read_parquet}
                df = readers.get(suffix, lambda p: pd.read_json(p, lines=(suffix == ".jsonl")))(tmp_path)
                st.session_state.df = df
                st.session_state.profile = profile_dataset(df)
                st.session_state.profile_text = profile_to_text(st.session_state.profile)
                st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
                st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Could not load file: {e}")

    with col2:
        st.subheader("🌐 Domain")
        domain_desc = st.text_area("Describe the domain and intended use",
            height=130,
            placeholder="e.g. CAN bus network traffic used to train anomaly detection models for automotive IDS. Must comply with ISO/SAE 21434.")
        dataset_name = st.text_input("Dataset name", placeholder="My Dataset 2024")

        st.subheader("📋 Metadata (optional)")
        mc = st.columns(2)
        source_name      = mc[0].text_input("Source / Organisation")
        creation_date    = mc[1].text_input("Creation date (YYYY)")
        license_         = mc[0].text_input("License", value="CC BY 4.0")
        collection_method = mc[1].text_input("Collection method")

    st.divider()
    can_run = api_key and st.session_state.df is not None and domain_desc.strip()

    if st.button("⚙️ Generate Rules", type="primary", disabled=not can_run):
        st.session_state.logs = []
        client = Mistral(api_key=api_key)
        st.session_state.client = client
        st.session_state.metadata = {
            "source_name": source_name, "creation_date": creation_date,
            "license": license_, "collection_method": collection_method,
        }
        st.session_state.dataset_name = dataset_name or "Dataset"

        with st.spinner("Agent 1: Analyzing domain…"):
            log("Agent 1: Analyzing domain context…")
            context, subject, auto_priorities = analyze_domain(
                client, domain_desc, st.session_state.profile_text, model=model_large
            )
            priorities = {**auto_priorities, **priority_overrides}
            for c in QualityCharacteristic:
                priorities.setdefault(c, 2)
            st.session_state.context  = context
            st.session_state.subject  = subject
            st.session_state.priorities = priorities
            st.session_state.context_text = build_context_text(context, subject)
            st.session_state.model_large = model_large
            st.session_state.model_small = model_small
            log(f"  Domain: {context.domain} | Task: {context.task_type}")

        with st.spinner("Agent 2: Generating rules…"):
            log("Agent 2: Generating syntactic + semantic rules…")
            syn_rules, sem_rules = generate_rules(
                client, context, st.session_state.profile_text, model=model_large
            )
            st.session_state.syn_rules_df = syn_rules_to_df(syn_rules)
            st.session_state.sem_rules_df = sem_rules_to_df(sem_rules)
            log(f"  {len(syn_rules)} syntactic rules, {len(sem_rules)} semantic rules generated")

        st.session_state.phase = 1
        st.rerun()

    if not can_run:
        if not api_key:
            st.warning("Enter your Mistral API key in the sidebar.")
        if st.session_state.df is None:
            st.info("Upload a dataset to continue.")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — RULE REVIEW
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 1:
    context = st.session_state.context

    st.subheader("✏️ Rule Review & Approval")
    st.info(
        f"**Domain:** {context.domain} · **Task:** {context.task_type}  \n"
        f"**Standards detected:** {', '.join(context.standards) or 'N/A'}  \n\n"
        "The agents generated the rules below based on your domain description and dataset profile. "
        "**Review, edit, add, or delete rows before approving.** "
        "Only approved rules will be used in the evaluation."
    )

    # ── Syntactic rules editor ────────────────────────────────────────────────
    st.markdown("### Syntactic Rules")
    st.caption(
        "Rule types: `range` → `{\"min\":0,\"max\":8}` · "
        "`enum` → `[\"A\",\"B\"]` · "
        "`format` → regex string · "
        "`dtype` → `int64` / `float64` / `object` · "
        "`length` → `{\"min\":1,\"max\":255}` · "
        "`not_null` → `true`"
    )

    syn_edited = st.data_editor(
        st.session_state.syn_rules_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "field":              st.column_config.TextColumn("Field", width="small"),
            "rule_type":          st.column_config.SelectboxColumn(
                "Type", options=["range","enum","format","dtype","length","not_null"], width="small"),
            "specification":      st.column_config.TextColumn("Specification (JSON or string)", width="medium"),
            "description":        st.column_config.TextColumn("Description", width="large"),
            "standard_reference": st.column_config.TextColumn("Standard ref.", width="medium"),
        },
        key="syn_editor",
    )

    st.divider()

    # ── Semantic rules editor ─────────────────────────────────────────────────
    st.markdown("### Semantic Rules")
    st.caption("These express domain-logic constraints. They are assessed by the LLM evaluator, not programmatically enforced.")

    sem_edited = st.data_editor(
        st.session_state.sem_rules_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "name":               st.column_config.TextColumn("Rule name", width="small"),
            "rule_type":          st.column_config.SelectboxColumn(
                "Type", options=["cross_field","domain_logic","temporal","referential"], width="small"),
            "fields_involved":    st.column_config.TextColumn("Fields (comma-separated)", width="medium"),
            "description":        st.column_config.TextColumn("Description", width="large"),
            "standard_reference": st.column_config.TextColumn("Standard ref.", width="medium"),
        },
        key="sem_editor",
    )

    st.divider()
    col_back, col_approve = st.columns([1, 3])

    if col_back.button("← Back to Input"):
        st.session_state.phase = 0
        st.rerun()

    if col_approve.button("✅ Approve Rules & Run Evaluation", type="primary"):
        # Persist the edited tables back
        st.session_state.syn_rules_df = syn_edited
        st.session_state.sem_rules_df = sem_edited

        syn_rules = rules_to_syntactic(syn_edited)
        sem_rules = rules_to_semantic(sem_edited)
        log(f"Human approved {len(syn_rules)} syntactic + {len(sem_rules)} semantic rules.")

        df       = st.session_state.df
        profile  = st.session_state.profile
        profile_text  = st.session_state.profile_text
        context_text  = st.session_state.context_text
        priorities    = st.session_state.priorities
        metadata      = st.session_state.metadata
        client        = st.session_state.client
        model_large   = st.session_state.model_large
        model_small   = st.session_state.model_small

        results = []
        progress = st.progress(0, "Starting evaluators…")

        try:
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
                    short_threshold_years=int(short_thresh),
                    long_threshold_years=int(long_thresh), model=model_small)),
            ], start=0):
                pct = int(10 + i * 13)   # 10, 23, 36, 49, 62, 75
                progress.progress(pct, f"Agent {i+3}: {label}…")
                log(f"Agent {i+3}: Evaluating {label}…")
                results.append(fn(client, **kwargs))

            progress.progress(92, "Agent 9: Generating report…")
            log("Agent 9: Generating report…")
            report = generate_report(
                client, st.session_state.get("dataset_name", "Dataset"),
                st.session_state.context, st.session_state.subject,
                results, model=model_large
            )
            st.session_state.results = results
            st.session_state.report_text = format_report_text(report)
            progress.progress(100, "Done ✅")
            log("Evaluation complete.")
            st.session_state.phase = 2
            st.rerun()

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            raise


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — RESULTS
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 2:
    results     = st.session_state.results
    report_text = st.session_state.report_text
    context     = st.session_state.context

    top_col1, top_col2 = st.columns([4, 1])
    top_col1.subheader(f"📊 Results — {context.domain}")
    if top_col2.button("🔄 Start Over"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    result_tabs = st.tabs(["📊 Scores", "📄 Report", "🔎 Rules Used", "📝 Log"])

    # Scores tab
    with result_tabs[0]:
        cols = st.columns(len(results))
        for col, r in zip(cols, sorted(results, key=lambda x: x.priority)):
            score_str = f"{r.overall_score:.0f}%" if r.overall_score is not None else "N/A"
            col.metric(r.characteristic.value.title(), score_str,
                       help=f"Priority {r.priority} · {r.summary}")

        st.pyplot(make_charts(results))

        st.divider()
        st.subheader("Dimension Details")
        for r in sorted(results, key=lambda x: x.priority):
            label = (f"**{r.characteristic.value.title()}** — "
                     f"{r.overall_score:.0f}% · Priority {r.priority}"
                     if r.overall_score else
                     f"**{r.characteristic.value.title()}** · Priority {r.priority}")
            with st.expander(label):
                for d in r.dimensions:
                    icon = "✅" if d.passed else ("❌" if d.passed is False else "ℹ️")
                    score_disp = f"{d.score:.0f}%" if isinstance(d.score, float) else str(d.score)
                    st.markdown(f"{icon} **{d.dimension}** — {score_disp}")
                    st.caption(d.explanation)

    # Report tab
    with result_tabs[1]:
        st.download_button("⬇️ Download Report (.md)", data=report_text,
                           file_name="dq_report.md", mime="text/markdown")
        st.markdown(report_text)

    # Rules used tab
    with result_tabs[2]:
        st.subheader("Syntactic Rules (as approved)")
        df_syn = st.session_state.syn_rules_df
        if df_syn is not None and not df_syn.empty:
            check = check_syntactic_rules(
                st.session_state.df, rules_to_syntactic(df_syn)
            )
            rows = []
            for key, res in check.items():
                rows.append({
                    "Rule": key,
                    "Type": res.get("rule_type", ""),
                    "Violations": res.get("violations", "N/A"),
                    "Pass Rate": f"{res.get('pass_rate', 0)*100:.1f}%"
                                 if res.get("pass_rate") is not None else "N/A",
                    "Description": res.get("description", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Semantic Rules (as approved)")
        st.dataframe(st.session_state.sem_rules_df, use_container_width=True)

    # Log tab
    with result_tabs[3]:
        st.code("\n".join(st.session_state.logs), language=None)

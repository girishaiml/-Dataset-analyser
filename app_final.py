try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import matplotlib
import pandas as pd
import streamlit as st
from fpdf import FPDF
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from datetime import datetime
import textwrap
import json
import base64
import os
import io
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use("Agg")


# -- PAGE CONFIG ---------------------------------------------------------------
st.set_page_config(
    page_title="Dataset Analyser Agent",
    page_icon="",
    layout="wide"
)

# -- CUSTOM CSS ----------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .agent-step {
        background: white;
        border-left: 4px solid #1D4A2A;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .agent-step-title {
        font-weight: 700;
        color: #1D4A2A;
        font-size: 14px;
    }
    .agent-step-body {
        color: #444;
        font-size: 13px;
        margin-top: 4px;
    }
    .insight-box {
        background: #f0f7f2;
        border: 1px solid #c8d8cc;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    h1 { color: #1D4A2A !important; }
    h2 { color: #1D4A2A !important; }
    h3 { color: #2A6040 !important; }
</style>
""", unsafe_allow_html=True)

# -- AGENT STATE ---------------------------------------------------------------


class AgentState(TypedDict):
    df: object
    filename: str
    api_key: str
    shape_info: str
    column_info: str
    stats_summary: str
    missing_info: str
    pattern_insights: str
    executive_summary: str
    recommendations: str
    charts: list
    report_ready: bool
    log: list

# -- TOOL FUNCTIONS ------------------------------------------------------------


def tool_inspect_dataset(state: AgentState) -> AgentState:
    """Step 1 - Understand what the dataset is"""
    df = state["df"]
    log = state.get("log", [])

    shape_info = f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}"

    col_info_lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_null = df[col].isnull().sum()
        sample = df[col].dropna().head(3).tolist()
        col_info_lines.append(
            f"- {col} | type={dtype} | unique={n_unique} | nulls={n_null} | sample={sample}"
        )
    column_info = "\n".join(col_info_lines)

    log.append({
        "step": "Step 1 - Dataset Inspection",
        "detail": f"Detected {df.shape[0]:,} rows and {df.shape[1]} columns. Identified column types and null values."
    })

    return {**state, "shape_info": shape_info, "column_info": column_info, "log": log}


def tool_statistical_analysis(state: AgentState) -> AgentState:
    """Step 2 - Run statistical analysis"""
    df = state["df"]
    log = state.get("log", [])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()

    lines = []

    if numeric_cols:
        lines.append("=== NUMERIC COLUMNS ===")
        desc = df[numeric_cols].describe().round(3)
        lines.append(desc.to_string())

        lines.append("\n=== CORRELATIONS (top pairs) ===")
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    pairs.append(
                        (corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3)))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for a, b, v in pairs[:5]:
                lines.append(f"  {a} vs {b}: {v}")

    if cat_cols:
        lines.append("\n=== CATEGORICAL COLUMNS ===")
        for col in cat_cols[:5]:
            vc = df[col].value_counts().head(5)
            lines.append(f"\n{col} (top 5):\n{vc.to_string()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_info = missing.to_string() if len(
        missing) > 0 else "No missing values found."

    stats_summary = "\n".join(lines)

    log.append({
        "step": "Step 2 - Statistical Analysis",
        "detail": f"Analysed {len(numeric_cols)} numeric and {len(cat_cols)} categorical columns. Computed descriptive stats and correlations."
    })

    return {**state, "stats_summary": stats_summary, "missing_info": missing_info, "log": log}


def tool_generate_charts(state: AgentState) -> AgentState:
    """Step 3 - Generate visualisations"""
    df = state["df"]
    log = state.get("log", [])
    charts = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#1D4A2A", "#B8842A", "#2A6040", "#D4A855", "#4A7A5A"]

    # Chart 1 - Distribution of first numeric column
    if numeric_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        col = numeric_cols[0]
        df[col].dropna().hist(ax=ax, bins=30, color=colors[0],
                              edgecolor="white", alpha=0.85)
        ax.set_title(
            f"Distribution of {col}", fontsize=13, fontweight="bold", color="#1D4A2A")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        charts.append({"title": f"Distribution of {col}", "img": buf.read()})
        plt.close(fig)

    # Chart 2 - Correlation heatmap
    if len(numeric_cols) >= 2:
        cols_to_use = numeric_cols[:8]
        fig, ax = plt.subplots(figsize=(7, 5))
        corr_matrix = df[cols_to_use].corr()
        mask = corr_matrix.isnull()
        sns.heatmap(
            corr_matrix, ax=ax, annot=True, fmt=".2f",
            cmap="YlOrRd", linewidths=0.5,
            annot_kws={"size": 9}, mask=mask
        )
        ax.set_title("Correlation Heatmap", fontsize=13,
                     fontweight="bold", color="#1D4A2A")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        charts.append({"title": "Correlation Heatmap", "img": buf.read()})
        plt.close(fig)

    # Chart 3 - Top categorical column bar chart
    if cat_cols:
        col = cat_cols[0]
        vc = df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        vc.plot(kind="bar", ax=ax,
                color=colors[1], edgecolor="white", alpha=0.9)
        ax.set_title(f"Top Values - {col}", fontsize=13,
                     fontweight="bold", color="#1D4A2A")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        charts.append({"title": f"Top Values - {col}", "img": buf.read()})
        plt.close(fig)

    # Chart 4 - Missing values bar
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(7, 3))
        missing.plot(kind="bar", ax=ax, color="#B03A2E", edgecolor="white")
        ax.set_title("Missing Values per Column", fontsize=13,
                     fontweight="bold", color="#1D4A2A")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        charts.append(
            {"title": "Missing Values per Column", "img": buf.read()})
        plt.close(fig)

    log.append({
        "step": "Step 3 - Chart Generation",
        "detail": f"Generated {len(charts)} visualisation(s): distributions, correlations, categorical breakdowns."
    })

    return {**state, "charts": charts, "log": log}


def tool_llm_analysis(state: AgentState) -> AgentState:
    """Step 4 - LLM generates insights, summary, recommendations"""
    log = state.get("log", [])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=state["api_key"],
        temperature=0.3
    )

    prompt = f"""You are an expert data analyst. Analyse the following dataset information and provide:

DATASET: {state['filename']}
SHAPE: {state['shape_info']}

COLUMN INFO:
{state['column_info']}

STATISTICAL SUMMARY:
{state['stats_summary']}

MISSING VALUES:
{state['missing_info']}

Respond ONLY with a valid JSON object (no markdown, no backticks) with exactly these keys:
{{
  "dataset_description": "2-3 sentences describing what this dataset appears to be about",
  "key_patterns": ["pattern 1", "pattern 2", "pattern 3", "pattern 4", "pattern 5"],
  "executive_summary": "A 4-5 sentence executive summary of the most important findings",
  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3", "recommendation 4"]
}}"""

    response = llm.invoke([
        SystemMessage(
            content="You are an expert data analyst. Always respond with valid JSON only."),
        HumanMessage(content=prompt)
    ])

    raw = response.content.strip()
    # Clean any markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    pattern_insights = "\n".join(
        [f"- {p}" for p in result.get("key_patterns", [])])
    executive_summary = result.get("executive_summary", "")
    recommendations = "\n".join(
        [f"- {r}" for r in result.get("recommendations", [])])
    dataset_description = result.get("dataset_description", "")

    log.append({
        "step": "Step 4 - AI Analysis",
        "detail": f"LLM identified dataset as: {dataset_description[:100]}..."
    })

    return {
        **state,
        "pattern_insights": pattern_insights,
        "executive_summary": executive_summary,
        "recommendations": recommendations,
        "log": log
    }


def tool_generate_pdf(state: AgentState) -> AgentState:
    """Step 5 - Build the PDF report"""
    log = state.get("log", [])

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # -- Header --
    pdf.set_fill_color(29, 74, 42)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(15, 8)
    pdf.cell(180, 10, "DATASET ANALYSIS REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_xy(15, 20)
    pdf.cell(
        180, 8, f"File: {state['filename']}   |   Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", ln=True, align="C")
    pdf.set_xy(15, 30)
    pdf.cell(180, 6, state["shape_info"], ln=True, align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.ln(12)

    def section_header(title):
        pdf.set_fill_color(29, 74, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    def body_text(text, size=10):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(50, 50, 50)
        for line in text.split("\n"):
            wrapped = textwrap.wrap(line, width=95) if line.strip() else [""]
            for wline in wrapped:
                pdf.cell(0, 6, wline, ln=True)
        pdf.ln(2)

    # -- Executive Summary --
    section_header("EXECUTIVE SUMMARY")
    body_text(state["executive_summary"])

    # -- Key Patterns --
    section_header("KEY PATTERNS & INSIGHTS")
    body_text(state["pattern_insights"])

    # -- Dataset Overview --
    section_header("DATASET OVERVIEW")
    body_text(state["shape_info"])
    body_text(state["column_info"])

    # -- Missing Values --
    section_header("DATA QUALITY - MISSING VALUES")
    body_text(state["missing_info"])

    # -- Statistical Summary --
    section_header("STATISTICAL SUMMARY")
    pdf.set_font("Courier", "", 8)
    pdf.set_text_color(30, 30, 30)
    for line in state["stats_summary"].split("\n")[:60]:
        safe = line.encode("latin-1", "replace").decode("latin-1")
        pdf.cell(0, 5, safe, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # -- Charts --
    if state["charts"]:
        section_header("VISUALISATIONS")
        for i, chart in enumerate(state["charts"]):
            if i % 2 == 0 and i > 0:
                pdf.add_page()
            img_buf = io.BytesIO(chart["img"])
            tmp_path = f"/tmp/chart_{i}.png"
            with open(tmp_path, "wb") as f:
                f.write(chart["img"])
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(29, 74, 42)
            pdf.cell(0, 7, chart["title"], ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.image(tmp_path, w=170)
            pdf.ln(5)

    # -- Recommendations --
    pdf.add_page()
    section_header("RECOMMENDATIONS")
    body_text(state["recommendations"])

    # -- Footer --
    pdf.set_fill_color(29, 74, 42)
    pdf.rect(0, 285, 210, 15, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_xy(10, 288)
    pdf.cell(190, 6, "Generated by Dataset Analyser Agent  |  Built with LangGraph + OpenAI + Streamlit", align="C")

    pdf_bytes = pdf.output()

    log.append({
        "step": "Step 5 - PDF Report Generated",
        "detail": f"Report compiled with {len(state['charts'])} charts, statistical analysis, and AI insights."
    })

    return {**state, "pdf_bytes": pdf_bytes, "report_ready": True, "log": log}


# -- BUILD LANGGRAPH -----------------------------------------------------------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("inspect",    tool_inspect_dataset)
    graph.add_node("stats",      tool_statistical_analysis)
    graph.add_node("charts",     tool_generate_charts)
    graph.add_node("llm",        tool_llm_analysis)
    graph.add_node("pdf",        tool_generate_pdf)

    graph.set_entry_point("inspect")
    graph.add_edge("inspect", "stats")
    graph.add_edge("stats",   "charts")
    graph.add_edge("charts",  "llm")
    graph.add_edge("llm",     "pdf")
    graph.add_edge("pdf",     END)

    return graph.compile()


# -- STREAMLIT UI --------------------------------------------------------------

st.title(" Dataset Analyser Agent")
st.markdown("*Upload any CSV. The agent analyses it, finds patterns, and generates a full PDF report - automatically.*")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    # Load from .env or Streamlit secrets first
    env_key = os.getenv("OPENAI_API_KEY", "")
    try:
        env_key = env_key or st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass

    if env_key:
        api_key = env_key
        st.success("API key loaded from environment.")
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-... (or set in .env file)",
            help="Add OPENAI_API_KEY to your .env file to skip this step."
        )

with col2:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file and api_key:
    df = pd.read_csv(uploaded_file)

    st.markdown("### 📂 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", f"{df.shape[0]:,}")
    col_b.metric("Columns", df.shape[1])
    col_c.metric("Numeric Cols", len(
        df.select_dtypes(include="number").columns))
    col_d.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    if st.button("🚀 Run Agent", type="primary", use_container_width=True):

        st.markdown("### 🔄 Agent Progress")
        progress_container = st.container()
        results_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Build and run agent
            agent = build_agent()

            initial_state = AgentState(
                df=df,
                filename=uploaded_file.name,
                api_key=api_key,
                shape_info="",
                column_info="",
                stats_summary="",
                missing_info="",
                pattern_insights="",
                executive_summary="",
                recommendations="",
                charts=[],
                report_ready=False,
                log=[]
            )

            steps = ["inspect", "stats", "charts", "llm", "pdf"]
            step_labels = [
                "Inspecting dataset...",
                "Running statistical analysis...",
                "Generating charts...",
                "AI is analysing patterns...",
                "Building PDF report..."
            ]

            final_state = None
            step_placeholders = []

            for i, (step, label) in enumerate(zip(steps, step_labels)):
                status_text.markdown(f"**{label}**")
                progress_bar.progress((i + 1) / len(steps))

            # Run full graph
            with st.spinner("Agent is working..."):
                final_state = agent.invoke(initial_state)

            progress_bar.progress(1.0)
            status_text.markdown("**✅ Analysis complete!**")

            # Show agent log
            st.markdown("#### Agent Steps Completed")
            for entry in final_state.get("log", []):
                st.markdown(f"""
                <div class="agent-step">
                    <div class="agent-step-title">{entry['step']}</div>
                    <div class="agent-step-body">{entry['detail']}</div>
                </div>
                """, unsafe_allow_html=True)

        with results_container:
            st.markdown("---")
            st.markdown("### 📊 Results")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["Executive Summary", "Key Patterns", "Charts", "Recommendations"])

            with tab1:
                st.markdown(f"""
                <div class="insight-box">
                {final_state['executive_summary']}
                </div>
                """, unsafe_allow_html=True)

            with tab2:
                for line in final_state["pattern_insights"].split("\n"):
                    if line.strip():
                        st.markdown(f"- {line.replace('-', '').strip()}")

            with tab3:
                chart_cols = st.columns(2)
                for i, chart in enumerate(final_state.get("charts", [])):
                    with chart_cols[i % 2]:
                        st.image(chart["img"], caption=chart["title"],
                                 use_container_width=True)

            with tab4:
                for line in final_state["recommendations"].split("\n"):
                    if line.strip():
                        st.success(line.replace("-", "").strip())

            # Download button
            st.markdown("---")
            pdf_bytes = final_state.get("pdf_bytes")
            if pdf_bytes:
                st.download_button(
                    label="📥 Download Full PDF Report",
                    data=bytes(pdf_bytes),
                    file_name=f"analysis_{uploaded_file.name.replace('.csv', '')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )

elif uploaded_file and not api_key:
    st.warning("Please enter your OpenAI API key to run the agent.")
elif api_key and not uploaded_file:
    st.info("Please upload a CSV file to get started.")
else:
    st.markdown("""
    ### How it works
    1. **Enter your OpenAI API key** - used only for this session, never stored
    2. **Upload any CSV file** - sales data, survey results, research data, anything
    3. **Click Run Agent** - the agent automatically:
       - Inspects the dataset structure
       - Runs statistical analysis
       - Generates visualisations
       - Uses AI to identify patterns and insights
       - Compiles everything into a downloadable PDF report

    ### Agent Architecture
    ```
    CSV Upload -> Inspect -> Statistics -> Charts -> AI Analysis -> PDF Report
    ```
    Built with **LangGraph** - **OpenAI GPT-4o-mini** - **Pandas** - **Streamlit**
    """)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.scorer import load_scores, run_full_analysis
from src.visualizer import (
    plot_overall_scores,
    plot_dimension_radar,
    plot_category_performance,
    plot_consistency
)

st.set_page_config(
    page_title="GenAI Benchmark",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp { background-color: #0F172A; }
        h1, h2, h3 { color: #F1F5F9; }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding: 24px 0'>
        <h1>🤖 GenAI Model Benchmark</h1>
        <p style='color:#94A3B8'>
            20 prompts · 6 models · 4 dimensions · 1 winner
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ── Load Data ──────────────────────────────────────────────────
try:
    df = load_scores()
    analysis = run_full_analysis(df)
except FileNotFoundError:
    st.warning("No scores data found. Add data/scores/scores.csv to get started.")
    st.stop()

# ── Overall Scores ─────────────────────────────────────────────
st.markdown("## 🏆 Overall Rankings")
col1, col2 = st.columns([2, 1])

with col1:
    fig = plot_overall_scores(analysis["model_summary"])
    st.pyplot(fig)

with col2:
    st.markdown("### Leaderboard")
    summary = analysis["model_summary"][["total", "rank"]].reset_index()
    summary.columns = ["Model", "Avg Score", "Rank"]
    summary = summary.sort_values("Rank")
    st.dataframe(summary, hide_index=True)

st.divider()

# ── Dimension Heatmap ──────────────────────────────────────────
st.markdown("## 📊 Score Breakdown by Dimension")
fig2 = plot_dimension_radar(analysis["model_summary"])
st.pyplot(fig2)

st.divider()

# ── Category Performance ───────────────────────────────────────
st.markdown("## 📂 Performance by Prompt Category")
fig3 = plot_category_performance(analysis["category_performance"])
st.pyplot(fig3)

st.divider()

# ── Dimension Leaders ──────────────────────────────────────────
st.markdown("## 🥇 Category Leaders")
leaders = analysis["dimension_leaders"]
cols = st.columns(4)
dimensions = ["accuracy", "reasoning", "tone", "instruction_following"]
icons = ["🎯", "🧠", "🗣", "📋"]

for col, dim, icon in zip(cols, dimensions, icons):
    with col:
        st.metric(
            label=f"{icon} {dim.replace('_', ' ').title()}",
            value=leaders[dim]["model"].title(),
            delta=f"avg {leaders[dim]['score']}/5"
        )

st.divider()

# ── Consistency ────────────────────────────────────────────────
st.markdown("## 📈 Model Consistency")
fig4 = plot_consistency(analysis["consistency"])
st.pyplot(fig4)

st.divider()

# ── Raw Data Explorer ──────────────────────────────────────────
with st.expander("🔍 Explore Raw Scores"):
    model_filter = st.multiselect(
        "Filter by model",
        options=df["model"].unique(),
        default=df["model"].unique()
    )
    filtered = df[df["model"].isin(model_filter)]
    st.dataframe(filtered, hide_index=True)
# 🤖 GenAI Model Benchmark

A structured evaluation framework that stress-tests six leading AI models across 20 carefully designed prompts, scored across four dimensions, and visualised through an interactive Streamlit dashboard.

---

## 📌 Overview

This project benchmarks **ChatGPT, Claude, Gemini, Grok, Kimi, and Qwen** by running them through a standardised prompt suite that covers seven capability categories. Each response is scored across four dimensions, and the results are surfaced through a live dashboard with rankings, radar charts, category breakdowns, and consistency analysis.

---

## 📁 Project Structure

```
GenAI Benchmark/
├── main.py                  # Streamlit dashboard entry point
├── src/
│   ├── scorer.py            # Data loading, normalisation, and all analysis logic
│   └── visualizer.py        # Chart and plot generation (matplotlib)
├── data/
│   └── scores/
│       └── scores.csv       # Final scored dataset (120 rows — 20 prompts × 6 models)
└── README.md
```

> **Note:** A `notebooks/` folder is intentionally excluded. All scoring logic lives in `src/scorer.py`.

---

## 🧪 Prompt Suite

20 prompts across 7 categories, designed to stress-test distinct capabilities:

| Category | Prompts | What it tests |
|---|---|---|
| Factual Accuracy | P1 – P3 | Depth, correctness, no hallucination |
| Reasoning & Logic | P4 – P6 | Deductive reasoning, problem solving |
| Coding | P7 – P9 | Code correctness, complexity analysis, API design |
| Creative Writing | P10 – P12 | Originality, craft, constraint adherence |
| Nuance & Ethics | P13 – P15 | Multi-perspective thinking, intellectual honesty |
| Instruction Following | P16 – P18 | Strict adherence to format and constraint rules |
| Self Awareness | P19 – P20 | Honest articulation of limitations |

---

## 📊 Scoring Dimensions

Each response is scored **out of 10** across four dimensions:

| Dimension | What it measures |
|---|---|
| **Accuracy** | Factual correctness, absence of hallucination |
| **Reasoning** | Logical depth and quality of explanation |
| **Tone** | Clarity, writing quality, appropriateness of voice |
| **Instruction** | Adherence to all explicit constraints in the prompt |

A **total score** is computed as the mean of all four dimensions.

---

## 🏆 Results Summary

| Rank | Model | Avg Score |
|---|---|---|
| 🥇 1 | Kimi | 9.50 |
| 🥈 2 | Claude | 9.35 |
| 🥉 3 | GPT | 7.55 |
| 4 | Gemini | 7.45 |
| 5 | Qwen | 7.45 |
| 6 | Grok | 7.40 |

**Notable findings:**
- **Kimi** led on factual depth — named the Barbara syllogism by form, specified N1-methylpseudouridine in the LNP composition, and fabricated a genuinely useful buggy code example when the prompt placeholder was empty.
- **Claude** dominated creative writing and nuanced ethics but missed the numbered list constraint in P17 — same as GPT, Grok, and Kimi.
- **Gemini** suffered a critical swap — it answered P14 (AI regulation) with the trolley problem response and vice versa, collapsing its Nuance & Ethics score.
- **Qwen** had a garbled P4 response where two prompt answers merged mid-sentence, tanking its Reasoning score.
- **Only Gemini** correctly numbered the business ideas list in P17 — all other models missed that constraint.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/genai-benchmark.git
cd genai-benchmark
```

### 2. Install dependencies

```bash
pip install streamlit pandas matplotlib
```

### 3. Run the dashboard

```bash
streamlit run main.py
```

> Make sure you run this from the project root (`GenAI Benchmark/`), not from inside a subfolder. The path resolution in `scorer.py` is anchored to `__file__`, so the CSV will be found automatically regardless.

---

## 🔧 Key Implementation Notes

**`src/scorer.py`** handles all data logic:
- Loads `scores.csv` using an absolute path anchored to `__file__` — no working directory dependency
- Normalises column names to lowercase on load
- Renames `Instruction` → `instruction_following` to match internal dimension keys
- Computes `total` (mean of 4 dimensions) and `category` (mapped from `Prompt_ID`) at load time since neither is stored in the CSV

**`main.py`** is purely presentational — it calls `load_scores()` and `run_full_analysis()` from `scorer.py`, then passes results to `visualizer.py` for rendering.

---

## 📋 Data Format

`data/scores/scores.csv` — 120 rows, 6 columns:

```
Prompt_ID, Model, Accuracy, Reasoning, Tone, Instruction
P1, Claude, 9, 9, 9, 9
P1, GPT, 7, 7, 7, 8
...
P20, Qwen, 8, 8, 8, 8
```

---

## 🛠 Tech Stack

- **Python 3.14**
- **Streamlit** — dashboard framework
- **Pandas** — data loading and analysis
- **Matplotlib** — charts and visualisations

---

## 👤 Author

**Sheikh Syeed**
B.Sc. Computer Science & Engineering — AIUB, Dhaka

import pandas as pd
import os

MODELS = ["Claude", "GPT", "Gemini", "Grok", "Qwen", "Kimi"]

DIMENSIONS = ["accuracy", "reasoning", "tone", "instruction_following"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_scores(scores_path: str = None) -> pd.DataFrame:
    """Load and normalise the scored dataset."""
    if scores_path is None:
        scores_path = os.path.join(BASE_DIR, "..", "data", "scores", "scores.csv")

    df = pd.read_csv(scores_path)

    # Normalise column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # Rename to match what the rest of the code expects
    df = df.rename(columns={
        "prompt_id":   "prompt_id",
        "model":       "model",
        "accuracy":    "accuracy",
        "reasoning":   "reasoning",
        "tone":        "tone",
        "instruction": "instruction_following"  # CSV has 'Instruction', code expects 'instruction_following'
    })

    # Compute total (average across 4 dimensions) — not in the CSV
    df["total"] = df[DIMENSIONS].mean(axis=1).round(2)

    # Map prompt IDs to categories — not in the CSV
    category_map = {
        "P1":  "FACTUAL ACCURACY",
        "P2":  "FACTUAL ACCURACY",
        "P3":  "FACTUAL ACCURACY",
        "P4":  "REASONING & LOGIC",
        "P5":  "REASONING & LOGIC",
        "P6":  "REASONING & LOGIC",
        "P7":  "CODING",
        "P8":  "CODING",
        "P9":  "CODING",
        "P10": "CREATIVE WRITING",
        "P11": "CREATIVE WRITING",
        "P12": "CREATIVE WRITING",
        "P13": "NUANCE & ETHICS",
        "P14": "NUANCE & ETHICS",
        "P15": "NUANCE & ETHICS",
        "P16": "INSTRUCTION FOLLOWING",
        "P17": "INSTRUCTION FOLLOWING",
        "P18": "INSTRUCTION FOLLOWING",
        "P19": "SELF AWARENESS",
        "P20": "SELF AWARENESS",
    }
    df["category"] = df["prompt_id"].map(category_map)

    return df


def compute_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average scores per model across
    all prompts and dimensions.
    """
    summary = df.groupby("model")[DIMENSIONS + ["total"]].mean().round(2)
    summary["rank"] = summary["total"].rank(ascending=False).astype(int)
    summary = summary.sort_values("total", ascending=False)
    return summary


def compute_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average total score per model
    broken down by prompt category.
    """
    return df.groupby(["model", "category"])["total"].mean().round(2).unstack()


def compute_dimension_leaders(df: pd.DataFrame) -> dict:
    """
    For each scoring dimension, find which
    model leads on average.
    """
    leaders = {}
    for dim in DIMENSIONS:
        leader = df.groupby("model")[dim].mean().idxmax()
        score  = df.groupby("model")[dim].mean().max()
        leaders[dim] = {"model": leader, "score": round(score, 2)}
    return leaders


def compute_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure consistency — low std deviation
    means the model performs reliably.
    """
    return df.groupby("model")["total"].std().round(2).sort_values()


def get_best_response_per_prompt(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each prompt, identify which model
    scored highest.
    """
    return df.loc[df.groupby("prompt_id")["total"].idxmax()][
        ["prompt_id", "model", "total"]
    ].reset_index(drop=True)


def run_full_analysis(df: pd.DataFrame) -> dict:
    """
    Master function — runs all analyses
    and returns unified report.
    """
    return {
        "model_summary":        compute_model_summary(df),
        "category_performance": compute_category_performance(df),
        "dimension_leaders":    compute_dimension_leaders(df),
        "consistency":          compute_consistency(df),
        "best_per_prompt":      get_best_response_per_prompt(df)
    }
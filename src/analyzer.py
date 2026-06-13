# src/analyzer.py

import pandas as pd


def compute_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each prompt, award 1 point to the
    highest scoring model. Count wins per model.
    """
    winners = df.loc[df.groupby("prompt_id")["total"].idxmax()]
    win_rate = winners["model"].value_counts().reset_index()
    win_rate.columns = ["model", "wins"]
    win_rate["win_rate_%"] = (win_rate["wins"] / 20 * 100).round(1)
    return win_rate


def compute_head_to_head(
    df: pd.DataFrame,
    model_a: str,
    model_b: str
) -> dict:
    """
    Direct comparison between two models
    across all prompts.
    """
    a = df[df["model"] == model_a].set_index("prompt_id")["total"]
    b = df[df["model"] == model_b].set_index("prompt_id")["total"]

    compared = pd.DataFrame({"a": a, "b": b}).dropna()

    a_wins = (compared["a"] > compared["b"]).sum()
    b_wins = (compared["b"] > compared["a"]).sum()
    ties   = (compared["a"] == compared["b"]).sum()

    return {
        "model_a":  model_a,
        "model_b":  model_b,
        "a_wins":   int(a_wins),
        "b_wins":   int(b_wins),
        "ties":     int(ties),
        "winner":   model_a if a_wins > b_wins else model_b if b_wins > a_wins else "tie"
    }


def compute_prompt_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify which prompts all models
    struggled with — low average = hard prompt.
    """
    difficulty = df.groupby("prompt_id")["total"].mean().round(2)
    difficulty = difficulty.sort_values()
    difficulty.name = "avg_score"
    return difficulty.reset_index()


def compute_category_winner(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each prompt category, find which
    model dominates.
    """
    return df.groupby(
        ["category", "model"]
    )["total"].mean().round(2).reset_index().sort_values(
        ["category", "total"], ascending=[True, False]
    ).groupby("category").first().reset_index()
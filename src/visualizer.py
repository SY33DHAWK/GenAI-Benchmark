import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="darkgrid")
COLORS = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#22D3EE", "#F97316"]


def plot_overall_scores(summary: pd.DataFrame, save_path: str = None):
    """Bar chart of overall average scores per model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        summary.index,
        summary["total"],
        color=COLORS[:len(summary)],
        edgecolor="white",
        linewidth=0.5
    )
    ax.set_title("Overall Average Score by Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Total Score (max 20)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 20)

    for bar, val in zip(bars, summary["total"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                str(val), ha="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_dimension_radar(summary: pd.DataFrame, save_path: str = None):
    """Heatmap of scores across all dimensions per model."""
    dims = ["accuracy", "reasoning", "tone", "instruction_following"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        summary[dims],
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        vmin=1, vmax=5
    )
    ax.set_title("Score Heatmap by Model and Dimension", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_category_performance(cat_df: pd.DataFrame, save_path: str = None):
    """Grouped bar chart by category."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cat_df.T.plot(kind="bar", ax=ax, color=COLORS[:len(cat_df)], edgecolor="white")
    ax.set_title("Model Performance by Prompt Category", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Score")
    ax.set_xlabel("Category")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_consistency(consistency: pd.Series, save_path: str = None):
    """Bar chart of score consistency (lower std = more consistent)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        consistency.index,
        consistency.values,
        color=COLORS[:len(consistency)],
        edgecolor="white"
    )
    ax.set_title("Model Consistency (Lower = More Consistent)", 
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Std Deviation of Total Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
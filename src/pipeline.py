"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.baselines import build_category_strategy_scores, build_strategy_scores
from src.collaborative import build_interaction_matrix, build_user_similarity
from src.demo_data import build_posts_base_from_parquet_or_synthetic
from src.evaluation import evaluate_all_models, prepare_evaluation_frames
from src.preprocess import build_posts_base, load_influencers_df, parse_metadata_directory


@dataclass
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    target_posts: int = 20_000
    k: int = 5
    seed: int = 42
    posts_parquet: Path | None = None
    extracted_metadata_dir: Path | None = None
    synthetic: bool = False
    synthetic_influencers: int = 120


def _ensure_dirs(config: PipelineConfig) -> dict[str, Path]:
    processed = config.output_dir / "processed"
    results = config.output_dir / "results"
    figures = config.output_dir / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {"processed": processed, "results": results, "figures": figures}


def load_posts_base(config: PipelineConfig) -> pd.DataFrame:
    influencers_path = config.data_dir / "influencers.txt"

    if config.posts_parquet:
        return build_posts_base_from_parquet_or_synthetic(
            influencers_path,
            posts_parquet=config.posts_parquet,
        )

    if config.extracted_metadata_dir:
        influencers_df = load_influencers_df(influencers_path)
        posts_df = parse_metadata_directory(
            config.extracted_metadata_dir,
            output_dir=config.output_dir / "processed" / "parts",
            max_files=config.target_posts,
        )
        return build_posts_base(posts_df, influencers_df)

    if config.synthetic:
        return build_posts_base_from_parquet_or_synthetic(
            influencers_path,
            synthetic=True,
            num_influencers=config.synthetic_influencers,
            target_posts=config.target_posts,
            seed=config.seed,
        )

    raise ValueError("No data source configured.")


def save_model_tables(posts_base_df: pd.DataFrame, processed_dir: Path, label: str) -> dict[str, Path]:
    paths = {
        "posts_base": processed_dir / f"posts_base_{label}.parquet",
        "strategy_scores": processed_dir / f"strategy_scores_{label}.parquet",
        "category_strategy_scores": processed_dir / f"category_strategy_scores_{label}.parquet",
        "interaction_matrix": processed_dir / f"interaction_matrix_{label}.parquet",
        "user_similarity": processed_dir / f"user_similarity_{label}.parquet",
    }

    posts_base_df.to_parquet(paths["posts_base"], index=False)

    strategy_scores = build_strategy_scores(posts_base_df)
    category_strategy_scores = build_category_strategy_scores(posts_base_df)
    interaction_matrix = build_interaction_matrix(posts_base_df)
    user_similarity_df = build_user_similarity(interaction_matrix)

    strategy_scores.to_parquet(paths["strategy_scores"], index=False)
    category_strategy_scores.to_parquet(paths["category_strategy_scores"], index=False)
    interaction_matrix.to_parquet(paths["interaction_matrix"])
    user_similarity_df.to_parquet(paths["user_similarity"])

    return paths


def plot_model_comparison(results_df: pd.DataFrame, figures_dir: Path, k: int) -> Path:
    plot_df = results_df.sort_values("ndcg_at_k", ascending=True)
    metrics = ["precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Model Comparison (K={k})", fontsize=14)

    for axis, metric in zip(axes.flat, metrics):
        axis.barh(plot_df["model"], plot_df[metric], color="#4C72B0")
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlim(0, max(0.01, plot_df[metric].max() * 1.15))
        axis.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    output_path = figures_dir / "model_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_engagement_by_category(posts_base_df: pd.DataFrame, figures_dir: Path) -> Path:
    summary = (
        posts_base_df.groupby("category")["log_engagement_score"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    summary.plot(kind="bar", ax=axis, color="#55A868")
    axis.set_title("Average Log Engagement by Category (Top 10)")
    axis.set_ylabel("Mean log engagement score")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = figures_dir / "engagement_by_category.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_pipeline(config: PipelineConfig) -> dict:
    dirs = _ensure_dirs(config)
    posts_base_df = load_posts_base(config)

    label = str(len(posts_base_df))
    artifact_paths = save_model_tables(posts_base_df, dirs["processed"], label)

    train_df, test_df = prepare_evaluation_frames(posts_base_df)
    results_df, hybrid_alpha = evaluate_all_models(train_df, test_df, k=config.k)

    results_path = dirs["results"] / "model_comparison.csv"
    split_summary_path = dirs["results"] / "split_summary.txt"
    results_df.to_csv(results_path, index=False)

    split_summary_path.write_text(
        "\n".join(
            [
                f"posts_total={len(posts_base_df)}",
                f"train_posts={len(train_df)}",
                f"test_posts={len(test_df)}",
                f"unique_influencers={posts_base_df['influencer_name'].nunique()}",
                f"unique_strategies={posts_base_df['strategy'].nunique()}",
                f"target_posts={config.target_posts}",
                f"k={config.k}",
                f"hybrid_alpha={hybrid_alpha}",
            ]
        ),
        encoding="utf-8",
    )

    figure_paths = [
        plot_model_comparison(results_df, dirs["figures"], config.k),
        plot_engagement_by_category(posts_base_df, dirs["figures"]),
    ]

    return {
        "posts_base_df": posts_base_df,
        "train_df": train_df,
        "test_df": test_df,
        "results_df": results_df,
        "hybrid_alpha": hybrid_alpha,
        "artifact_paths": artifact_paths,
        "results_path": results_path,
        "split_summary_path": split_summary_path,
        "figure_paths": figure_paths,
    }

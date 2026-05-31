"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

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
    run_label: str | None = field(default=None)


def run_dir_for_scale(base_output_dir: Path, target_posts: int) -> Path:
    """Per-scale artifact folder so runs do not overwrite each other."""
    return base_output_dir / "runs" / f"n{target_posts}"


def resolve_output_dir(config: PipelineConfig) -> Path:
    if config.run_label:
        return config.output_dir / "runs" / config.run_label
    return run_dir_for_scale(config.output_dir, config.target_posts)


def _ensure_dirs(output_dir: Path) -> dict[str, Path]:
    processed = output_dir / "processed"
    results = output_dir / "results"
    figures = output_dir / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {"processed": processed, "results": results, "figures": figures}


def load_posts_base(config: PipelineConfig, run_output_dir: Path) -> pd.DataFrame:
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
            output_dir=run_output_dir / "processed" / "parts",
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


def plot_model_comparison(results_df: pd.DataFrame, figures_dir: Path, k: int, title_suffix: str = "") -> Path:
    plot_df = results_df.sort_values("ndcg_at_k", ascending=True)
    metrics = ["precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    title = f"Model Comparison (K={k})"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    fig.suptitle(title, fontsize=14)

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


def aggregate_scale_comparisons(artifacts_dir: Path) -> dict[str, Path]:
    """Combine per-scale run metrics into comparison tables and charts."""
    runs_root = artifacts_dir / "runs"
    comparisons_dir = artifacts_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    model_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for run_dir in sorted(runs_root.glob("n*")):
        if not run_dir.is_dir():
            continue

        results_path = run_dir / "results" / "model_comparison.csv"
        summary_path = run_dir / "results" / "run_summary.txt"
        if not results_path.exists():
            continue

        target_posts = int(run_dir.name.removeprefix("n"))
        results_df = pd.read_csv(results_path)
        results_df.insert(0, "target_posts", target_posts)
        model_frames.append(results_df)

        summary = {"target_posts": target_posts, "run_dir": str(run_dir)}
        if summary_path.exists():
            for line in summary_path.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    summary[key.strip()] = value.strip()
        best = results_df.sort_values("ndcg_at_k", ascending=False).iloc[0]
        summary["best_model"] = best["model"]
        summary["best_ndcg_at_k"] = best["ndcg_at_k"]
        summary_rows.append(summary)

    if not model_frames:
        raise FileNotFoundError(f"No completed runs found under {runs_root}")

    scale_model_comparison = pd.concat(model_frames, ignore_index=True)
    scale_summary = pd.DataFrame(summary_rows).sort_values("target_posts")

    model_path = comparisons_dir / "scale_model_comparison.csv"
    summary_path_out = comparisons_dir / "scale_summary.csv"
    scale_model_comparison.to_csv(model_path, index=False)
    scale_summary.to_csv(summary_path_out, index=False)

    pivot = scale_model_comparison.pivot(index="model", columns="target_posts", values="ndcg_at_k")
    fig, axis = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=axis, rot=0)
    axis.set_title("NDCG@K by Model Across Dataset Scales")
    axis.set_xlabel("Model")
    axis.set_ylabel("NDCG@K")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    chart_path = comparisons_dir / "scale_ndcg_by_model.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "scale_model_comparison": model_path,
        "scale_summary": summary_path_out,
        "scale_ndcg_chart": chart_path,
    }


def run_pipeline(config: PipelineConfig) -> dict:
    run_output_dir = resolve_output_dir(config)
    dirs = _ensure_dirs(run_output_dir)
    started = time.perf_counter()

    posts_base_df = load_posts_base(config, run_output_dir)
    label = str(config.target_posts)
    artifact_paths = save_model_tables(posts_base_df, dirs["processed"], label)

    train_df, test_df = prepare_evaluation_frames(posts_base_df)
    results_df, hybrid_alpha = evaluate_all_models(train_df, test_df, k=config.k)

    elapsed_seconds = time.perf_counter() - started

    results_path = dirs["results"] / "model_comparison.csv"
    run_summary_path = dirs["results"] / "run_summary.txt"
    results_df.to_csv(results_path, index=False)

    run_summary_path.write_text(
        "\n".join(
            [
                f"target_posts={config.target_posts}",
                f"posts_total={len(posts_base_df)}",
                f"train_posts={len(train_df)}",
                f"test_posts={len(test_df)}",
                f"unique_influencers={posts_base_df['influencer_name'].nunique()}",
                f"unique_strategies={posts_base_df['strategy'].nunique()}",
                f"k={config.k}",
                f"hybrid_alpha={hybrid_alpha}",
                f"runtime_seconds={elapsed_seconds:.1f}",
                f"run_dir={run_output_dir}",
            ]
        ),
        encoding="utf-8",
    )

    figure_paths = [
        plot_model_comparison(
            results_df,
            dirs["figures"],
            config.k,
            title_suffix=f"n={config.target_posts:,}",
        ),
        plot_engagement_by_category(posts_base_df, dirs["figures"]),
    ]

    return {
        "posts_base_df": posts_base_df,
        "train_df": train_df,
        "test_df": test_df,
        "results_df": results_df,
        "hybrid_alpha": hybrid_alpha,
        "runtime_seconds": elapsed_seconds,
        "run_output_dir": run_output_dir,
        "artifact_paths": artifact_paths,
        "results_path": results_path,
        "run_summary_path": run_summary_path,
        "figure_paths": figure_paths,
    }

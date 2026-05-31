"""Run the full recommender pipeline (Phase 1 + Phase 2)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineConfig, aggregate_scale_comparisons, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed datasets, train recommenders, and evaluate all models."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing influencers.txt and optional raw metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory for processed tables, metrics, and figures.",
    )
    parser.add_argument(
        "--posts-parquet",
        type=Path,
        default=None,
        help="Use an existing processed posts parquet (e.g. from Colab).",
    )
    parser.add_argument(
        "--extracted-metadata-dir",
        type=Path,
        default=None,
        help="Parse metadata .info/.json files from an extracted folder.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic posts from influencers.txt for local runs.",
    )
    parser.add_argument(
        "--synthetic-influencers",
        type=int,
        default=120,
        help="Minimum influencer pool size when --synthetic is set.",
    )
    parser.add_argument(
        "--target-posts",
        type=int,
        default=20_000,
        help="Target number of posts for synthetic generation or metadata parsing cap.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K for recommendation and ranking metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic generation.",
    )
    parser.add_argument(
        "--compare-scales",
        action="store_true",
        help="After this run, rebuild artifacts/comparisons/ from all runs under artifacts/runs/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not any([args.posts_parquet, args.extracted_metadata_dir, args.synthetic]):
        print("No real metadata detected locally; defaulting to --synthetic for a full pipeline run.")
        args.synthetic = True

    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_posts=args.target_posts,
        posts_parquet=args.posts_parquet,
        extracted_metadata_dir=args.extracted_metadata_dir,
        synthetic=args.synthetic,
        synthetic_influencers=args.synthetic_influencers,
        k=args.k,
        seed=args.seed,
    )

    outputs = run_pipeline(config)

    print("\n=== Pipeline Complete ===")
    print(f"Run directory: {outputs['run_output_dir']}")
    print(f"Posts base: {outputs['artifact_paths']['posts_base']}")
    print(f"Model comparison: {outputs['results_path']}")
    print(f"Run summary: {outputs['run_summary_path']}")
    print(f"Runtime: {outputs['runtime_seconds']:.1f}s")
    print(f"Hybrid alpha: {outputs['hybrid_alpha']:.2f}")
    print("\nMetrics:")
    print(outputs["results_df"].to_string(index=False))
    print("\nFigures:")
    for path in outputs["figure_paths"]:
        print(f"- {path}")

    if args.compare_scales:
        print("\n=== Updating scale comparison ===")
        comparison_paths = aggregate_scale_comparisons(args.output_dir)
        for name, path in comparison_paths.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()

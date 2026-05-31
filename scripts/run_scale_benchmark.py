"""Run pipeline at multiple post scales and build comparison artifacts."""

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
        description="Run scale benchmarks without overwriting prior results."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[10_000, 20_000, 50_000],
        help="Target post counts to benchmark (each saved under artifacts/runs/n{scale}/).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic posts for each scale.",
    )
    parser.add_argument(
        "--synthetic-influencers",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip runs and only rebuild comparison CSVs/charts from existing runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.compare_only:
        if not args.synthetic:
            print("Defaulting to --synthetic for scale benchmarks.")
            args.synthetic = True

        for target_posts in args.scales:
            print(f"\n=== Running scale n={target_posts:,} ===")
            config = PipelineConfig(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                target_posts=target_posts,
                synthetic=args.synthetic,
                synthetic_influencers=args.synthetic_influencers,
                k=args.k,
                seed=args.seed,
            )
            outputs = run_pipeline(config)
            print(f"Saved to: {outputs['run_output_dir']}")
            print(f"Posts: {len(outputs['posts_base_df']):,}")
            print(f"Runtime: {outputs['runtime_seconds']:.1f}s")
            print(outputs["results_df"][["model", "ndcg_at_k", "evaluated_users"]].to_string(index=False))

    print("\n=== Building scale comparison ===")
    comparison_paths = aggregate_scale_comparisons(args.output_dir)
    for name, path in comparison_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

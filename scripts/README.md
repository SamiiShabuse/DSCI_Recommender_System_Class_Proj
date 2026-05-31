# Scripts (`scripts/`)

Command-line entry points for running the pipeline, benchmarking at multiple scales, and inspecting local data. All scripts add the repo root to `sys.path` and import from `src/`.

Run every command from the **repository root**:

```bash
cd path/to/DSCI_Recommender_System_Class_Proj
python scripts/<script>.py [options]
```

---

## Script guide

| Script | Purpose |
|--------|---------|
| **`run_pipeline.py`** | **Main entry point** — build data, train all 5 models, evaluate, save metrics and figures |
| `run_scale_benchmark.py` | Run the pipeline at multiple post counts (10k, 20k, 50k, …) and build scale comparison charts |
| `dataset_summary.py` | Print high-level stats about `data/influencers.txt` and mapping files |
| `preview_dataset.py` | Show sample rows from influencers and mapping; optional sample image extraction |
| `build_training_subset.py` | Build a smaller influencer/post subset for Colab or limited-RAM environments |

---

## `run_pipeline.py` (primary)

Runs the full recommender pipeline via `src.pipeline.run_pipeline()`.

```bash
# Local dev — synthetic posts (no metadata zip needed)
python scripts/run_pipeline.py --synthetic

# Real metadata from an extracted folder
python scripts/run_pipeline.py --extracted-metadata-dir data/Post_metadata --target-posts 10000

# Reuse a processed parquet from Colab
python scripts/run_pipeline.py --posts-parquet path/to/posts_base_10000.parquet

# After a run, refresh artifacts/comparisons/ from all runs under artifacts/runs/
python scripts/run_pipeline.py --synthetic --compare-scales
```

**Common flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--target-posts` | 20000 | Number of posts to use or generate |
| `--k` | 5 | Top-K for recommendations and metrics |
| `--seed` | 42 | Random seed for synthetic data |
| `--data-dir` | `data/` | Location of `influencers.txt` |
| `--output-dir` | `artifacts/` | Root for run outputs |

**Outputs** (per run): `artifacts/runs/n{target_posts}/`

- `processed/` — parquets (posts, interaction matrix, strategy scores)
- `results/model_comparison.csv` — metrics for all 5 models
- `results/run_summary.txt` — post counts, runtime, hybrid α
- `figures/` — model comparison and engagement charts

If no data source is specified, the script defaults to `--synthetic`.

---

## `run_scale_benchmark.py`

Runs `run_pipeline.py` logic at several scales without overwriting prior runs, then aggregates results.

```bash
# Default scales: 10k, 20k, 50k (synthetic)
python scripts/run_scale_benchmark.py --synthetic

# Custom scales
python scripts/run_scale_benchmark.py --synthetic --scales 10000 20000 50000 100000

# Rebuild comparison CSVs/charts only (no new runs)
python scripts/run_scale_benchmark.py --compare-only
```

**Outputs:** `artifacts/comparisons/`

- `scale_model_comparison.csv`
- `scale_summary.csv`
- `scale_ndcg_by_model.png`

---

## `dataset_summary.py`

Quick sanity check on local data files. No arguments required.

```bash
python scripts/dataset_summary.py
```

Prints influencer count, category breakdown, follower/post ranges, and mapping row estimate. Stats are also documented in [data/README.md](../data/README.md).

---

## `preview_dataset.py`

Shows example records from local files for debugging and report writing.

```bash
python scripts/preview_dataset.py
python scripts/preview_dataset.py --extract-sample-images
```

---

## `build_training_subset.py`

Creates a manageable training subset when the full dataset is too large for local RAM or Colab free tier.

```bash
python scripts/build_training_subset.py --num-influencers 500 --posts-per-influencer 20
```

See `--help` for output paths and sampling options. Used mainly during early data prep; final pipeline runs typically use Colab-extracted metadata.

---

## Related docs

- [src/README.md](../src/README.md) — modules called by these scripts
- [docs/Pipeline.md](../docs/Pipeline.md) — detailed pipeline reference
- [docs/Colab_Setup.md](../docs/Colab_Setup.md) — large-data workflow

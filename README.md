# Creator Intelligence Recommender System

**Team:** Samii Shabuse, Savit Tumuluri, Han Truong  
**Course:** DSCI351 - Recommender Systems

Recommends Instagram **content strategies** to influencers based on past engagement. A strategy is a pattern like:

`evening + medium_caption + few_hashtags + not_ad + image`

(not a specific post or caption - when to post, caption length, hashtags, ad vs organic, image vs video).

---

## What this project does

| | |
|---|---|
| **Users** | Instagram influencers |
| **Items** | Content strategy labels (combinations of posting context + format) |
| **Task** | Top-N recommendation — suggest which strategies an influencer should try next |
| **Evaluation** | Time-based split (last 20% of each influencer's posts held out); Precision@5, Recall@5, NDCG@5, Hit Rate@5 |

Five models are compared: global baseline, category baseline, user-based collaborative filtering, content-based (TF-IDF captions), and a weighted hybrid.

---

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Git** to clone the repository
- **~500 MB disk** for dependencies and a local run; full post metadata is ~189 GB and stays on Google Drive / Colab

---

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/SamiiShabuse/DSCI_Recommender_System_Class_Proj
cd DSCI_Recommender_System_Class_Proj
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `pyarrow`, `tqdm`, `matplotlib`.

### 2. Choose how to review the project

| Goal | What to do |
|------|------------|
| **See results without running code** | Open pre-computed metrics below |
| **Run the full pipeline locally** | Follow [Run locally (synthetic)](#run-locally-synthetic) |
| **Run on real Instagram metadata** | Follow [Run on real data](#run-on-real-data-colab) |
| **Step through interactively** | Open [notebooks/01_pipeline_and_models.ipynb](notebooks/01_pipeline_and_models.ipynb) |

---

## Pre-computed results (no run required)

Committed run outputs are included for grading. Primary 10k-post run:

| File | Description |
|------|-------------|
| [artifacts/runs/n10000/results/model_comparison.csv](artifacts/runs/n10000/results/model_comparison.csv) | Metrics for all 5 models |
| [artifacts/runs/n10000/results/run_summary.txt](artifacts/runs/n10000/results/run_summary.txt) | Split sizes, runtime, hyperparameters |
| [artifacts/runs/n10000/figures/model_comparison.png](artifacts/runs/n10000/figures/model_comparison.png) | Bar chart of metrics |
| [artifacts/comparisons/scale_model_comparison.csv](artifacts/comparisons/scale_model_comparison.csv) | Results across 10k / 20k / 50k / 100k scales |

Additional runs: `artifacts/runs/n20000/`, `n50000/`, `n100000/`.

---

## Run locally (synthetic)

The easiest way to reproduce end-to-end on a laptop **without the full dataset**.

### Step 1 — Influencer profiles

For a quick local run, no download is required. If `data/influencers.txt` is not present, synthetic mode uses built-in demo influencer profiles so the pipeline still runs in a fresh clone.

For the most realistic synthetic run, download `influencers.txt` from the [official dataset](https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset) or [Google Drive mirror](https://drive.google.com/drive/folders/1ISiSH4-aM6kP_0lKQYejpk1sa6Jei-7e) and place it in:

```
data/influencers.txt
```

See [data/README.md](data/README.md) for other optional files. Real metadata and images are not required for a synthetic run.

### Step 2 — Run the pipeline

```bash
python scripts/run_pipeline.py --synthetic --target-posts 10000
```

If you omit all data flags, the script **defaults to `--synthetic`** automatically.

**Common options:**

```bash
python scripts/run_pipeline.py --target-posts 10
python scripts/run_pipeline.py --target-posts 1000
python scripts/run_pipeline.py --synthetic --target-posts 10000 --k 5 --seed 42
python scripts/run_scale_benchmark.py --synthetic --scales 10000 20000 50000
```

`--target-posts 10` is only a fast smoke test. It verifies that the pipeline runs and writes artifacts, but it may show `evaluated_users=0` because there are too few posts per influencer for a meaningful train/test evaluation. Use `--target-posts 1000` or higher to see non-empty metrics.

### Step 3 — Check outputs

After a successful run, the terminal prints a metrics table and paths like:

```
artifacts/runs/n10000/results/model_comparison.csv
artifacts/runs/n10000/figures/model_comparison.png
```

Default `--target-posts` is **20,000** (output folder `n20000`) if you do not set it explicitly.

---

## Run on real data (Colab)

Full post metadata is too large for most local machines. The team workflow:

1. Extract a subset of post metadata in Google Colab — see [docs/Colab_Setup.md](docs/Colab_Setup.md)
2. Run the pipeline on the extracted files or a saved parquet:

```bash
# From extracted .info / JSON metadata folder
python scripts/run_pipeline.py --extracted-metadata-dir "path/to/Post_metadata" --target-posts 10000

# From a posts_base parquet produced in Colab
python scripts/run_pipeline.py --posts-parquet "path/to/posts_base_10000.parquet" --target-posts 10000
```

If you prepare an additional narrative beyond the required slides/repo, prefer numbers from **real metadata** (at least 10k posts) over synthetic data.

---

## What the pipeline does (end-to-end)

`scripts/run_pipeline.py` orchestrates the full flow in `src/pipeline.py`:

```
Raw input (synthetic / metadata / parquet)
        ↓
  preprocess.py — parse posts, build strategy labels, engagement scores
        ↓
  posts_base parquet — one row per post with features
        ↓
  Train recommenders (baselines, CF, content-based, hybrid)
        ↓
  evaluation.py — time-based train/test split, ranking metrics
        ↓
  CSV metrics + PNG figures → artifacts/runs/n{scale}/
```

**Strategy label** (defined in `src/data/preprocess.py`):

```
{time_bucket} + {caption_bucket} + {hashtag_bucket} + {ad_bucket} + {media_bucket}
```

**Engagement score:** `log1p(likes + 2×comments) / log1p(followers)`

**Test relevance:** strategies in the held-out set with pseudo-rating ≥ 5 (top engagement quintile per influencer).

Module details: [src/README.md](src/README.md) · CLI reference: [scripts/README.md](scripts/README.md)

---

## Outputs reference

| Path | Description |
|------|-------------|
| `artifacts/runs/n*/processed/posts_base_*.parquet` | Modeling-ready post table |
| `artifacts/runs/n*/processed/interaction_matrix_*.parquet` | Influencer × strategy matrix |
| `artifacts/runs/n*/results/model_comparison.csv` | Precision, Recall, NDCG, Hit Rate for all 5 models |
| `artifacts/runs/n*/results/run_summary.txt` | Post counts, influencers, k, hybrid α, runtime |
| `artifacts/runs/n*/figures/model_comparison.png` | Metrics bar chart |
| `artifacts/runs/n*/figures/engagement_by_category.png` | EDA figure |
| `artifacts/comparisons/` | Multi-scale benchmark CSVs and charts |

Processed parquets are gitignored; metrics and figures are committed.

---

## Project structure

Each folder has a README with more detail:

| Folder | README | Contents |
|--------|--------|----------|
| `src/` | [src/README.md](src/README.md) | `data/` (preprocess), `models/` (5 recommenders), `pipeline.py`, `evaluation.py` |
| `scripts/` | [scripts/README.md](scripts/README.md) | CLI entry points (`run_pipeline.py`, benchmarks, data tools) |
| `notebooks/` | [notebooks/README.md](notebooks/README.md) | Jupyter notebooks (primary: `01_pipeline_and_models.ipynb`) |
| `docs/` | [docs/README.md](docs/README.md) | Proposal, pipeline guide, submission checklist, Colab setup |
| `data/` | [data/README.md](data/README.md) | Raw dataset files (gitignored) and dataset snapshot |
| `artifacts/` | — | Generated metrics, figures, and processed tables |

---

## Models evaluated

| Model | Module | Approach |
|-------|--------|----------|
| Global baseline | `src/models/baselines.py` | Top strategies by average engagement (all influencers) |
| Category baseline | `src/models/baselines.py` | Top strategies within the influencer's category |
| User-based CF | `src/models/collaborative.py` | Cosine similarity on influencer × strategy matrix |
| Content-based | `src/models/content_based.py` | TF-IDF caption similarity to strategy profiles |
| Hybrid | `src/models/hybrid.py` | Weighted blend of CF + content-based (α tuned on validation users) |

---

## Dataset

- **Source:** [Instagram Influencer Dataset](https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset) (Seungbae Kim)
- **Mirror:** [Google Drive](https://drive.google.com/drive/folders/1ISiSH4-aM6kP_0lKQYejpk1sa6Jei-7e)
- **What we use:** `influencers.txt`, mapping files, and post metadata JSON (no images required)
- **Working subset:** ~10,000 randomly sampled posts for primary evaluation (expandable to 20k–100k)

Local `data/` contents are gitignored except [data/README.md](data/README.md). Download files from the links above.

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/README.md](docs/README.md) | Full documentation index |
| [docs/How_It_Works.md](docs/How_It_Works.md) | Plain-language overview — **start here for grading** |
| [docs/Pipeline.md](docs/Pipeline.md) | Technical pipeline reference |
| [docs/Project_Proposal.md](docs/Project_Proposal.md) | Original project scope |
| [docs/Final_Submission.md](docs/Final_Submission.md) | Final checklist and rubric mapping |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: influencers.txt` | Download `influencers.txt` into `data/` (see [data/README.md](data/README.md)) |
| Pipeline runs but metrics look empty | Check `evaluated_users` in `model_comparison.csv`; need influencers with enough posts in train and test |
| Out of memory on real metadata | Use Colab and cap with `--target-posts 10000`, or use `--posts-parquet` from a prior Colab run |
| Want to inspect raw data first | `python scripts/dataset_summary.py` and `python scripts/preview_dataset.py` |

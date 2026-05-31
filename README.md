# Creator Intelligence Recommender System

**Team:** Samii Shabuse, Savit Tumuluri, Han Truong  
**Course:** DSCI351 — Recommender Systems

Recommends Instagram **content strategies** (posting time, caption style, hashtags, ad/organic, image/video) to influencers based on engagement patterns.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --synthetic
```

Outputs land in `artifacts/`:

| Path | Description |
|------|-------------|
| `artifacts/processed/posts_base_*.parquet` | Modeling-ready post table |
| `artifacts/processed/interaction_matrix_*.parquet` | Influencer × strategy matrix |
| `artifacts/results/model_comparison.csv` | Phase 2 metrics for all 5 models |
| `artifacts/figures/model_comparison.png` | Metrics chart for report/slides |

## Run on Real Colab Data

After extracting metadata in Colab (see `docs/Colab_Setup.md`):

```bash
# Option A: from extracted metadata folder
python scripts/run_pipeline.py \
  --extracted-metadata-dir "/path/to/Post_metadata_10000_extracted"

# Option B: from an existing posts_base parquet
python scripts/run_pipeline.py \
  --posts-parquet "/path/to/posts_base_10000.parquet"
```

## Project Structure

```
src/
  preprocess.py       # Metadata parsing, strategy labels, engagement scores
  baselines.py        # Global + category popularity models
  collaborative.py    # User-based collaborative filtering
  content_based.py    # TF-IDF caption recommender
  hybrid.py           # Weighted CF + content blend
  evaluation.py       # Time split, Precision@K, NDCG@K
  demo_data.py        # Synthetic local data generator
  pipeline.py         # End-to-end orchestration
scripts/
  run_pipeline.py     # Main entry point (Phase 1 + 2)
  build_training_subset.py
  dataset_summary.py
  preview_dataset.py
notebooks/
  01_pipeline_and_models.ipynb
docs/
  Project_Proposal.md
  Milestones.md
  Final_Submission.md
  Pipeline.md
  Colab_Setup.md
```

## Models Evaluated

1. **Global baseline** — top strategies by average engagement
2. **Category baseline** — top strategies within influencer category
3. **User-based CF** — similar influencers' successful strategies
4. **Content-based** — TF-IDF caption similarity
5. **Hybrid** — weighted combination (α tuned on validation users)

## Dataset

- [Instagram Influencer Dataset](https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset)
- [Google Drive mirror](https://drive.google.com/drive/folders/1ISiSH4-aM6kP_0lKQYejpk1sa6Jei-7e)

Local `data/` folder (gitignored): `influencers.txt`, mapping files, optional sample images. Full post metadata stays on Google Drive / Colab.

## Documentation

- [Pipeline guide](docs/Pipeline.md) — modules, outputs, reproduction
- [Milestones](docs/Milestones.md) — project status and phase checklist
- [Final submission](docs/Final_Submission.md) — report outline and Canvas checklist
- [Colab setup](docs/Colab_Setup.md) — large-data workflow

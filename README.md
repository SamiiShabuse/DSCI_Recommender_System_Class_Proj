# Creator Intelligence Recommender System

**Team:** Samii Shabuse, Savit Tumuluri, Han Truong  
**Course:** DSCI351 — Recommender Systems

Recommends Instagram **content strategies** (posting time, caption style, hashtags, ad/organic, image/video) to influencers based on engagement patterns.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --synthetic
```

Outputs land in `artifacts/runs/n{target_posts}/` (e.g. `n10000`, `n20000`):

| Path | Description |
|------|-------------|
| `artifacts/runs/n*/processed/posts_base_*.parquet` | Modeling-ready post table |
| `artifacts/runs/n*/processed/interaction_matrix_*.parquet` | Influencer × strategy matrix |
| `artifacts/runs/n*/results/model_comparison.csv` | Metrics for all 5 models |
| `artifacts/runs/n*/figures/model_comparison.png` | Metrics chart for report/slides |
| `artifacts/comparisons/` | Multi-scale benchmark tables and charts |

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

Each folder has a README explaining its purpose:

| Folder | README | Contents |
|--------|--------|----------|
| `src/` | [src/README.md](src/README.md) | Python modules: preprocess, 5 models, evaluation, pipeline |
| `scripts/` | [scripts/README.md](scripts/README.md) | CLI entry points (`run_pipeline.py`, benchmarks, data tools) |
| `notebooks/` | [notebooks/README.md](notebooks/README.md) | Jupyter notebooks (primary: `01_pipeline_and_models.ipynb`) |
| `docs/` | [docs/README.md](docs/README.md) | Proposal, pipeline guide, submission checklist, Colab setup |
| `data/` | [data/README.md](data/README.md) | Raw dataset files (gitignored) and dataset snapshot |
| `artifacts/` | — | Generated metrics, figures, and processed tables (see `scripts/README.md`) |

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

Start at [docs/README.md](docs/README.md) for a full index. Key docs:

- [How it works](docs/How_It_Works.md) — plain-language overview for graders and teammates
- [Pipeline guide](docs/Pipeline.md) — modules, outputs, reproduction
- [Final submission](docs/Final_Submission.md) — report outline and Canvas checklist
- [Colab setup](docs/Colab_Setup.md) — large-data workflow

# Notebooks (`notebooks/`)

Jupyter notebooks for data extraction, exploration, and interactive pipeline runs. The **canonical** notebook for grading is `01_pipeline_and_models.ipynb`; the others are earlier prototypes kept for reference.

**Prefer scripts for reproduction:** `python scripts/run_pipeline.py` produces the same end-to-end outputs without manual notebook steps. See [docs/Pipeline.md](../docs/Pipeline.md).

---

## Notebook guide

| Notebook | Status | Purpose |
|----------|--------|---------|
| **`01_pipeline_and_models.ipynb`** | **Primary** | End-to-end walkthrough: load data, build features, train all 5 models, view metrics |
| `Data_Extraction.ipynb` | Legacy | Colab workflow to extract post metadata from `posts_info.zip` and build subsets |
| `Cleaned Up Notebook.ipynb` | Legacy | Early EDA, baselines, and CF prototyping before code moved to `src/` |

---

## When to use which

### `01_pipeline_and_models.ipynb` (recommended for demos)

Use when you want to:

- Step through the pipeline cell-by-cell for a presentation
- Inspect intermediate DataFrames (posts, strategies, interaction matrix)
- Run on Colab with Google Drive paths

Imports from `src/` — logic lives in modules, not duplicated in the notebook.

### `Data_Extraction.ipynb`

Use when you need to:

- Extract a sample of post metadata from the full archive on Colab
- Produce `posts_base_*.parquet` for `--posts-parquet` or local analysis

After extraction, run the pipeline from the repo root:

```bash
python scripts/run_pipeline.py --posts-parquet path/to/posts_base_10000.parquet
```

### `Cleaned Up Notebook.ipynb`

Historical notebook from early development. Superseded by `src/` modules and `01_pipeline_and_models.ipynb`. Kept for audit trail only.

---

## Environment setup

From the repo root:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_pipeline_and_models.ipynb
```

For large data, open in **Google Colab** and follow [docs/Colab_Setup.md](../docs/Colab_Setup.md).

---

## Related docs

- [src/README.md](../src/README.md) — what each Python module does
- [scripts/README.md](../scripts/README.md) — command-line entry points
- [docs/How_It_Works.md](../docs/How_It_Works.md) — project overview

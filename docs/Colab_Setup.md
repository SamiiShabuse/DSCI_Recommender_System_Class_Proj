# Google Colab Workflow (Recommended for This Dataset)

The raw dataset is too large for many local machines, so use a Colab-first workflow and keep raw data in Google Drive.

## Recommended First Pass

For the easiest first version, keep the project metadata-only and ignore the image files for now.

Use this target:

- Recommend an Instagram content strategy from past metadata and engagement.

Use this as the first working dataset scope:

- `influencers.txt`
- `JSON-Image_files_mapping.txt`
- one metadata folder such as `Post_metadata/`

The first notebook goal is simply to load the data, inspect it, and define a small subset.

## 0. Set Up Path Variables

In Colab, after mounting Drive, define one base path and build everything from it:

```python
from pathlib import Path

DATA_PATH = Path("/content/drive/MyDrive/dsci351_data")
INFLUENCERS_PATH = DATA_PATH / "influencers.txt"
MAPPING_PATH = DATA_PATH / "JSON-Image_files_mapping.txt"
METADATA_DIR = DATA_PATH / "Post_metadata"
```

If your folder uses a different name, change only `DATA_PATH`.

## 1. First Checks

Before modeling, verify that the basic files exist:

```python
print(INFLUENCERS_PATH.exists())
print(MAPPING_PATH.exists())
print(METADATA_DIR.exists())
```

Then list the first few items:

```python
print([p.name for p in DATA_PATH.iterdir()][:20])
```

## 2. First Goal

The first real output should be a small, clean subset plus a dataset summary. That means:

- confirm row counts and categories in `influencers.txt`
- confirm how many mapping rows exist
- confirm whether the metadata folder is present and readable
- build a small subset for faster experimentation

## 1. Put Data in Google Drive

Create a folder like this in Drive:

- MyDrive/dsci351_data/influencers.txt
- MyDrive/dsci351_data/JSON-Image_files_mapping.txt
- MyDrive/dsci351_data/Post_metadata/ (or post_metadata, metadata, JSON_files)

Use any metadata folder name above because the loader checks all of them.

## 2. Open Colab and Run Setup Cells

```python
from google.colab import drive
import os

drive.mount('/content/drive')
```

```python
%cd /content
!git clone https://github.com/<your-team-org-or-user>/DSCI_Recommender_System_Class_Proj.git
%cd /content/DSCI_Recommender_System_Class_Proj
!python -m pip install -U pip
!python -m pip install -r requirements.txt
```

## 3. Build a Training Subset First

Start with a subset so your modeling loop is fast and stable:

```python
!python scripts/build_training_subset.py \
  --data-dir "/content/drive/MyDrive/dsci351_data" \
  --output-dir "/content/drive/MyDrive/dsci351_artifacts/processed" \
  --num-influencers 1500 \
  --min-followers 5000 \
  --max-posts-per-influencer 150 \
  --max-metadata-files 200000
```

Outputs:

- selected_influencers.csv
- selected_mapping.csv
- selected_metadata.jsonl

## 4. Iterate on Subset, Then Scale

During development:

- Keep num-influencers between 500 and 2000.
- Keep max-posts-per-influencer between 50 and 200.
- Keep max-metadata-files capped to avoid long runs.

For a final run:

- Increase limits gradually.
- Keep output in Drive to avoid Colab VM reset data loss.

## 5. Current Fast Health Checks

```python
!python scripts/preview_dataset.py
!python scripts/dataset_summary.py
```

## 6. Run Full Pipeline (Phases 1 & 2)

After extracting metadata (e.g. 10k sample to `/content/Post_metadata_10000_extracted`):

```bash
!python scripts/run_pipeline.py \
  --extracted-metadata-dir "/content/Post_metadata_10000_extracted" \
  --output-dir "/content/drive/MyDrive/dsci351_artifacts" \
  --k 5
```

Or reuse a saved posts base parquet:

```bash
!python scripts/run_pipeline.py \
  --posts-parquet "/content/drive/MyDrive/dsci351_artifacts/processed/posts_base_10000.parquet" \
  --output-dir "/content/drive/MyDrive/dsci351_artifacts" \
  --k 5
```

Outputs:

- `artifacts/runs/n{target_posts}/processed/posts_base_*.parquet`
- `artifacts/runs/n{target_posts}/results/model_comparison.csv`
- `artifacts/runs/n{target_posts}/figures/model_comparison.png`

See [Pipeline.md](Pipeline.md) for module details.

## Team Drive Paths

```python
DATA_PATH = Path("/content/drive/.shortcut-targets-by-id/1ISiSH4-aM6kP_0lKQYejpk1sa6Jei-7e/Instagram influencer dataset")
REPO_CODE = Path("/content/drive/MyDrive/DSCI351/Class Project/DSCI_Recommender_System_Class_Proj")
POSTS_INFO_ZIP = DATA_PATH / "Post_metadata" / "posts_info.zip"
```

## Notes

- If metadata is not present yet, run `python scripts/run_pipeline.py --synthetic` locally for a smoke test.
- Large processed parquets stay gitignored; final metrics CSVs and figures are saved under `artifacts/runs/n{target_posts}/`.

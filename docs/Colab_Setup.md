# Google Colab Workflow (Recommended for This Dataset)

The raw dataset is too large for many local machines, so use a Colab-first workflow and keep raw data in Google Drive.

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

## Notes

- If metadata is not present yet, subset build still works for influencer and mapping tables.
- This flow is designed so your team can make progress immediately and avoid local storage bottlenecks.

# Data Folder

Raw and processed inputs for the Instagram influencer recommender system. Large files are gitignored; clone or download them locally as needed.

## Official sources

- [Instagram Influencer Dataset](https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset)
- [Google Drive mirror](https://drive.google.com/drive/folders/1ISiSH4-aM6kP_0lKQYejpk1sa6Jei-7e)

See [Project Proposal](../docs/Project_Proposal.md) and [Colab Setup](../docs/Colab_Setup.md) for the full workflow.

## Expected local files

| File | Description |
|------|-------------|
| `influencers.txt` | 33,935 influencer profiles (username, category, followers, etc.) |
| `JSON-Image_files_mapping.txt` | Maps influencers to post metadata and image filenames |
| `clean_json_image_mapping.parquet` | Deduplicated mapping (~10M rows) |
| `Post_metadata/` | Extracted post `.info` / JSON metadata (optional locally; often on Colab/Drive) |
| `sample_images.zip` | Optional sample images (project uses metadata only) |

Post metadata from `posts_info.zip` is too large to commit. Extract a subset in Colab and pass the folder to `scripts/run_pipeline.py --extracted-metadata-dir`.

## Dataset snapshot

Summary from `scripts/dataset_summary.py` (regenerate for current counts):

| Metric | Value |
|--------|-------|
| Influencer rows | 33,935 |
| Unique categories | 11 |
| Followers (min / avg / max) | 1,000 / 140,329 / 96,476,007 |
| Posts per influencer (min / avg / max) | 100 / 1,487 / 127,520 |
| Mapping rows (approx.) | 10,078,910 |

**Top categories by influencer count:**

| Category | Count |
|----------|------:|
| fashion | 11,911 |
| other | 5,720 |
| travel | 4,210 |
| family | 4,070 |
| food | 3,565 |
| beauty | 1,542 |
| interior | 1,195 |
| fitness | 1,133 |
| pet | 587 |

## Sample records

From `scripts/preview_dataset.py`:

**Influencer row:**

```text
Username: makeupbynvs
Category: beauty
#Followers: 1432
#Followees: 1089
#Posts: 363
```

**Mapping row:**

```text
influencer_name: 00_rocketgirl
json_postmetadata_file_name: 1188140434601337485.info
image_files: ['1188140434601337485.jpg']
```

## Refresh these stats

```bash
python scripts/dataset_summary.py
python scripts/preview_dataset.py
```

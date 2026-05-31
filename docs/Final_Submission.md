# Final Submission Requirements (DSCI351)

## What to Submit

| Artifact | Status | Notes |
|----------|--------|-------|
| PDF project report (max 3 pages) | **Not started** | Use outline below; map directly to rubric |
| Link to project repository | **Ready** | Ensure README explains how to reproduce results |
| Presentation slides | **Not started** | 8–12 slides; demo 2–3 influencer examples |
| Dataset or link to dataset | **Ready** | Cite official source + describe our 10k-post subset |

Official dataset links (include in report and README):

- Dataset page: https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset
- Google Drive: https://drive.google.com/drive/folders/1ISiSH4-aM6kP_0lKQYejpk1sa6Jei-7e

---

## Report Outline (3 Pages Maximum)

Write one short section per rubric item. Suggested page budget:

### 1. Dataset Description (~0.5 page)

- **Source:** Instagram Influencer Dataset (Seungbae Kim).
- **What we use:** `influencers.txt`, `JSON-Image_files_mapping.txt`, and post metadata from `Post_metadata/posts_info.zip` (metadata-only; no images).
- **Working subset:** ~10,000 randomly sampled posts from the archive, joined to 33,935 influencer profiles across 11 categories.
- **Key fields:** captions, likes, comments, timestamps, sponsorship/video flags, influencer category and follower count.
- **Preprocessing:** cleaned mapping parquet (~10M rows → deduplicated metadata keys), parsed JSON metadata, engineered content-strategy labels.

### 2. Recommender System Type (~0.25 page)

- **Users:** Instagram influencers.
- **Items:** Content strategies (combinations of posting context and format).
- **Task:** Top-N ranked recommendation — suggest which content strategies an influencer should try next to maximize engagement.
- **Approaches:** Popularity baseline, category baseline, user-based collaborative filtering, content-based (caption similarity), and a weighted hybrid.

### 3. Methods / Models (~0.75 page)

| Model | Method | Role |
|-------|--------|------|
| Global popularity | Mean engagement per strategy | Baseline |
| Category popularity | Mean engagement per (category, strategy) | Baseline |
| User-based CF | Cosine similarity on influencer × strategy matrix | Cross-user learning |
| Content-based | TF-IDF on captions → strategy similarity | Text/content signal |
| Hybrid | Weighted blend of CF + content-based scores | Final model |

**Content strategy label (locked definition):**

```
{time_bucket} + {caption_bucket} + {hashtag_bucket} + {ad_bucket} + {media_bucket}
```

Example: `evening + medium_caption + few_hashtags + not_ad + image`

**Engagement signal:**

- Raw: `(likes + 2 × comments) / followers`
- Modeling score: `log1p(raw_engagement) / log1p(followers)`
- Evaluation labels: per-influencer quintile pseudo-ratings (1–5) as described in the proposal

### 4. How to Interpret Recommendations (~0.25 page)

- Each output is a ranked list of **actionable content strategies**, not individual posts.
- A strategy string tells the creator: when to post, caption length, hashtag density, ad vs organic, and image vs video.
- Higher predicted score = strategies that worked well for this influencer (CF) or similar influencers (CF) / similar captions (content-based).
- Show 1–2 concrete examples in slides: influencer history → top-5 recommended strategies → why they make sense.

### 5. Evaluation (~0.5 page)

**Split:** Time-based — train on older posts, test on most recent 20% per influencer (minimum 2 posts per influencer in test when possible).

**Metrics:**

| Metric | Purpose |
|--------|---------|
| Precision@K, Recall@K, NDCG@K | Top-N recommendation quality vs held-out high-engagement strategies |
| MAE / RMSE | Pseudo-rating prediction (optional, secondary) |
| Hit rate@K | Whether any recommended strategy appears in test set |

**Comparison table (required):** Global baseline vs category baseline vs CF vs content-based vs hybrid.

### 6. Limitations and Future Work (~0.25 page)

- Sample of 10k posts, not full 10M+ mapping rows.
- Strategy labels are rule-based buckets, not learned clusters.
- Sparse influencer × strategy matrix limits CF for low-activity users.
- No image features; captions may miss visual content signals.
- Engagement ≠ causation; correlation with platform algorithm changes.
- **Future:** larger sample, matrix factorization (SVD/ALS), learned strategy clustering, image embeddings, causal/offline policy evaluation.

---

## Presentation Slides Checklist

1. Title + team + one-sentence problem
2. Dataset overview (one figure: category distribution or engagement distribution)
3. Problem formulation (users, items, output)
4. Content strategy definition (one example string)
5. Pipeline diagram (raw data → features → models → recommendations)
6. Model comparison table (metrics)
7. Demo: 2 influencer case studies (history + recommendations)
8. Limitations + future work
9. Q&A backup slide with hyperparameters and split details

---

## Repository Submission Checklist

Before submitting the repo link, confirm:

- [ ] `requirements.txt` lists all dependencies (pandas, numpy, scikit-learn, pyarrow, tqdm, etc.)
- [ ] README has Colab setup link and one-command reproduction path
- [ ] Notebook or scripts produce evaluation metrics CSV/table
- [ ] Processed subset instructions documented (Colab/Drive paths)
- [ ] No secrets or personal Drive paths hardcoded without comments
- [ ] `.gitignore` keeps large raw data out of git (link to dataset instead)

---

## Submission Mapping

The 3-page report sections above map 1:1 to the course rubric:

1. Dataset description
2. Recommender system type
3. Methods/models
4. Interpreting results
5. Evaluation
6. Limitations and future improvement

# Project Proposal: Creator Intelligence Recommender System

**Team:** Samii Shabuse, Savit Tumuluri, Han Truong  
**Course:** DSCI351 — Recommender Systems  
**Status:** Phases 1–2 complete (`src/` pipeline + evaluation). Report and slides (Phase 3) in progress.

---

## Project Title

Creator Intelligence Recommender System for Instagram Influencers

---

## Problem Statement

Recommender systems help users make decisions in complex information environments by predicting relevant items or producing top-N ranked recommendations. For this project, **users are influencers** and **items are content strategies** (actionable posting patterns derived from metadata).

The system recommends what kind of content an influencer should create next based on their own history and patterns from similar influencers. This fits the course definition: the model predicts user preference and generates ranked recommendations.

**One-sentence task:** Given an influencer's past posts, rank content strategies by predicted engagement so the creator knows what to post next.

---

## Dataset

### Source

- **Dataset page:** https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset
- **Google Drive:** https://drive.google.com/drive/folders/1ISiSH4-aM6kP_0lKQYejpk1sa6Jei-7e
- **Creator:** Seungbae Kim — ksb2043@gmail.com — https://www.linkedin.com/in/seungbae-kim/

The full dataset is approximately 189 GB including images. We use **metadata only** for practicality.

### Components

| Component | Description | Local status |
|-----------|-------------|--------------|
| `influencers.txt` | 33,935 influencers; username, category, followers, followees, posts | Available locally |
| `JSON-Image_files_mapping.txt` | ~10M rows linking influencers to metadata and image files | Available locally |
| `clean_json_image_mapping.parquet` | Deduplicated mapping (one row per metadata file) | Available locally |
| `Post_metadata/posts_info.zip` | Compressed post JSON metadata | Google Drive (Colab) |
| Post images | JPG files | Not used in this project |

### Dataset Statistics

- **Followers:** 1,000 to 96,476,007 (avg ~140,329) — highly skewed
- **Posts per influencer:** 100 to 127,520 (avg ~1,487)
- **Categories (11):** fashion (11,911), other (5,720), travel (4,210), family (4,070), food (3,565), beauty, interior, fitness, pets, and one typo (`fasion`)

### Working Subset for Modeling

Because full metadata extraction is resource-intensive, we sample **10,000 posts** (random seed 42) from `posts_info.zip` via selective 7zip extraction in Google Colab. This subset supports fast iteration while preserving diversity across influencers and categories.

---

## Recommendation Target (Locked)

We recommend **content strategies**, not individual posts. Each strategy combines five interpretable dimensions:

| Dimension | Buckets |
|-----------|---------|
| Time of day | morning, afternoon, evening, night, unknown |
| Caption length | short (<80), medium (80–200), long (>200) |
| Hashtags | none, few (1–3), many (>3) |
| Sponsored | ad, not_ad |
| Media type | image, video |

**Strategy string example:**  
`evening + medium_caption + few_hashtags + not_ad + image`

This gives creators concrete guidance (when to post, caption style, hashtag use, format) while keeping the item space finite enough for collaborative filtering.

---

## System Design

### Inputs

- Influencer profile (category, followers)
- Post captions and hashtags
- Likes, comments, timestamps
- Sponsorship and video flags

### Outputs

- Top-N ranked content strategies for a given influencer
- Predicted engagement scores per strategy
- Side-by-side comparison across model types

### Models

| # | Model | Description |
|---|-------|-------------|
| 1 | **Global popularity baseline** | Top strategies by average engagement across all posts |
| 2 | **Category baseline** | Top strategies within the influencer's category |
| 3 | **User-based collaborative filtering** | Cosine similarity between influencers on strategy engagement vectors; recommend strategies successful for similar users |
| 4 | **Content-based** | TF-IDF similarity on captions; recommend strategies from semantically similar high-performing posts |
| 5 | **Hybrid** | Weighted combination of CF and content-based scores |

Implementation status: all five models implemented in `src/` and evaluated via `scripts/run_pipeline.py`. Scale benchmarks at 10k / 20k / 50k saved under `artifacts/runs/` (see [How_It_Works.md](How_It_Works.md)).

---

## Engagement and Pseudo-Ratings

Instagram provides implicit feedback (likes, comments), not star ratings.

**Raw engagement rate:**

```
(likes + 2 × comments) / followers
```

Comments are weighted 2× because they indicate stronger interaction than likes.

**Modeling score (used in interaction matrix):**

```
log_engagement_score = log1p(likes + 2×comments) / log1p(followers)
```

**Pseudo-ratings (1–5) for optional rating prediction metrics:**  
Within each influencer's posts, rank by engagement rate into quintiles: top 20% → 5, next 20% → 4, etc.

**User-item matrix:** rows = influencers, columns = content strategies, values = mean `log_engagement_score` (or pseudo-rating) for that pair.

---

## Evaluation Plan

### Split Strategy

**Time-based holdout:** For each influencer with enough posts, the most recent 20% of posts form the test set; older posts are used for training. This mimics real deployment (recommend before the next post).

### Metrics

| Metric | Use |
|--------|-----|
| **Precision@K, Recall@K, NDCG@K** | Primary — do recommended strategies match high-engagement strategies in the test set? |
| **Hit rate@K** | At least one recommended strategy appears in test relevant set |
| **MAE / RMSE** | Secondary — pseudo-rating prediction accuracy |

**Relevance definition:** A strategy is relevant in test if it appears in the influencer's top engagement quintile on held-out posts.

### Model Comparison

Report a single table comparing all five models on the same split. Include at least two qualitative case studies (influencer history + recommendations + whether they align with test performance).

---

## Implementation Approach

### Environment

- **Google Colab + Google Drive** for large data and GPU/RAM
- **GitHub repo** for versioned code and documentation
- See [Colab_Setup.md](Colab_Setup.md) and [Milestones.md](Milestones.md)

### Repository Structure (Target)

```
src/
  data_import.py      # Load influencers, mapping (exists)
  preprocess.py       # Parse metadata, build strategies
  baselines.py        # Global + category recommenders
  collaborative.py    # User-based CF
  content_based.py    # TF-IDF recommender
  hybrid.py           # Weighted hybrid
  evaluation.py       # Splits and metrics
notebooks/
  Recommender Pipeline and Model Development.ipynb
artifacts/            # Processed data and results (gitignored)
docs/                 # Proposal, milestones, submission guide
```

### Current Progress

| Milestone | Status |
|-----------|--------|
| Data access and mapping cleanup | Done |
| Metadata parsing (10k sample) | Done (`src/preprocess.py` + Colab) |
| Strategy features + engagement scores | Done (`src/preprocess.py`) |
| Baselines + user-based CF | Done (`src/baselines.py`, `src/collaborative.py`) |
| Content-based + hybrid | Done (`src/content_based.py`, `src/hybrid.py`) |
| Formal evaluation metrics | Done (`artifacts/runs/n*/results/model_comparison.csv`) |
| Scale benchmarks (10k / 20k / 50k) | Done (`artifacts/comparisons/`) |
| Report + slides | To do (Phase 3) |

---

## Limitations (Anticipated)

- 10k-post sample may not represent full dataset distribution
- Rule-based strategy buckets may miss nuanced content themes
- Sparse matrix for influencers with few posts limits CF
- No image or video content analysis
- Engagement correlates with success but does not prove causation
- Platform algorithm changes over time are not modeled

## Future Improvements

- Scale to 50k–100k posts; matrix factorization (SVD/ALS)
- Learned strategy clusters (K-means on caption embeddings)
- Image embeddings (CLIP) for visual content signal
- Item-based CF and neural collaborative filtering
- Online A/B testing framework (outside scope of course project)

---

## Planning Documents

- Execution roadmap and remaining tasks: [Milestones.md](Milestones.md)
- Final submission checklist and report outline: [Final_Submission.md](Final_Submission.md)
- Pipeline documentation: [Pipeline.md](Pipeline.md)
- Team guide (how it works): [How_It_Works.md](How_It_Works.md)

---

## Project Goal

1. Analyze influencer content and engagement data
2. Identify patterns in high-performing posts
3. Build and compare multiple recommender approaches
4. Evaluate with ranking metrics on a realistic time-based split
5. Deliver a clear 3-page report, slides, and reproducible repository

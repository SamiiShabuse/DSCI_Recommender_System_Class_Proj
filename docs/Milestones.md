# Project Execution Plan: Creator Intelligence Recommender System

This plan reflects the final repo state for the DSCI351 project submission.

## Current Status Snapshot

| Area | Status | Evidence |
|------|--------|----------|
| Problem definition | **Done** | Proposal + locked strategy label |
| Influencer data loaded | **Done** | `data/influencers.txt` (33,935 rows) |
| Mapping data cleaned | **Done** | `data/clean_json_image_mapping.parquet` |
| Metadata parsing pipeline | **Done** | `src/data/preprocess.py` + Colab notebooks |
| Feature engineering | **Done** | `src/data/preprocess.py` |
| Global + category baselines | **Done** | `src/models/baselines.py` |
| User-based CF | **Done** | `src/models/collaborative.py` |
| Content-based recommender | **Done** | `src/models/content_based.py` |
| Hybrid model | **Done** | `src/models/hybrid.py` |
| Train/test split + metrics | **Done** | `src/evaluation.py` → `artifacts/runs/n*/results/` |
| Scale benchmarks (10k / 20k / 50k) | **Done** | `artifacts/comparisons/` |
| Reproducible pipeline | **Done** | `scripts/run_pipeline.py`, `run_scale_benchmark.py` |
| Team guide | **Done** | `docs/How_It_Works.md` |
| Metrics + figures | **Done** | Per-run under `artifacts/runs/n*/` |
| Final report (PDF) | **Waived** | Professor said a separate final report is no longer required |
| Presentation slides | **Done** | `docs/Final Presentation.pptx` |

**Bottom line:** Code, metrics, artifacts, docs, and slides are ready. Final action is Canvas upload.

---

## Locked Technical Decisions

These are no longer open questions — use them consistently in code, report, and slides.

1. **Recommendation unit:** Content strategy string  
   `{time_bucket} + {caption_bucket} + {hashtag_bucket} + {ad_bucket} + {media_bucket}`

2. **Working dataset size:** 10,000 posts for the primary run, with additional 20k, 50k, and 100k scale benchmarks.

3. **Engagement score (training signal):**  
   `log_engagement_score = log1p(likes + 2×comments) / log1p(followers)`

4. **Evaluation labels:** Per-influencer quintile pseudo-ratings (1-5) from engagement rate; rating 5 is treated as relevant for top-N evaluation.

5. **Evaluation split:** Time-based — most recent 20% of each influencer's posts held out for test.

6. **Primary ranking metrics:** Precision@5, Recall@5, NDCG@5, and Hit Rate@5.

7. **Environment:** Google Colab + Google Drive for data; repo cloned into Drive for team access.

---

## Remaining Work Phases

### Phase 1: Reproducibility + Port to `src/` — COMPLETE

**Goal:** One notebook or script run produces the same outputs every time.

Tasks:

- [x] Expand `requirements.txt` (pandas, numpy, scikit-learn, pyarrow, tqdm, matplotlib).
- [x] Port notebook functions into modules:
  - `src/data/preprocess.py` — parse metadata, build strategy labels, engagement scores
  - `src/models/baselines.py` — global and category recommenders
  - `src/models/collaborative.py` — interaction matrix + user-based CF
  - `src/models/content_based.py` — TF-IDF caption → strategy recommendations
  - `src/models/hybrid.py` — weighted combination
  - `src/evaluation.py` — time split, Precision@K, Recall@K, NDCG@K
- [x] Consolidate notebooks: `notebooks/Recommender Pipeline and Model Development.ipynb` (legacy notebooks kept as the development audit trail).
- [x] Document pipeline in `docs/Pipeline.md` and updated README.
- [x] Save processed outputs to `artifacts/processed/` (gitignored).

**Exit criteria:** `python scripts/run_pipeline.py --synthetic` → metrics table without manual edits.

---

### Phase 2: Missing Models + Evaluation — COMPLETE

**Goal:** Compare all five approaches with held-out metrics.

Tasks:

- [x] Implement **content-based** recommender (TF-IDF on captions).
- [x] Implement **hybrid** with tuned alpha (`src/models/hybrid.py`).
- [x] Build time-based train/test split per influencer.
- [x] Define relevant test items as pseudo-rating ≥ 5.
- [x] Run comparison table: Global | Category | CF | Content-based | Hybrid.
- [x] Save per-scale results under `artifacts/runs/n{target}/` and comparison under `artifacts/comparisons/`.

**Exit criteria:** Metrics reproducible via `scripts/run_pipeline.py`.

**Team guide:** [How_It_Works.md](How_It_Works.md) explains the process for teammates.

---

### Phase 3: Slides + Submission Narrative — COMPLETE

**Goal:** Submit-ready narrative aligned with rubric.

Tasks:

- [x] Create slides with pipeline diagram, metrics table, and 2 demo influencers.
- [x] Add limitations and future work.
- [x] Document that the separate final report was waived.

**Exit criteria:** Another student can explain problem, method, and results in 5 minutes using slides alone.

---

### Phase 4: Polish + Submission

**Goal:** Professional repo and complete Canvas upload.

Tasks:

- [x] Update root README: problem, setup, how to reproduce, dataset links, team names.
- [x] Final QA: all rubric sections covered, figures labeled, citations included.
- [ ] Upload repo link, slides, and dataset link to Canvas.

**Exit criteria:** Canvas submission complete.

---

## Suggested Team Split

| Member | Primary ownership |
|--------|-------------------|
| Samii | Pipeline port, Colab infra, notebook consolidation |
| Savit | Content-based + hybrid models, hyperparameter tuning |
| Han | Evaluation module, metrics tables, slides + figures |

Rotate review: each PR/notebook section gets a second pair of eyes before merge.

---

## Weekly Rhythm (Until Due Date)

- **Monday (30 min):** Pick Phase tasks; assign owners; note blockers.
- **Mid-week:** Push code/notebook updates; log metrics in a shared sheet.
- **Friday (30 min):** Demo what runs end-to-end; update status table above.

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Colab session crash on 10k parse | Batch parquet writes (already in notebook); lower SAMPLE_SIZE to 5k if needed |
| Sparse CF matrix | Fall back to category baseline for cold-start influencers; mention in limitations |
| Hybrid not beating baselines | Report trade-offs honestly; category baseline may win on sparse data — still valid analysis |
| Canvas expectations differ from updated professor guidance | Note that the separate report was waived and submit repo/slides/dataset links |
| Drive path differs per teammate | Document canonical path in Colab_Setup.md; use `DATA_PATH` variable at top of notebook |

---

## Original 8-Week Plan (Reference)

The original week-by-week plan assumed starting from scratch. Mapped to today:

| Original week | Actual status |
|---------------|---------------|
| Week 1: Scope + data access | **Complete** |
| Week 2: Data pipeline | **Complete** (notebooks + parquet mapping) |
| Week 3: Features + EDA | **Complete** (in notebook) |
| Week 4: Baselines | **Complete** (in notebook) |
| Week 5: Collaborative filtering | **Complete** (in notebook) |
| Week 6: Hybrid + ablations | **Complete** (hybrid + scale comparison) |
| Week 7: Final evaluation + presentation | **Complete** |
| Week 8: Polish + submission | **Ready for Canvas upload** |

Focus remaining effort on Canvas upload.

## Related Docs

- [How_It_Works.md](How_It_Works.md) — teammate guide: what the pipeline does and what 10k/20k/50k runs mean
- [Pipeline.md](Pipeline.md) — technical commands and module reference

---

## High-Score Checklist

- [ ] Problem statement is precise and measurable
- [ ] Time-aware evaluation split
- [ ] Metrics match recommendation objective (ranking, not just MAE)
- [ ] Strong baselines included (global + category)
- [ ] Every modeling choice supported by metric or ablation
- [x] Limitations and future work in slides
- [x] Repository clean, documented, and runnable from README

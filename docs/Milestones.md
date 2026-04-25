# 8-Week Milestone Plan: Creator Intelligence Recommender System

This plan is designed to maximize project quality, technical depth, and presentation quality for a top grade and strong portfolio value.

## Success Targets

- Deliver a reproducible recommendation pipeline from raw metadata to ranked recommendations.
- Implement and compare three recommenders: content-based, collaborative filtering, and hybrid.
- Report results with clear metrics, ablations, and error analysis.
- Produce polished final artifacts: report, slides, demo, and clean repository.

## Working Rhythm (Every Week)

- Team sync (30 to 45 minutes): Monday planning and Friday review.
- Deliverable check: one concrete output committed by end of week.
- Experiment logging: record parameters, metrics, and observations.
- Risk check: identify one blocker and one mitigation each week.

## Week 1: Scope Lock + Data Access

Goals:

- Freeze problem definition, recommendation target, and evaluation protocol.
- Finalize dataset subset and local data access plan.

Tasks:

- Decide exact recommendation unit (post strategy label, style cluster, or post template).
- Decide evaluation split strategy (time-based preferred for realistic recommendation).
- Build data inventory of available metadata fields and missingness.
- Define baseline assumptions and project constraints.

Deliverables:

- Final project proposal document.
- Data inventory table and target-definition note.

Exit Criteria:

- Team can describe the task in one sentence and name exactly what a recommendation output looks like.

## Week 2: Data Pipeline + Cleaning

Goals:

- Build robust and repeatable preprocessing for influencer and post metadata.

Tasks:

- Parse influencer and post metadata files into tabular format.
- Standardize timestamps, text fields, and engagement fields.
- Handle missing values and obvious outliers.
- Create versioned processed dataset snapshots.
- Set up Google Colab + Drive data workflow for large-scale preprocessing.

Deliverables:

- Reproducible preprocessing script(s).
- Data quality report (row counts, null rates, key distributions).
- Colab runbook with exact commands for setup and subset generation.

Exit Criteria:

- A single command can regenerate a clean analysis-ready dataset.

## Week 3: Feature Engineering + EDA

Goals:

- Extract features needed by all recommender variants.

Tasks:

- Build text features from captions and hashtags (TF-IDF or embeddings).
- Create engagement-normalized labels (for example engagement rate).
- Engineer time/context features (weekday, posting time bucket, sponsorship flag).
- Segment influencers by category/scale to support fair comparisons.

Deliverables:

- Feature dictionary document.
- EDA notebook/charts with insights that motivate modeling choices.

Exit Criteria:

- Feature matrix and interaction matrix are ready for model training.

## Week 4: Baseline Recommenders

Goals:

- Set performance floor with simple, interpretable baselines.

Tasks:

- Implement popularity and recent-performance baselines.
- Implement a first content-based recommender.
- Build Top-K recommendation generation function.

Deliverables:

- Baseline metrics table.
- First recommendation examples for several influencer profiles.

Exit Criteria:

- Baselines run end-to-end on held-out data and produce reproducible metrics.

## Week 5: Collaborative Filtering Model

Goals:

- Add cross-influencer learning signal.

Tasks:

- Build influencer-content interaction matrix.
- Implement collaborative filtering model (matrix factorization or nearest-neighbor CF).
- Tune key hyperparameters with validation split.

Deliverables:

- CF model training/evaluation scripts.
- Validation results and hyperparameter summary.

Exit Criteria:

- CF beats at least one baseline on primary ranking metric.

## Week 6: Hybrid Model + Ablations

Goals:

- Combine content and collaborative signals into a stronger model.

Tasks:

- Implement weighted or stacked hybrid recommender.
- Run ablation study to quantify contribution of each component.
- Evaluate by influencer category and influencer size buckets.

Deliverables:

- Hybrid model outputs and comparison charts.
- Ablation table with concise interpretation.

Exit Criteria:

- Hybrid is best overall or justified with strong trade-off reasoning.

## Week 7: Final Evaluation + Storytelling

Goals:

- Convert technical results into clear project narrative.

Tasks:

- Run final metric suite: Precision@K, Recall@K, NDCG@K, plus engagement-oriented outcomes.
- Add qualitative examples: good recommendations and failure cases.
- Write error analysis and limitations section.
- Draft report sections and slide deck structure.

Deliverables:

- Final results tables/figures.
- Draft final report and draft presentation slides.

Exit Criteria:

- Another student can understand problem, method, and value in 5 minutes.

## Week 8: Polish + Demo + Submission

Goals:

- Ship a professional, complete final package.

Tasks:

- Refactor code for readability and reproducibility.
- Add README quickstart and experiment reproduction instructions.
- Record or rehearse short demo with representative examples.
- Final QA pass on report, citations, figures, and rubric coverage.

Deliverables:

- Final code, report, slides, and demo-ready outputs.
- Submission checklist completed.

Exit Criteria:

- Project is reproducible, defensible, and presentation-ready.

## High-Score Checklist (Use Weekly)

- Problem statement is precise and measurable.
- Evaluation protocol is realistic (time-aware split preferred).
- Metrics match recommendation objective.
- Baselines are strong enough to make improvements meaningful.
- Every major modeling choice has evidence (plot, metric, or ablation).
- Report includes limitations, ethics/privacy note, and future work.
- Repository is clean, documented, and runnable.

## Portfolio/Recruiting Boost (Google-Ready Signal)

- Keep architecture diagram simple and professional.
- Include one-page technical summary with quantified gains.
- Emphasize reproducibility, experiment rigor, and error analysis.
- Show practical impact: how recommendations change creator strategy decisions.
